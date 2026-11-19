"""Train PPO (Stable-Baselines3) on a Walking Hand env.

Examples (run from the repo root):

    # 2-minute smoke test
    python scripts/train_ppo.py --env WalkingHand-v0 --total-steps 50000 --n-envs 4

    # a real run: tactile policy, seed 1, friction randomization and pushes
    python scripts/train_ppo.py --env WalkingHandTactile-v1 --seed 1 --total-steps 5000000 \
        --friction-range 0.3 1.5 --push-range 0.2 0.8

    # change reward weights (any ThingHandWalkEnv constructor argument works)
    python scripts/train_ppo.py --env WalkingHand-v1 --env-kwarg alive_bonus=0.2

Each run writes runs/<name>/ with:
    config.json          env id, env kwargs, PPO settings, seed, git commit, library versions
    final_model.zip      policy after the last update
    vecnormalize.pkl     observation/reward normalization statistics (needed to run the policy)
    best/                best model by the periodic deterministic evaluation
    checkpoints/         periodic snapshots (model + vecnormalize)
    eval/evaluations.npz SB3 evaluation history
    tb/                  TensorBoard logs (tensorboard --logdir runs)

Pure RL vs tactile RL: train both with identical arguments except --env.
run_matrix.py does that for several seeds.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import rl_common  # noqa: F401  (puts the repo root on sys.path; must come first)
from rl_common import REPO_ROOT, parse_kv, provenance, save_json, to_tuple_ranges

# Module level on purpose: Windows worker processes re-import this file (not main()),
# and they need the env ids registered before gym.make runs inside them.
import walking_hand  # noqa: F401,E402


def build_parser():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--env", default="WalkingHand-v0", help="gym id, e.g. WalkingHand-v1 or WalkingHandTactile-v1")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--total-steps", type=int, default=2_000_000, help="environment steps summed over all envs")
    ap.add_argument("--n-envs", type=int, default=8, help="parallel envs (one process each unless --dummy-vec)")
    ap.add_argument("--dummy-vec", action="store_true", help="run all envs in the main process (debugging)")
    ap.add_argument("--out", default=str(REPO_ROOT / "runs"), help="parent folder for run directories")
    ap.add_argument("--name", default=None, help="run folder name (default: <env>_s<seed>_<time>)")

    g = ap.add_argument_group("environment (identical for pure and tactile RL)")
    g.add_argument("--friction-range", type=float, nargs=2, default=None, metavar=("LO", "HI"),
                   help="randomize floor friction each reset")
    g.add_argument("--push-range", type=float, nargs=2, default=None, metavar=("LO", "HI"),
                   help="random horizontal pushes on the palm, newtons")
    g.add_argument("--env-kwarg", action="append", metavar="KEY=VALUE",
                   help="any other ThingHandWalkEnv argument, e.g. alive_bonus=0.2 (repeatable)")

    p = ap.add_argument_group("PPO")
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--n-steps", type=int, default=1024, help="rollout length per env")
    p.add_argument("--batch-size", type=int, default=4096)
    p.add_argument("--n-epochs", type=int, default=10)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--gae-lambda", type=float, default=0.95)
    p.add_argument("--clip-range", type=float, default=0.2)
    p.add_argument("--ent-coef", type=float, default=0.0)
    p.add_argument("--vf-coef", type=float, default=0.5)
    p.add_argument("--max-grad-norm", type=float, default=0.5)
    p.add_argument("--target-kl", type=float, default=None)
    p.add_argument("--log-std-init", type=float, default=-1.0,
                   help="initial log std of the Gaussian policy; -1 gives std ~0.37 on [-1, 1] actions")
    p.add_argument("--net", type=int, nargs="+", default=[256, 256], help="hidden sizes for policy and value nets")
    p.add_argument("--no-norm-reward", action="store_true", help="normalize observations only")
    p.add_argument("--device", default="cpu", help="cpu is usually fastest for small MLPs")
    p.add_argument("--torch-threads", type=int, default=1,
                   help="threads for the PPO update in the main process (1 avoids fighting the env workers)")

    l = ap.add_argument_group("logging")
    l.add_argument("--eval-every", type=int, default=100_000, help="env steps between evaluations (0 = off)")
    l.add_argument("--eval-episodes", type=int, default=5)
    l.add_argument("--checkpoint-every", type=int, default=500_000, help="env steps between checkpoints (0 = off)")
    return ap


def env_kwargs_from_args(args) -> dict:
    kw = parse_kv(args.env_kwarg)
    if args.friction_range is not None:
        kw["friction_range"] = tuple(args.friction_range)
    if args.push_range is not None:
        kw["push_force_range"] = tuple(args.push_range)
    return kw


def main():
    args = build_parser().parse_args()

    import numpy as np
    import torch
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import BaseCallback, CallbackList, CheckpointCallback, EvalCallback
    from stable_baselines3.common.env_util import make_vec_env
    from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize

    class WalkInfoCallback(BaseCallback):
        """Log walking quantities from the env's info dict to TensorBoard."""

        def _on_training_start(self):
            self.start_x = [None] * self.training_env.num_envs

        def _on_rollout_start(self):
            self.buf = {"x_velocity": [], "palm_height": [], "upright": [], "touch_any": []}

        def _on_step(self):
            for i, (info, done) in enumerate(zip(self.locals["infos"], self.locals["dones"])):
                x = info.get("x_position", np.nan)
                if self.start_x[i] is None:  # first step of an episode (off by one step, fine for logging)
                    self.start_x[i] = x
                self.buf["x_velocity"].append(info.get("x_velocity", np.nan))
                self.buf["palm_height"].append(info.get("palm_height", np.nan))
                self.buf["upright"].append(info.get("upright", np.nan))
                touch = info.get("touch")
                if touch is not None:
                    self.buf["touch_any"].append(float(np.any(np.asarray(touch) > 0)))
                if done:  # infos of a done step still describe the terminal state (before auto-reset)
                    self.logger.record_mean("walk/episode_distance", x - self.start_x[i])
                    self.logger.record_mean("walk/fell", float(not info.get("TimeLimit.truncated", False)))
                    self.start_x[i] = None
            return True

        def _on_rollout_end(self):
            for k, v in self.buf.items():
                if v:
                    self.logger.record(f"walk/{k}", float(np.nanmean(v)))

    env_kwargs = env_kwargs_from_args(args)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    name = args.name or f"{args.env}_s{args.seed}_{stamp}"
    run_dir = Path(args.out) / name
    run_dir.mkdir(parents=True, exist_ok=False)

    torch.set_num_threads(args.torch_threads)  # many env processes + one small MLP: avoid oversubscription
    vec_cls = DummyVecEnv if (args.dummy_vec or args.n_envs == 1) else SubprocVecEnv
    venv = make_vec_env(args.env, n_envs=args.n_envs, seed=args.seed, vec_env_cls=vec_cls,
                        env_kwargs=to_tuple_ranges(env_kwargs))
    venv = VecNormalize(venv, norm_obs=True, norm_reward=not args.no_norm_reward, clip_obs=10.0, gamma=args.gamma)

    callbacks = [WalkInfoCallback()]
    if args.eval_every > 0:
        eval_env = make_vec_env(args.env, n_envs=1, seed=args.seed + 10_000, vec_env_cls=DummyVecEnv,
                                env_kwargs=to_tuple_ranges(env_kwargs))
        eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=False, clip_obs=10.0, training=False)
        callbacks.append(EvalCallback(
            eval_env,
            n_eval_episodes=args.eval_episodes,
            eval_freq=max(args.eval_every // args.n_envs, 1),
            best_model_save_path=str(run_dir / "best"),
            log_path=str(run_dir / "eval"),
            deterministic=True,
            callback_on_new_best=save_vecnormalize_cb(run_dir / "best" / "vecnormalize.pkl"),
        ))
    if args.checkpoint_every > 0:
        callbacks.append(CheckpointCallback(
            save_freq=max(args.checkpoint_every // args.n_envs, 1),
            save_path=str(run_dir / "checkpoints"),
            name_prefix="rl_model",
            save_vecnormalize=True,
        ))

    rollout = args.n_steps * args.n_envs
    batch_size = min(args.batch_size, rollout)
    if batch_size != args.batch_size:
        print(f"batch_size {args.batch_size} > rollout size {rollout} (n_steps x n_envs); using {batch_size}.")
    ppo_kwargs = dict(
        learning_rate=args.lr, n_steps=args.n_steps, batch_size=batch_size, n_epochs=args.n_epochs,
        gamma=args.gamma, gae_lambda=args.gae_lambda, clip_range=args.clip_range, ent_coef=args.ent_coef,
        vf_coef=args.vf_coef, max_grad_norm=args.max_grad_norm, target_kl=args.target_kl,
        policy_kwargs=dict(net_arch=dict(pi=list(args.net), vf=list(args.net)), log_std_init=args.log_std_init),
    )
    config = {
        "env_id": args.env,
        "env_kwargs": env_kwargs,
        "seed": args.seed,
        "total_steps": args.total_steps,
        "n_envs": args.n_envs,
        "norm_reward": not args.no_norm_reward,
        "ppo": ppo_kwargs,
        "command_args": vars(args),
        **provenance(),
    }
    save_json(run_dir / "config.json", config)
    if config["git"]["dirty"]:
        print("WARNING: uncommitted changes in tracked files; results are tied to commit", config["git"]["commit"][:8],
              "plus local edits.")

    model = PPO("MlpPolicy", venv, seed=args.seed, device=args.device, verbose=1,
                tensorboard_log=str(run_dir / "tb"), **ppo_kwargs)
    print(f"Run folder: {run_dir}")
    print(f"env={args.env} kwargs={env_kwargs} n_envs={args.n_envs} total_steps={args.total_steps}")
    t0 = time.time()
    try:
        model.learn(total_timesteps=args.total_steps, callback=CallbackList(callbacks), progress_bar=False)
    except KeyboardInterrupt:
        print("Interrupted; saving the current model.")
    finally:
        # Ctrl+C on Windows also reaches the worker processes, so closing them may fail;
        # save the model and normalization statistics first and never let close() hide them.
        model.save(str(run_dir / "final_model"))
        venv.save(str(run_dir / "vecnormalize.pkl"))
        try:
            venv.close()
        except Exception as exc:  # noqa: BLE001
            print(f"(ignored error while closing env workers: {exc!r})")
    config["train_seconds"] = round(time.time() - t0, 1)
    config["steps_done"] = int(model.num_timesteps)
    save_json(run_dir / "config.json", config)
    print(f"Done in {config['train_seconds']:.0f} s, {model.num_timesteps} steps. Saved to {run_dir}")
    print(f"Evaluate with: python scripts/eval_policy.py --run {run_dir}")


def save_vecnormalize_cb(path: Path):
    """EvalCallback hook: save the training VecNormalize next to each new best model."""
    from stable_baselines3.common.callbacks import BaseCallback

    class Cb(BaseCallback):
        def _on_step(self):
            vn = self.model.get_vec_normalize_env()
            if vn is not None:
                path.parent.mkdir(parents=True, exist_ok=True)
                vn.save(str(path))
            return True

    return Cb()


if __name__ == "__main__":  # required on Windows: SubprocVecEnv spawns workers that re-import this file
    main()
