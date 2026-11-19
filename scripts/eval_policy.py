"""Evaluate a controller on fixed test conditions and write CSVs, plots and videos.

Works for a trained PPO run or any Python controller, so all three controllers
(baseline, pure RL, tactile RL) are measured by the same code.

Examples (run from the repo root):

    # a PPO run: env id and env settings come from runs/<name>/config.json
    python scripts/eval_policy.py --run runs/WalkingHand-v1_s0_20261119-080000

    # quick look: 5 episodes, nominal condition only, 2 videos
    python scripts/eval_policy.py --run runs/<name> --conditions nominal --episodes 5 --video 2

    # reference controllers (no training needed)
    python scripts/eval_policy.py --controller hold --env WalkingHand-v1
    python scripts/eval_policy.py --controller random --env WalkingHand-v1

    # any Python controller: module:factory, factory(env) returns an object with
    # reset(env) and act(obs, env) -> action in [-1, 1]^6
    python scripts/eval_policy.py --controller my_module:make_controller --env WalkingHand-v1

    # Nicole's classical baseline (needs walking_hand/baseline from train/baseline-balance)
    python scripts/eval_policy.py --controller baseline-walk --env WalkingHand-v1

Conditions (--conditions, default: all three):
    nominal    scene friction, no pushes
    friction   floor friction in FRICTIONS, including values outside the training range
    push       pushes of fixed magnitude PUSHES (N) every 1-3 s for 0.1 s, nominal friction

Every condition value uses the same episode seeds (--seed-base + 0..N-1), so
controllers are compared on identical initial states and push timings.

Outputs in --out (default: <run>/eval or runs/eval_<controller>):
    episodes.csv    one row per episode with every metric
    summary.csv     mean / std / 95% CI per condition value
    pushes.csv      one row per push (recovery analysis)
    eval_config.json
    *.png           plots;  videos/*.mp4 if --video
"""
from __future__ import annotations

import argparse
import csv
import importlib
import math
import time
from pathlib import Path

import rl_common  # noqa: F401  (puts the repo root on sys.path; must come first)
from rl_common import (REPO_ROOT, HoldPolicy, PPOPolicy, RandomPolicy, find_model, load_json, make_env,
                       provenance, save_json)

FRICTIONS = [0.1, 0.2, 0.3, 0.5, 0.8, 1.0, 1.5, 2.0]
PUSHES = [0.2, 0.4, 0.6, 0.8, 1.0, 1.2]
TIP_SITES = ["index_tip", "middle_tip"]
# Env arguments that evaluation sets itself; a run's training randomization is removed.
EVAL_OWNED = {"friction_range", "push_force_range", "render_mode"}

# Recovery after a push
RECOVERY_WINDOW_S = 2.0      # must not fall within this long after the push ends
SPEED_WINDOW_S = 0.5         # moving-average window for "back to nominal speed"
PRE_PUSH_WINDOW_S = 1.0      # speed reference before the push
SPEED_RECOVERY_FRAC = 0.8    # recovered when the moving average reaches this fraction of the reference
MIN_REF_SPEED = 0.002        # m/s; below this the reference speed is too small to define speed recovery
CONTACT_THRESHOLD = 1e-3     # N; a fingertip counts as in contact above this touch force


# --------------------------------------------------------------------------- controllers
def build_controller(args):
    """Return (controller, env_id, base_env_kwargs, label)."""
    if args.run:
        run = Path(args.run)
        cfg = load_json(run / "config.json")
        model_path, vn_path = find_model(run, args.model)
        if not model_path.exists():
            raise SystemExit(f"model not found: {model_path}")
        if vn_path is None:
            print("WARNING: no vecnormalize.pkl found; running the policy on raw observations.")
        ctrl = PPOPolicy(model_path, vn_path, deterministic=not args.stochastic)
        env_id = args.env or cfg["env_id"]
        if args.env and args.env != cfg["env_id"]:
            print(f"WARNING: evaluating on {args.env} but the policy was trained on {cfg['env_id']}.")
        return ctrl, env_id, dict(cfg.get("env_kwargs", {})), f"ppo:{run.name}:{model_path.stem}"
    if not args.env:
        raise SystemExit("--env is required with --controller")
    name = args.controller
    if name == "hold":
        return HoldPolicy(), args.env, {}, "hold"
    if name == "random":
        return RandomPolicy(), args.env, {}, "random"
    if name in BASELINES:
        return _FactoryController(BASELINES[name]), args.env, {}, name
    if ":" not in name:
        raise SystemExit(f"--controller must be hold, random, {', '.join(BASELINES)} or module:factory")
    mod, fn = name.split(":", 1)
    factory = getattr(importlib.import_module(mod), fn)
    return _FactoryController(factory), args.env, {}, name


def _baseline_walk(env):
    # Nicole's classical walk (train/baseline-balance). Imported lazily: absent on main until that PR merges.
    from walking_hand.baseline import WalkController

    u = env.unwrapped
    return WalkController(u.model, u.data, dt=u.dt)


def _baseline_stand(env):
    from walking_hand.baseline import BalanceHoldController

    return BalanceHoldController()


BASELINES = {"baseline-walk": _baseline_walk, "baseline-stand": _baseline_stand}


class _FactoryController:
    """Adapter for factory controllers; the factory is called with the env at each reset.

    The object it returns may expose act(obs, env), act(obs), or be callable as
    policy(obs) (the style used by walking_hand.baseline); reset() / reset(env) is optional.
    """

    def __init__(self, factory):
        self.factory = factory
        self.inner = None

    def reset(self, env):
        self.inner = self.factory(env)
        if hasattr(self.inner, "reset"):
            try:
                self.inner.reset(env)
            except TypeError:
                self.inner.reset()
        if hasattr(self.inner, "act"):
            import inspect

            n = len(inspect.signature(self.inner.act).parameters)
            self._call = (lambda o, e: self.inner.act(o, e)) if n >= 2 else (lambda o, e: self.inner.act(o))
        elif callable(self.inner):
            self._call = lambda o, e: self.inner(o)
        else:
            raise TypeError("controller must have act() or be callable")

    def act(self, obs, env=None):
        return self._call(obs, env)


# --------------------------------------------------------------------------- one episode
def run_episode(env, ctrl, seed, reset_options=None, frames=None):
    import numpy as np

    u = env.unwrapped
    m = u.model
    tip_ids = [m.site(s).id for s in TIP_SITES]

    obs, info = env.reset(seed=seed, options=reset_options or {})
    ctrl.reset(env)
    dt = u.dt
    x0 = info["x_position"]
    xs, vs, heights, uprights, pushing = [x0], [], [], [], []
    slip = np.zeros(len(tip_ids))
    contact_steps = np.zeros(len(tip_ids))
    prev_tip = np.array([u.data.site_xpos[i][:2].copy() for i in tip_ids])
    prev_contact = np.asarray(info["touch"]) > CONTACT_THRESHOLD
    terminated = truncated = False
    if frames is not None:
        frames.append(env.render())

    while not (terminated or truncated):
        action = ctrl.act(obs, env)
        obs, _, terminated, truncated, info = env.step(action)
        tip = np.array([u.data.site_xpos[i][:2].copy() for i in tip_ids])
        contact = np.asarray(info["touch"]) > CONTACT_THRESHOLD
        both = contact & prev_contact  # in contact at both ends of the step: tangential motion is slip
        slip += np.where(both, np.linalg.norm(tip - prev_tip, axis=1), 0.0)
        contact_steps += contact
        prev_tip, prev_contact = tip, contact
        xs.append(info["x_position"])
        vs.append(info["x_velocity"])
        heights.append(info["palm_height"])
        uprights.append(info["upright"])
        pushing.append(bool(info.get("pushing", False)))
        if frames is not None:
            frames.append(env.render())

    steps = len(vs)
    distance = xs[-1] - x0
    ep = {
        "steps": steps,
        "duration_s": steps * dt,
        "fell": bool(terminated),
        "distance_m": distance,
        "mean_speed_mps": distance / (steps * dt) if steps else 0.0,
        "mean_palm_height_m": float(np.mean(heights)) if heights else math.nan,
        "min_upright": float(np.min(uprights)) if uprights else math.nan,
        "slip_index_m": float(slip[0]),
        "slip_middle_m": float(slip[1]),
        "slip_total_m": float(slip.sum()),
        "slip_per_m_walked": float(slip.sum() / distance) if distance > 1e-3 else math.nan,
        "contact_frac_index": float(contact_steps[0] / steps) if steps else 0.0,
        "contact_frac_middle": float(contact_steps[1] / steps) if steps else 0.0,
        "friction": float(info["friction"]),
    }
    pushes = analyse_pushes(np.array(vs), np.array(pushing, dtype=bool), dt, steps, bool(terminated))
    ep["n_pushes"] = len(pushes)
    ep["push_survival_rate"] = (float(np.mean([p["survived"] for p in pushes])) if pushes else math.nan)
    return ep, pushes


def analyse_pushes(vs, pushing, dt, steps, fell):
    """One record per push: did the hand survive it, and how long until it was back to its pre-push speed."""
    import numpy as np

    out = []
    if len(pushing) == 0:
        return out
    starts = np.flatnonzero(pushing & ~np.concatenate([[False], pushing[:-1]]))
    win = max(int(round(SPEED_WINDOW_S / dt)), 1)
    pre = max(int(round(PRE_PUSH_WINDOW_S / dt)), 1)
    rec = int(round(RECOVERY_WINDOW_S / dt))
    for s in starts:
        e = s
        while e < len(pushing) and pushing[e]:
            e += 1  # e = first step after the push
        if e + rec > steps and not fell:
            continue  # episode ended (time limit) before the recovery window closed: not scored
        survived = not (fell and steps <= e + rec)
        ref = float(np.mean(vs[max(s - pre, 0):s])) if s > 0 else math.nan
        t_rec = math.nan
        if survived and ref > MIN_REF_SPEED:
            for k in range(e, steps - win + 1):
                if np.mean(vs[k:k + win]) >= SPEED_RECOVERY_FRAC * ref:
                    t_rec = (k - e) * dt
                    break
        out.append({"push_step": int(s), "push_end_step": int(e), "survived": bool(survived),
                    "ref_speed_mps": ref, "time_to_recover_s": t_rec})
    return out


# --------------------------------------------------------------------------- conditions
def condition_list(names):
    conds = []
    for n in names:
        if n == "nominal":
            conds.append(("nominal", None, {}, {}))
        elif n == "friction":
            for f in FRICTIONS:
                conds.append(("friction", f, {}, {"friction": f}))
        elif n == "push":
            for p in PUSHES:
                conds.append(("push", p, {"push_force_range": (p, p)}, {}))
        else:
            raise SystemExit(f"unknown condition {n!r}")
    return conds


def ci95(x):
    import numpy as np

    x = np.asarray([v for v in x if not (isinstance(v, float) and math.isnan(v))], dtype=float)
    if len(x) < 2:
        return (float(x.mean()) if len(x) else math.nan), math.nan, math.nan
    return float(x.mean()), float(x.std(ddof=1)), float(1.96 * x.std(ddof=1) / math.sqrt(len(x)))


SUMMARY_METRICS = ["fell", "distance_m", "mean_speed_mps", "slip_total_m", "slip_per_m_walked",
                   "mean_palm_height_m", "push_survival_rate"]


def summarize(rows, push_rows):
    groups = {}
    for r in rows:
        groups.setdefault((r["condition"], r["value"]), []).append(r)
    out = []
    for (cond, val), rs in groups.items():
        s = {"controller": rs[0]["controller"], "condition": cond, "value": val, "episodes": len(rs)}
        for k in SUMMARY_METRICS:
            mean, std, ci = ci95([float(r[k]) for r in rs])
            name = "fall_rate" if k == "fell" else k
            s[f"{name}_mean"], s[f"{name}_std"], s[f"{name}_ci95"] = mean, std, ci
        prs = [p for p in push_rows if p["condition"] == cond and p["value"] == val]
        if prs:
            s["pushes_scored"] = len(prs)
            s["push_recovery_rate"] = sum(p["survived"] for p in prs) / len(prs)
            t = [p["time_to_recover_s"] for p in prs if not math.isnan(p["time_to_recover_s"])]
            s["speed_recovered_frac"] = len(t) / len(prs)
            s["time_to_recover_s_median"] = sorted(t)[len(t) // 2] if t else math.nan
        out.append(s)
    return out


def write_csv(path, rows):
    if not rows:
        Path(path).write_text("", encoding="utf-8")  # e.g. no push lasted long enough to be scored
        return
    keys = []
    for r in rows:
        keys += [k for k in r if k not in keys]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def plot(summary, out_dir, label):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    def series(cond, key):
        rs = sorted([s for s in summary if s["condition"] == cond], key=lambda s: s["value"])
        return ([s["value"] for s in rs], [s[f"{key}_mean"] for s in rs],
                [0 if math.isnan(s[f"{key}_ci95"]) else s[f"{key}_ci95"] for s in rs])

    panels = [("friction", "distance_m", "floor friction", "distance (m)"),
              ("friction", "fall_rate", "floor friction", "fall rate"),
              ("friction", "slip_per_m_walked", "floor friction", "slip per m walked"),
              ("push", "fall_rate", "push force (N)", "fall rate"),
              ("push", "distance_m", "push force (N)", "distance (m)")]
    panels = [p for p in panels if any(s["condition"] == p[0] for s in summary)]
    if not panels:
        return
    fig, axes = plt.subplots(1, len(panels), figsize=(4 * len(panels), 3.4), squeeze=False)
    for ax, (cond, key, xl, yl) in zip(axes[0], panels):
        x, y, e = series(cond, key)
        ax.errorbar(x, y, yerr=e, marker="o", capsize=3)
        ax.set_xlabel(xl)
        ax.set_ylabel(yl)
        if key == "fall_rate":
            ax.set_ylim(-0.05, 1.05)  # fixed scale so 0% and 100% are not auto-zoomed
        ax.grid(alpha=0.3)
    fig.suptitle(label, fontsize=9)
    fig.tight_layout()
    fig.savefig(out_dir / "eval_summary.png", dpi=130)
    plt.close(fig)


def save_video(frames, path, fps):
    try:
        import imageio.v2 as imageio
    except ImportError:
        print("imageio not installed; skipping video (pip install imageio[ffmpeg])")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        imageio.mimsave(str(path), frames, fps=fps, macro_block_size=1)
    except Exception as exc:  # noqa: BLE001  (e.g. ffmpeg backend missing); keep the evaluation going
        print(f"WARNING: could not write {path.name}: {exc!r}. Try: pip install imageio[ffmpeg]")


# --------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--run", help="a train_ppo.py run folder")
    src.add_argument("--controller", help="hold, random, or module:factory")
    ap.add_argument("--env", default=None, help="env id (required with --controller; overrides the run's)")
    ap.add_argument("--model", default="best", help="best, final, or a checkpoint .zip path (with --run)")
    ap.add_argument("--stochastic", action="store_true", help="sample PPO actions instead of the mean")
    ap.add_argument("--conditions", nargs="+", default=["nominal", "friction", "push"])
    ap.add_argument("--episodes", type=int, default=50, help="episodes per condition value")
    ap.add_argument("--seed-base", type=int, default=1000)
    ap.add_argument("--video", type=int, default=0, help="record this many nominal episodes (plus one per push value)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    ctrl, env_id, base_kwargs, label = build_controller(args)
    base_kwargs = {k: v for k, v in base_kwargs.items() if k not in EVAL_OWNED}
    if args.out:
        out_dir = Path(args.out)
    elif args.run:
        out_dir = Path(args.run) / "eval"
    else:
        out_dir = REPO_ROOT / "runs" / f"eval_{label.replace(':', '_').replace('.', '_')}_{env_id}"
    out_dir.mkdir(parents=True, exist_ok=True)

    conds = condition_list(args.conditions)
    seeds = [args.seed_base + i for i in range(args.episodes)]
    rows, push_rows = [], []
    print(f"Evaluating {label} on {env_id}: {len(conds)} condition values x {len(seeds)} episodes"
          f" -> {out_dir}")
    t0 = time.time()
    video_broken = False
    for cond, value, extra_kwargs, reset_opts in conds:
        want_video = args.video > 0 and (cond == "nominal" or cond == "push")
        n_video = args.video if cond == "nominal" else (1 if want_video else 0)
        env = make_env(env_id, {**base_kwargs, **extra_kwargs})
        venv = None
        if n_video and not video_broken:
            try:
                venv = make_env(env_id, {**base_kwargs, **extra_kwargs}, render_mode="rgb_array")
                venv.reset(seed=0)
                venv.render()  # fail early if offscreen rendering is unavailable
            except Exception as exc:  # noqa: BLE001
                print(f"WARNING: rendering failed ({exc!r}); continuing without videos."
                      " On Windows, clear MUJOCO_GL (set MUJOCO_GL=) and retry.")
                video_broken = True
                venv = None
        if venv is None:
            n_video = 0
        for i, seed in enumerate(seeds):
            record = i < n_video
            frames = [] if record else None
            ep, pushes = run_episode(venv if record else env, ctrl, seed, reset_opts, frames)
            tag = {"controller": label, "env_id": env_id, "condition": cond,
                   "value": "" if value is None else value, "seed": seed}
            rows.append({**tag, **ep})
            push_rows += [{**tag, **p} for p in pushes]
            if record:
                vname = f"{cond}{'' if value is None else '_' + str(value)}_seed{seed}.mp4"
                save_video(frames, out_dir / "videos" / vname, env.metadata.get("render_fps", 50))
        env.close()
        if venv is not None:
            venv.close()
        rs = [r for r in rows if r["condition"] == cond and r["value"] == ("" if value is None else value)]
        fall = sum(r["fell"] for r in rs) / len(rs)
        dist = sum(r["distance_m"] for r in rs) / len(rs)
        print(f"{cond:9s} {'' if value is None else value!s:>5}  fall {fall:5.0%}  distance {dist * 100:6.1f} cm"
              f"  ({time.time() - t0:.0f} s)")

    summary = summarize(rows, push_rows)
    write_csv(out_dir / "episodes.csv", rows)
    write_csv(out_dir / "summary.csv", summary)
    write_csv(out_dir / "pushes.csv", push_rows)
    save_json(out_dir / "eval_config.json", {
        "controller": label, "env_id": env_id, "env_kwargs": base_kwargs, "conditions": args.conditions,
        "frictions": FRICTIONS, "pushes": PUSHES, "episodes": args.episodes, "seed_base": args.seed_base,
        "recovery": {"window_s": RECOVERY_WINDOW_S, "speed_window_s": SPEED_WINDOW_S,
                     "pre_push_window_s": PRE_PUSH_WINDOW_S, "speed_frac": SPEED_RECOVERY_FRAC},
        "contact_threshold_N": CONTACT_THRESHOLD, **provenance(),
    })
    try:
        plot(summary, out_dir, f"{label} on {env_id}")
    except ImportError:
        print("matplotlib not installed; skipping plots")
    print(f"Wrote {out_dir}")


if __name__ == "__main__":
    main()
