"""Sanity checks for the walking envs.

Usage: python scripts/check_env.py [--render out.png]

1. Gymnasium's API checker on both env ids.
2. Same seed, same actions: both envs produce identical physics.
3. Rollouts with a "hold the stand pose" policy and a random policy,
   reporting episode length, distance and why the episode ended.
4. Friction option and pushes take effect.
"""
import argparse

import gymnasium as gym
import numpy as np
from gymnasium.utils.env_checker import check_env

import walking_hand  # noqa: F401  (registers the envs)


def rollout(env, policy, seed):
    obs, info = env.reset(seed=seed)
    x0, total, steps = info["x_position"], 0.0, 0
    while True:
        obs, r, term, trunc, info = env.step(policy(obs))
        total += r
        steps += 1
        if term or trunc:
            break
    end = "fell" if term else "time limit"
    return steps, info["x_position"] - x0, total, end


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--render", default=None, help="save a frame from rgb_array rendering")
    args = ap.parse_args()

    for env_id in ["WalkingHand-v1", "WalkingHandTactile-v1"]:
        env = gym.make(env_id)
        check_env(env.unwrapped, skip_render_check=True)
        print(f"{env_id}: check_env passed, obs {env.observation_space.shape}, act {env.action_space.shape}")
        env.close()

    # Identical physics across the two variants.
    a, b = gym.make("WalkingHand-v1"), gym.make("WalkingHandTactile-v1")
    oa, _ = a.reset(seed=3)
    ob, _ = b.reset(seed=3)
    rng = np.random.default_rng(0)
    for _ in range(100):
        act = rng.uniform(-1, 1, 6)
        oa, ra, *_ = a.step(act)
        ob, rb, *_ = b.step(act)
    same = np.allclose(oa, ob[: len(oa)]) and np.isclose(ra, rb)
    print(f"same seed + actions -> identical physics: {same}; tactile extra obs = {ob[len(oa):].round(3)}")

    env = gym.make("WalkingHandTactile-v1")
    hold = lambda obs: obs[22:28]  # previous action = keep the current targets
    rng = np.random.default_rng(1)
    rand = lambda obs: rng.uniform(-1, 1, 6)
    for name, pol in [("hold stand pose", hold), ("random", rand)]:
        res = [rollout(env, pol, seed) for seed in range(5)]
        steps = [r[0] for r in res]
        dist = [1000 * r[1] for r in res]
        print(f"{name:16s} steps {steps}  distance mm {np.round(dist, 1).tolist()}  ended: {[r[3] for r in res]}")

    _, info = env.reset(seed=0, options={"friction": 0.3})
    print(f"reset(options={{'friction': 0.3}}) -> floor friction {info['friction']}")

    pushed = gym.make("WalkingHand-v1", push_force_range=(0.5, 1.0), push_interval_range=(0.2, 0.4))
    pushed.reset(seed=0)
    n_push = sum(pushed.step(np.zeros(6))[4]["pushing"] for _ in range(50))
    print(f"push test: pushing on {n_push}/50 steps")

    if args.render:
        from render import write_png  # scripts/render.py

        r = gym.make("WalkingHand-v1", render_mode="rgb_array")
        r.reset(seed=0)
        write_png(args.render, r.render())
        print(f"wrote {args.render}")


if __name__ == "__main__":
    main()
