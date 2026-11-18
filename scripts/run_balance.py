"""STAND / balance only (no walking clock).

Holds the tall two-finger stand with lean feedback. For walking, use
``scripts/run_walk.py`` instead.

Usage (from repo root, with the project venv)::

    source .venv/bin/activate
    export PYTHONPATH=.
    python scripts/run_balance.py
    python scripts/run_balance.py --push
    mjpython scripts/run_balance.py --render          # macOS: use mjpython for the viewer
    mjpython scripts/run_balance.py --push --render

Reports steps survived, mean upright, and fall vs time-limit for each policy.
"""
from __future__ import annotations

import argparse
import time

import gymnasium as gym
import mujoco
import mujoco.viewer
import numpy as np

import walking_hand  # noqa: F401
from walking_hand.baseline import BalanceHoldController


def rollout(env, policy, seed):
    obs, info = env.reset(seed=seed)
    uprights = [info["upright"]]
    steps = 0
    while True:
        obs, _, term, trunc, info = env.step(policy(obs))
        uprights.append(info["upright"])
        steps += 1
        if term or trunc:
            break
    end = "fell" if term else "time limit"
    return steps, float(np.mean(uprights)), end


def make_env(args, *, render: bool = False):
    kwargs = dict(stub_contact=False)
    if args.push:
        kwargs["push_force_range"] = (0.15, 0.35)
        kwargs["push_interval_range"] = (1.0, 2.0)
        kwargs["push_duration"] = 0.1
    # Headless rollouts never use render_mode=human (macOS needs mjpython + passive viewer).
    if render:
        kwargs["render_mode"] = None
    return gym.make("WalkingHand-v1", **kwargs)


def run_viewer(args):
    """Live viewer via launch_passive (macOS: run this script with mjpython)."""
    env = make_env(args, render=True)
    raw = env.unwrapped
    balance = BalanceHoldController()
    obs, info = env.reset(seed=0)
    print(f"viewer: STAND (balance hold), stub off, upright={info['upright']:.3f}")
    print("Close the window or Ctrl+C to quit.")

    with mujoco.viewer.launch_passive(raw.model, raw.data) as viewer:
        viewer.sync()
        while viewer.is_running():
            step_start = time.time()
            obs, _, term, trunc, info = env.step(balance(obs))
            viewer.sync()
            if term or trunc:
                print(f"ended: {'fell' if term else 'time limit'} upright={info['upright']:.3f}")
                obs, info = env.reset()
            # Pace roughly to control rate (50 Hz).
            remaining = raw.dt - (time.time() - step_start)
            if remaining > 0:
                time.sleep(remaining)
    env.close()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seeds", type=int, default=5, help="number of rollout seeds")
    ap.add_argument("--push", action="store_true", help="apply light horizontal pushes")
    ap.add_argument("--render", action="store_true", help="live MuJoCo viewer (balance only)")
    args = ap.parse_args()

    if args.render:
        run_viewer(args)
        return

    env = make_env(args)
    hold = lambda obs: obs[22:28]  # previous action = keep reset targets
    balance = BalanceHoldController()

    print(f"STAND  WalkingHand-v1  stub_contact=False  pushes={'on' if args.push else 'off'}  seeds={args.seeds}")
    for name, pol in [("open-loop hold", hold), ("balance hold", balance)]:
        rows = [rollout(env, pol, seed) for seed in range(args.seeds)]
        steps = [r[0] for r in rows]
        upright = [r[1] for r in rows]
        ends = [r[2] for r in rows]
        print(
            f"{name:16s}  steps {steps}  "
            f"mean_upright {[round(u, 3) for u in upright]}  "
            f"ended {ends}"
        )
    env.close()


if __name__ == "__main__":
    main()
