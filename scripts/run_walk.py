"""Roll out the classical WALK controller (elevated stance+swing IK).

Stand / balance only:  ``python scripts/run_balance.py``
Walk (this script):    ``python scripts/run_walk.py``

Default gait keeps the palm ~10 cm and advances ~1 cm/s (~2 cm/cycle at 0.5 Hz).
The old fast crawl (~3 cm/s, palm drops) is ``--fast``.

Usage::

    source .venv/bin/activate
    export PYTHONPATH=.
    python scripts/run_walk.py
    mjpython scripts/run_walk.py --render
"""
from __future__ import annotations

import argparse
import time

import gymnasium as gym
import mujoco.viewer
import numpy as np

import walking_hand  # noqa: F401
from walking_hand.baseline import WalkController


def rollout(env, make_policy, seed):
    obs, info = env.reset(seed=seed)
    policy = make_policy(env)
    if hasattr(policy, "reset"):
        policy.reset()
    x0 = info["x_position"]
    uprights = [info["upright"]]
    heights = [info["palm_height"]]
    steps = 0
    while True:
        obs, _, term, trunc, info = env.step(policy(obs))
        uprights.append(info["upright"])
        heights.append(info["palm_height"])
        steps += 1
        if term or trunc:
            break
    end = "fell" if term else "time limit"
    dx = info["x_position"] - x0
    t = steps * env.unwrapped.dt
    return (
        steps,
        dx,
        dx / t if t > 0 else 0.0,
        float(np.mean(uprights)),
        float(np.min(heights)),
        float(np.mean(heights)),
        end,
    )


def make_env(args):
    kwargs = dict(stub_contact=False)
    if args.push:
        kwargs["push_force_range"] = (0.15, 0.35)
        kwargs["push_interval_range"] = (1.0, 2.0)
        kwargs["push_duration"] = 0.1
    return gym.make("WalkingHand-v1", **kwargs)


def make_walker(env, args):
    raw = env.unwrapped
    mode = "fast" if args.fast else "elevated"
    kwargs = dict(
        mode=mode,
        frequency=args.frequency,
        min_palm_height=args.min_palm_height,
        balance_scale=0.5 if mode == "fast" else 1.0,
    )
    if mode == "elevated":
        kwargs.update(
            step_length=args.step_length,
            step_height=args.step_height,
            duty=args.duty,
            max_joint_delta=args.max_joint_delta,
        )
    else:
        kwargs.update(
            mcp_amp=args.mcp_amp,
            pip_amp=args.pip_amp,
            abd_amp=args.abd_amp,
        )
    return WalkController(raw.model, raw.data, dt=raw.dt, **kwargs)


def run_viewer(args):
    env = make_env(args)
    raw = env.unwrapped
    obs, info = env.reset(seed=0)
    walker = make_walker(env, args)
    walker.reset()
    x0 = info["x_position"]
    mode = "fast" if args.fast else "elevated"
    print(
        f"viewer: WALK mode={mode}  f={args.frequency}Hz  "
        f"upright={info['upright']:.3f}  palm={100*info['palm_height']:.1f}cm"
    )
    with mujoco.viewer.launch_passive(raw.model, raw.data) as viewer:
        viewer.sync()
        while viewer.is_running():
            t0 = time.time()
            obs, _, term, trunc, info = env.step(walker(obs))
            viewer.sync()
            if term or trunc:
                dx = info["x_position"] - x0
                t = env.unwrapped.data.time
                print(
                    f"ended: {'fell' if term else 'time limit'}  "
                    f"dx={100*dx:.1f}cm  v≈{100*dx/max(t,1e-6):.2f}cm/s  "
                    f"palm_h={100*info['palm_height']:.1f}cm"
                )
                obs, info = env.reset()
                walker.reset()
                x0 = info["x_position"]
            rem = raw.dt - (time.time() - t0)
            if rem > 0:
                time.sleep(rem)
    env.close()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument(
        "--fast",
        action="store_true",
        help="opt-in joint crawl (~3 cm/s); palm drops near the floor",
    )
    ap.add_argument("--frequency", type=float, default=None, help="gait cycles per second")
    ap.add_argument("--step-length", type=float, default=0.025, help="elevated tip travel (m)")
    ap.add_argument("--step-height", type=float, default=0.012, help="elevated tip lift (m)")
    ap.add_argument("--duty", type=float, default=0.55, help="elevated stance duty")
    ap.add_argument("--max-joint-delta", type=float, default=0.45, help="elevated IK clamp (rad)")
    ap.add_argument("--mcp-amp", type=float, default=0.35, help="fast MCP amp (rad)")
    ap.add_argument("--pip-amp", type=float, default=0.25, help="fast PIP amp (rad)")
    ap.add_argument("--abd-amp", type=float, default=0.05, help="fast abd amp (rad)")
    ap.add_argument("--min-palm-height", type=float, default=None)
    ap.add_argument("--push", action="store_true")
    ap.add_argument("--render", action="store_true")
    args = ap.parse_args()

    if args.frequency is None:
        args.frequency = 2.5 if args.fast else 0.5
    if args.min_palm_height is None:
        args.min_palm_height = 0.0 if args.fast else 0.075

    if args.render:
        run_viewer(args)
        return

    env = make_env(args)
    mode = "fast" if args.fast else "elevated"
    print(
        f"WALK  mode={mode}  WalkingHand-v1  stub=off  f={args.frequency}Hz  "
        f"L={args.step_length} H={args.step_height}  minh={args.min_palm_height}  "
        f"pushes={'on' if args.push else 'off'}"
    )
    rows = [rollout(env, lambda e: make_walker(e, args), seed) for seed in range(args.seeds)]
    speed = [100 * r[2] for r in rows]
    meanh = [100 * r[5] for r in rows]
    minh = [100 * r[4] for r in rows]
    upright = [round(r[3], 3) for r in rows]
    ends = [r[6] for r in rows]
    cm_cycle = float(np.mean(speed)) / args.frequency if args.frequency else 0.0
    print(
        f"speed_cm_s {np.round(speed, 2).tolist()}  mean={np.mean(speed):.2f}  "
        f"cm/cycle≈{cm_cycle:.2f}  mean_palm_cm {np.round(meanh, 1).tolist()}  "
        f"min_palm_cm {np.round(minh, 1).tolist()}  mean_upright {upright}  ended {ends}"
    )
    env.close()


if __name__ == "__main__":
    main()
