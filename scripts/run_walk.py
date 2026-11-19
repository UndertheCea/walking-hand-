"""Roll out the classical WALK controller (elevated stance+swing IK).

Stand / balance only:  ``python scripts/run_balance.py``
Walk (this script):    ``python scripts/run_walk.py``

Default gait keeps the palm ~10 cm and advances ~1 cm/s (~2 cm/cycle at 0.5 Hz).
The old fast crawl (~3 cm/s, palm drops) is ``--fast``.

Usage::

    source .venv/bin/activate
    export PYTHONPATH=.
    python scripts/run_walk.py
    python scripts/run_walk.py --push
    python scripts/run_walk.py --sweep --out runs/walk_push
    mjpython scripts/run_walk.py --render
"""
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import time
from pathlib import Path

import gymnasium as gym
import mujoco.viewer
import numpy as np

import walking_hand  # noqa: F401
from walking_hand.baseline import WalkController

# Named push conditions for the sweep: (label, force_range or None).
PUSH_CONDITIONS = [
    ("no_push", None),
    ("light", (0.10, 0.20)),
    ("default", (0.15, 0.35)),
    ("medium", (0.25, 0.45)),
    ("strong", (0.35, 0.60)),
]


def git_short_hash() -> str:
    try:
        return (
            subprocess.check_output(
                ["git", "rev-parse", "--short", "HEAD"],
                stderr=subprocess.DEVNULL,
            )
            .decode()
            .strip()
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def rollout(env, make_policy, seed):
    obs, info = env.reset(seed=seed)
    policy = make_policy(env)
    if hasattr(policy, "reset"):
        policy.reset()
    x0 = info["x_position"]
    uprights = [float(info["upright"])]
    heights = [float(info["palm_height"])]
    n_push_steps = int(bool(info.get("pushing", False)))
    steps = 0
    while True:
        obs, _, term, trunc, info = env.step(policy(obs))
        uprights.append(float(info["upright"]))
        heights.append(float(info["palm_height"]))
        n_push_steps += int(bool(info.get("pushing", False)))
        steps += 1
        if term or trunc:
            break
    fell = bool(term)
    dx = float(info["x_position"] - x0)
    t = steps * env.unwrapped.dt
    speed = dx / t if t > 0 else 0.0
    return {
        "seed": seed,
        "steps": steps,
        "survived": int(not fell),
        "ended": "fell" if fell else "time_limit",
        "distance_m": dx,
        "speed_m_s": speed,
        "mean_upright": float(np.mean(uprights)),
        "min_upright": float(np.min(uprights)),
        "mean_palm_height": float(np.mean(heights)),
        "min_palm_height": float(np.min(heights)),
        "push_steps": n_push_steps,
    }


def make_env(*, push_force_range=None):
    kwargs = {}
    if push_force_range is not None:
        kwargs["push_force_range"] = push_force_range
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
    env = make_env(push_force_range=(0.15, 0.35) if args.push else None)
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


def print_rows(label, force, args, rows):
    force_s = "off" if force is None else f"{force[0]:.2f}-{force[1]:.2f} N"
    mode = "fast" if args.fast else "elevated"
    n_ok = sum(r["survived"] for r in rows)
    speed = [100 * r["speed_m_s"] for r in rows]
    meanh = [100 * r["mean_palm_height"] for r in rows]
    minh = [100 * r["min_palm_height"] for r in rows]
    upright = [round(r["mean_upright"], 3) for r in rows]
    ends = [r["ended"] for r in rows]
    cm_cycle = float(np.mean(speed)) / args.frequency if args.frequency else 0.0
    print(f"\n=== {label}  mode={mode}  push={force_s}  seeds={args.seeds} ===")
    print(
        f"survive {n_ok}/{len(rows)}  speed_cm_s {np.round(speed, 2).tolist()}  "
        f"mean={np.mean(speed):.2f}  cm/cycle≈{cm_cycle:.2f}  "
        f"mean_palm_cm {np.round(meanh, 1).tolist()}  "
        f"min_palm_cm {np.round(minh, 1).tolist()}  "
        f"mean_upright {upright}  ended {ends}"
    )


def run_once(args):
    force = (0.15, 0.35) if args.push else None
    env = make_env(push_force_range=force)
    rows = [rollout(env, lambda e: make_walker(e, args), seed) for seed in range(args.seeds)]
    print_rows("single", force, args, rows)
    env.close()
    if args.out:
        write_results(Path(args.out), [("single", force, rows)], args)


def run_sweep(args):
    out = Path(args.out or "runs/walk_push")
    all_conditions = []
    for label, force in PUSH_CONDITIONS:
        env = make_env(push_force_range=force)
        rows = [rollout(env, lambda e: make_walker(e, args), seed) for seed in range(args.seeds)]
        print_rows(label, force, args, rows)
        all_conditions.append((label, force, rows))
        env.close()
    write_results(out, all_conditions, args)


def write_results(out: Path, conditions, args):
    out.mkdir(parents=True, exist_ok=True)
    mode = "fast" if args.fast else "elevated"
    meta = {
        "env_id": "WalkingHand-v1",
        "git": git_short_hash(),
        "mode": mode,
        "frequency": args.frequency,
        "step_length": args.step_length,
        "step_height": args.step_height,
        "duty": args.duty,
        "min_palm_height": args.min_palm_height,
        "seeds": args.seeds,
        "push_interval_range": [1.0, 2.0],
        "push_duration": 0.1,
        "conditions": [
            {
                "name": label,
                "push_force_range": None if force is None else list(force),
            }
            for label, force, _ in conditions
        ],
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")

    episode_path = out / "episodes.csv"
    with episode_path.open("w", newline="") as f:
        fieldnames = [
            "condition",
            "push_lo",
            "push_hi",
            "seed",
            "steps",
            "survived",
            "ended",
            "distance_m",
            "speed_m_s",
            "speed_cm_s",
            "mean_upright",
            "min_upright",
            "mean_palm_height",
            "min_palm_height",
            "push_steps",
        ]
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for label, force, rows in conditions:
            lo = "" if force is None else f"{force[0]:.3f}"
            hi = "" if force is None else f"{force[1]:.3f}"
            for r in rows:
                w.writerow(
                    {
                        "condition": label,
                        "push_lo": lo,
                        "push_hi": hi,
                        "seed": r["seed"],
                        "steps": r["steps"],
                        "survived": r["survived"],
                        "ended": r["ended"],
                        "distance_m": f"{r['distance_m']:.6f}",
                        "speed_m_s": f"{r['speed_m_s']:.6f}",
                        "speed_cm_s": f"{100 * r['speed_m_s']:.4f}",
                        "mean_upright": f"{r['mean_upright']:.6f}",
                        "min_upright": f"{r['min_upright']:.6f}",
                        "mean_palm_height": f"{r['mean_palm_height']:.6f}",
                        "min_palm_height": f"{r['min_palm_height']:.6f}",
                        "push_steps": r["push_steps"],
                    }
                )

    summary_rows = []
    summary_path = out / "summary.csv"
    with summary_path.open("w", newline="") as f:
        fieldnames = [
            "condition",
            "push_lo",
            "push_hi",
            "n",
            "survive_rate",
            "mean_steps",
            "mean_speed_cm_s",
            "mean_distance_cm",
            "mean_upright",
            "mean_palm_cm",
            "mean_min_palm_cm",
        ]
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for label, force, rows in conditions:
            lo = "" if force is None else f"{force[0]:.3f}"
            hi = "" if force is None else f"{force[1]:.3f}"
            row = {
                "condition": label,
                "push_lo": lo,
                "push_hi": hi,
                "n": len(rows),
                "survive_rate": sum(r["survived"] for r in rows) / len(rows),
                "mean_steps": float(np.mean([r["steps"] for r in rows])),
                "mean_speed_cm_s": float(np.mean([100 * r["speed_m_s"] for r in rows])),
                "mean_distance_cm": float(np.mean([100 * r["distance_m"] for r in rows])),
                "mean_upright": float(np.mean([r["mean_upright"] for r in rows])),
                "mean_palm_cm": float(np.mean([100 * r["mean_palm_height"] for r in rows])),
                "mean_min_palm_cm": float(np.mean([100 * r["min_palm_height"] for r in rows])),
            }
            summary_rows.append(row)
            w.writerow(
                {
                    **row,
                    "survive_rate": f"{row['survive_rate']:.3f}",
                    "mean_steps": f"{row['mean_steps']:.1f}",
                    "mean_speed_cm_s": f"{row['mean_speed_cm_s']:.3f}",
                    "mean_distance_cm": f"{row['mean_distance_cm']:.2f}",
                    "mean_upright": f"{row['mean_upright']:.4f}",
                    "mean_palm_cm": f"{row['mean_palm_cm']:.2f}",
                    "mean_min_palm_cm": f"{row['mean_min_palm_cm']:.2f}",
                }
            )

    lines = [
        f"Walk push sweep  env=WalkingHand-v1  mode={mode}  "
        f"f={args.frequency}Hz  git={meta['git']}  seeds={args.seeds}",
        "",
    ]
    for row in summary_rows:
        push = "off" if row["push_lo"] == "" else f"{row['push_lo']}-{row['push_hi']} N"
        lines.append(
            f"{row['condition']:10s}  push={push:16s}  "
            f"survive={row['survive_rate']:.0%}  "
            f"speed={row['mean_speed_cm_s']:.2f} cm/s  "
            f"dist={row['mean_distance_cm']:.1f} cm  "
            f"palm={row['mean_palm_cm']:.1f} cm  "
            f"upright={row['mean_upright']:.3f}"
        )
    (out / "summary.txt").write_text("\n".join(lines) + "\n")

    plot_notes = []
    try:
        plot_metric_bars(
            summary_rows,
            out / "survive_rate.png",
            meta["git"],
            mode,
            metric="survive_rate",
            ylabel="Survive rate (no fall)",
            title="Walk vs push — survival",
            ylim=(0.0, 1.05),
        )
        plot_metric_bars(
            summary_rows,
            out / "mean_speed.png",
            meta["git"],
            mode,
            metric="mean_speed_cm_s",
            ylabel="Mean forward speed (cm/s)",
            title="Walk vs push — speed",
            ylim=None,
        )
        plot_metric_bars(
            summary_rows,
            out / "mean_palm_height.png",
            meta["git"],
            mode,
            metric="mean_palm_cm",
            ylabel="Mean palm height (cm)",
            title="Walk vs push — palm height",
            ylim=None,
        )
        plot_notes.extend(["survive_rate.png", "mean_speed.png", "mean_palm_height.png"])
    except Exception as exc:  # noqa: BLE001
        plot_notes.append(f"(plot skipped: {exc})")

    print(f"\nWrote results to {out}/")
    print(f"  {episode_path.name}  {summary_path.name}  summary.txt  {' '.join(plot_notes)}")


def plot_metric_bars(summary_rows, path: Path, git: str, mode: str, *, metric, ylabel, title, ylim):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    conditions = [row["condition"] for row in summary_rows]
    vals = [row[metric] for row in summary_rows]
    x = np.arange(len(conditions))
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(x, vals, width=0.55, color="#4C78A8")
    ax.set_ylabel(ylabel)
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.set_xticks(x)
    ax.set_xticklabels(conditions)
    ax.set_xlabel("Push condition")
    ax.set_title(f"{title}  ({mode}, WalkingHand-v1, {git})")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


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
    ap.add_argument(
        "--sweep",
        action="store_true",
        help="sweep several push-force levels and save CSV/plots",
    )
    ap.add_argument(
        "--out",
        type=str,
        default=None,
        help="output directory (default: runs/walk_push for --sweep)",
    )
    ap.add_argument("--render", action="store_true")
    args = ap.parse_args()

    if args.frequency is None:
        args.frequency = 2.5 if args.fast else 0.5
    if args.min_palm_height is None:
        args.min_palm_height = 0.0 if args.fast else 0.075

    if args.render:
        run_viewer(args)
        return
    if args.sweep:
        run_sweep(args)
        return
    run_once(args)


if __name__ == "__main__":
    main()
