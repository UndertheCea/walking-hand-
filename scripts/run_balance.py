"""STAND / balance only (no walking clock).

Holds the tall two-finger stand with lean feedback. For walking, use
``scripts/run_walk.py`` instead.

Usage (from repo root, with the project venv)::

    source .venv/bin/activate
    export PYTHONPATH=.
    python scripts/run_balance.py
    python scripts/run_balance.py --push
    python scripts/run_balance.py --sweep --out runs/balance_push
    mjpython scripts/run_balance.py --render          # macOS: use mjpython for the viewer
    mjpython scripts/run_balance.py --push --render

Reports steps survived, mean upright, and fall vs time-limit for each policy.
With ``--sweep``, writes per-episode CSV, summary CSV, and a bar-chart PNG
under ``--out`` for several push-force levels.
"""
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import time
from pathlib import Path

import gymnasium as gym
import mujoco
import mujoco.viewer
import numpy as np

import walking_hand  # noqa: F401
from walking_hand.baseline import BalanceHoldController

# Named push conditions for the sweep: (label, force_range or None).
# force_range is (lo, hi) in Newtons; None = no pushes.
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


def rollout(env, policy, seed):
    """Run one episode; return metrics dict."""
    obs, info = env.reset(seed=seed)
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
    return {
        "seed": seed,
        "steps": steps,
        "survived": int(not fell),
        "ended": "fell" if fell else "time_limit",
        "mean_upright": float(np.mean(uprights)),
        "min_upright": float(np.min(uprights)),
        "mean_palm_height": float(np.mean(heights)),
        "min_palm_height": float(np.min(heights)),
        "push_steps": n_push_steps,
    }


def make_env(*, push_force_range=None, render: bool = False):
    kwargs = {}
    if push_force_range is not None:
        kwargs["push_force_range"] = push_force_range
        kwargs["push_interval_range"] = (1.0, 2.0)
        kwargs["push_duration"] = 0.1
    if render:
        kwargs["render_mode"] = None
    return gym.make("WalkingHand-v1", **kwargs)


def run_viewer(args):
    """Live viewer via launch_passive (macOS: run this script with mjpython)."""
    force = (0.15, 0.35) if args.push else None
    env = make_env(push_force_range=force, render=True)
    raw = env.unwrapped
    balance = BalanceHoldController()
    obs, info = env.reset(seed=0)
    print(f"viewer: STAND (balance hold), upright={info['upright']:.3f}")
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
            remaining = raw.dt - (time.time() - step_start)
            if remaining > 0:
                time.sleep(remaining)
    env.close()


def policies():
    hold = lambda obs: obs[22:28]  # previous action = keep reset targets
    return [("open-loop hold", hold), ("balance hold", BalanceHoldController())]


def print_condition(name, force, seeds, rows_by_policy):
    force_s = "off" if force is None else f"{force[0]:.2f}-{force[1]:.2f} N"
    print(f"\n=== {name}  push={force_s}  seeds={seeds} ===")
    for pname, rows in rows_by_policy.items():
        steps = [r["steps"] for r in rows]
        upright = [r["mean_upright"] for r in rows]
        ends = [r["ended"] for r in rows]
        n_ok = sum(r["survived"] for r in rows)
        print(
            f"{pname:16s}  survive {n_ok}/{len(rows)}  steps {steps}  "
            f"mean_upright {[round(u, 3) for u in upright]}  ended {ends}"
        )


def run_once(args):
    force = (0.15, 0.35) if args.push else None
    env = make_env(push_force_range=force)
    rows_by_policy = {}
    for name, pol in policies():
        rows_by_policy[name] = [rollout(env, pol, seed) for seed in range(args.seeds)]
    print_condition("single", force, args.seeds, rows_by_policy)
    env.close()
    if args.out:
        write_results(Path(args.out), [("single", force, rows_by_policy)], args)


def run_sweep(args):
    out = Path(args.out or "runs/balance_push")
    all_conditions = []
    for label, force in PUSH_CONDITIONS:
        env = make_env(push_force_range=force)
        rows_by_policy = {}
        for name, pol in policies():
            rows_by_policy[name] = [rollout(env, pol, seed) for seed in range(args.seeds)]
        print_condition(label, force, args.seeds, rows_by_policy)
        all_conditions.append((label, force, rows_by_policy))
        env.close()
    write_results(out, all_conditions, args)


def write_results(out: Path, conditions, args):
    out.mkdir(parents=True, exist_ok=True)
    meta = {
        "env_id": "WalkingHand-v1",
        "git": git_short_hash(),
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
            "policy",
            "seed",
            "steps",
            "survived",
            "ended",
            "mean_upright",
            "min_upright",
            "mean_palm_height",
            "min_palm_height",
            "push_steps",
        ]
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for label, force, rows_by_policy in conditions:
            lo = "" if force is None else f"{force[0]:.3f}"
            hi = "" if force is None else f"{force[1]:.3f}"
            for pname, rows in rows_by_policy.items():
                for r in rows:
                    w.writerow(
                        {
                            "condition": label,
                            "push_lo": lo,
                            "push_hi": hi,
                            "policy": pname,
                            "seed": r["seed"],
                            "steps": r["steps"],
                            "survived": r["survived"],
                            "ended": r["ended"],
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
            "policy",
            "n",
            "survive_rate",
            "mean_steps",
            "mean_upright",
            "mean_min_upright",
        ]
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for label, force, rows_by_policy in conditions:
            lo = "" if force is None else f"{force[0]:.3f}"
            hi = "" if force is None else f"{force[1]:.3f}"
            for pname, rows in rows_by_policy.items():
                row = {
                    "condition": label,
                    "push_lo": lo,
                    "push_hi": hi,
                    "policy": pname,
                    "n": len(rows),
                    "survive_rate": sum(r["survived"] for r in rows) / len(rows),
                    "mean_steps": float(np.mean([r["steps"] for r in rows])),
                    "mean_upright": float(np.mean([r["mean_upright"] for r in rows])),
                    "mean_min_upright": float(np.mean([r["min_upright"] for r in rows])),
                }
                summary_rows.append(row)
                w.writerow(
                    {
                        **row,
                        "survive_rate": f"{row['survive_rate']:.3f}",
                        "mean_steps": f"{row['mean_steps']:.1f}",
                        "mean_upright": f"{row['mean_upright']:.4f}",
                        "mean_min_upright": f"{row['mean_min_upright']:.4f}",
                    }
                )

    lines = [
        f"Balance push sweep  env=WalkingHand-v1  git={meta['git']}  seeds={args.seeds}",
        "",
    ]
    for row in summary_rows:
        push = "off" if row["push_lo"] == "" else f"{row['push_lo']}-{row['push_hi']} N"
        lines.append(
            f"{row['condition']:10s}  {row['policy']:16s}  push={push:16s}  "
            f"survive={row['survive_rate']:.0%}  mean_steps={row['mean_steps']:.0f}  "
            f"mean_upright={row['mean_upright']:.3f}"
        )
    (out / "summary.txt").write_text("\n".join(lines) + "\n")

    plot_notes = []
    try:
        plot_metric_bars(
            summary_rows,
            out / "survive_rate.png",
            meta["git"],
            metric="survive_rate",
            ylabel="Survive rate (no fall)",
            title="Stand balance vs push — survival",
            ylim=(0.0, 1.05),
        )
        plot_metric_bars(
            summary_rows,
            out / "mean_upright.png",
            meta["git"],
            metric="mean_upright",
            ylabel="Mean upright (episode avg)",
            title="Stand balance vs push — uprightness",
            ylim=None,
        )
        plot_notes.append("survive_rate.png")
        plot_notes.append("mean_upright.png")
    except Exception as exc:  # noqa: BLE001 — plotting is best-effort
        plot_notes.append(f"(plot skipped: {exc})")

    print(f"\nWrote results to {out}/")
    print(f"  {episode_path.name}  {summary_path.name}  summary.txt  {' '.join(plot_notes)}")


def plot_metric_bars(summary_rows, path: Path, git: str, *, metric, ylabel, title, ylim):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    conditions = []
    for row in summary_rows:
        if row["condition"] not in conditions:
            conditions.append(row["condition"])
    policies_order = []
    for row in summary_rows:
        if row["policy"] not in policies_order:
            policies_order.append(row["policy"])

    x = np.arange(len(conditions))
    width = 0.35
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for i, pname in enumerate(policies_order):
        vals = []
        for cond in conditions:
            match = next(
                r for r in summary_rows if r["condition"] == cond and r["policy"] == pname
            )
            vals.append(match[metric])
        ax.bar(x + (i - 0.5) * width, vals, width, label=pname)

    ax.set_ylabel(ylabel)
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.set_xticks(x)
    ax.set_xticklabels(conditions)
    ax.set_xlabel("Push condition")
    ax.set_title(f"{title}  (WalkingHand-v1, {git})")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seeds", type=int, default=5, help="number of rollout seeds")
    ap.add_argument("--push", action="store_true", help="apply default horizontal pushes")
    ap.add_argument(
        "--sweep",
        action="store_true",
        help="sweep several push-force levels and save CSV/plots",
    )
    ap.add_argument(
        "--out",
        type=str,
        default=None,
        help="output directory (default: runs/balance_push for --sweep)",
    )
    ap.add_argument("--render", action="store_true", help="live MuJoCo viewer (balance only)")
    args = ap.parse_args()

    if args.render:
        run_viewer(args)
        return
    if args.sweep:
        run_sweep(args)
        return
    run_once(args)


if __name__ == "__main__":
    main()
