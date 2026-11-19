"""Train and evaluate pure RL and tactile RL with identical settings over several seeds.

Examples (run from the repo root):

    # full comparison on v1: 2 policies x 3 seeds, same budget and randomization
    python scripts/run_matrix.py --tag cmp1 --version v1 --seeds 0 1 2 --total-steps 5000000 \
        -- --friction-range 0.3 1.5 --push-range 0.2 0.8

    # tiny end-to-end check of the whole pipeline (a few minutes)
    python scripts/run_matrix.py --tag smoke --version v0 --seeds 0 --total-steps 20000 \
        --eval-episodes 3 -- --n-envs 2 --eval-every 0 --checkpoint-every 0

    # only (re)evaluate and compare runs that already exist, and include other controllers
    python scripts/run_matrix.py --tag cmp1 --skip-train --extra runs/eval_hold_WalkingHand-v1

Everything after "--" is passed unchanged to train_ppo.py for BOTH policies.
That is the point of this script: the two policies can only differ in --env.

Writes runs/matrix_<tag>/:
    <env>_s<seed>/          the train_ppo.py run (with eval/ inside)
    comparison.csv          per controller and condition value: mean and spread over seeds
    comparison.png          pure vs tactile (and any --extra controllers) across conditions
    matrix_config.json
"""
from __future__ import annotations

import argparse
import csv
import math
import subprocess
import sys
from pathlib import Path

import rl_common  # noqa: F401  (puts the repo root on sys.path; must come first)
from rl_common import REPO_ROOT, provenance, save_json

HERE = Path(__file__).resolve().parent
COMPARE = ["fall_rate_mean", "distance_m_mean", "mean_speed_mps_mean", "slip_per_m_walked_mean",
           "push_survival_rate_mean"]


def run(cmd):
    print("\n$ " + " ".join(str(c) for c in cmd), flush=True)
    r = subprocess.run([str(c) for c in cmd], cwd=REPO_ROOT)
    if r.returncode != 0:
        raise SystemExit(f"command failed with exit code {r.returncode}")


def read_csv(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def num(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return math.nan


def aggregate(summaries):
    """summaries: list of (group_label, seed, rows). Average each metric over seeds per group/condition/value."""
    import numpy as np

    groups = {}
    for label, seed, rows in summaries:
        for r in rows:
            groups.setdefault((label, r["condition"], r["value"]), []).append((seed, r))
    out = []
    for (label, cond, val), items in sorted(groups.items(), key=lambda kv: (kv[0][0], kv[0][1], num(kv[0][2]))):
        row = {"controller": label, "condition": cond, "value": val, "n_seeds": len(items)}
        for k in COMPARE:
            vals = np.array([num(r.get(k)) for _, r in items])
            vals = vals[~np.isnan(vals)]
            row[k.replace("_mean", "")] = float(vals.mean()) if len(vals) else math.nan
            row[k.replace("_mean", "") + "_seed_std"] = float(vals.std(ddof=1)) if len(vals) > 1 else math.nan
        out.append(row)
    return out


def plot(rows, path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    panels = [("friction", "fall_rate", "floor friction"), ("friction", "distance_m", "floor friction"),
              ("push", "fall_rate", "push force (N)"), ("push", "distance_m", "push force (N)")]
    panels = [p for p in panels if any(r["condition"] == p[0] for r in rows)]
    if not panels:
        return
    labels = sorted({r["controller"] for r in rows})
    fig, axes = plt.subplots(1, len(panels), figsize=(4.2 * len(panels), 3.6), squeeze=False)
    for ax, (cond, key, xl) in zip(axes[0], panels):
        for lab in labels:
            rs = sorted([r for r in rows if r["controller"] == lab and r["condition"] == cond],
                        key=lambda r: num(r["value"]))
            if not rs:
                continue
            x = [num(r["value"]) for r in rs]
            y = [r[key] for r in rs]
            e = [0 if math.isnan(r[key + "_seed_std"]) else r[key + "_seed_std"] for r in rs]
            ax.errorbar(x, y, yerr=e, marker="o", capsize=3, label=lab)
        ax.set_xlabel(xl)
        ax.set_ylabel(key.replace("_", " "))
        if key == "fall_rate":
            ax.set_ylim(-0.05, 1.05)
        ax.grid(alpha=0.3)
    axes[0][0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main():
    argv = sys.argv[1:]
    train_extra = []
    if "--" in argv:
        i = argv.index("--")
        argv, train_extra = argv[:i], argv[i + 1:]
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tag", required=True, help="name of this comparison (folder runs/matrix_<tag>)")
    ap.add_argument("--version", default="v1", help="env version suffix: WalkingHand-<v> and WalkingHandTactile-<v>")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--total-steps", type=int, default=5_000_000)
    ap.add_argument("--eval-episodes", type=int, default=50)
    ap.add_argument("--conditions", nargs="+", default=["nominal", "friction", "push"])
    ap.add_argument("--skip-train", action="store_true", help="reuse existing runs in the matrix folder")
    ap.add_argument("--skip-eval", action="store_true", help="reuse existing eval/ results")
    ap.add_argument("--extra", nargs="*", default=[], help="eval folders of other controllers to include")
    args = ap.parse_args(argv)
    if any(a in ("--env", "--seed", "--total-steps", "--out", "--name") for a in train_extra):
        raise SystemExit("--env/--seed/--total-steps/--out/--name are set by run_matrix.py; remove them after --")

    root = REPO_ROOT / "runs" / f"matrix_{args.tag}"
    root.mkdir(parents=True, exist_ok=True)
    envs = [f"WalkingHand-{args.version}", f"WalkingHandTactile-{args.version}"]
    save_json(root / "matrix_config.json", {"envs": envs, "seeds": args.seeds, "total_steps": args.total_steps,
                                            "train_extra_args": train_extra, "eval_episodes": args.eval_episodes,
                                            "conditions": args.conditions, **provenance()})

    summaries = []
    for env_id in envs:
        for seed in args.seeds:
            name = f"{env_id}_s{seed}"
            run_dir = root / name
            if not args.skip_train:
                if run_dir.exists():
                    raise SystemExit(f"{run_dir} exists; use a new --tag or --skip-train")
                run([sys.executable, HERE / "train_ppo.py", "--env", env_id, "--seed", seed,
                     "--total-steps", args.total_steps, "--out", root, "--name", name, *train_extra])
            if not args.skip_eval:
                run([sys.executable, HERE / "eval_policy.py", "--run", run_dir, "--episodes", args.eval_episodes,
                     "--conditions", *args.conditions])
            summaries.append((env_id, seed, read_csv(run_dir / "eval" / "summary.csv")))
    for d in args.extra:
        summaries.append((Path(d).name, 0, read_csv(Path(d) / "summary.csv")))

    rows = aggregate(summaries)
    with open(root / "comparison.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    try:
        plot(rows, root / "comparison.png")
    except ImportError:
        print("matplotlib not installed; skipping plot")
    print(f"\nWrote {root / 'comparison.csv'}")
    for r in rows:
        if r["condition"] == "nominal":
            print(f"  {r['controller']:28s} nominal: fall {r['fall_rate']:.0%}, distance {r['distance_m'] * 100:.1f} cm"
                  f" over {r['n_seeds']} seed(s)")


if __name__ == "__main__":
    main()
