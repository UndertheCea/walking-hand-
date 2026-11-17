"""Open a MuJoCo scene in the interactive viewer, starting from a keyframe.

Usage:
    mjpython scripts/view.py models/thing_hand/scene.xml          # macOS: use mjpython
    python scripts/view.py models/thing_hand/scene.xml            # Linux / Windows

By default the first keyframe in the scene is loaded (for scene.xml that is
the "stand" pose). Pass --key "" to start from qpos0.
"""
from __future__ import annotations

import argparse
import time

import mujoco
import mujoco.viewer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("scene")
    ap.add_argument("--key", default=None, help="keyframe name (default: first keyframe)")
    ap.add_argument(
        "--paused",
        action="store_true",
        help="start with physics paused (useful to inspect the stand pose)",
    )
    args = ap.parse_args()

    model = mujoco.MjModel.from_xml_path(args.scene)
    data = mujoco.MjData(model)
    if args.key == "":
        pass
    elif args.key is None and model.nkey > 0:
        mujoco.mj_resetDataKeyframe(model, data, 0)
    elif args.key:
        mujoco.mj_resetDataKeyframe(model, data, model.key(args.key).id)
    mujoco.mj_forward(model, data)

    # launch_passive is more reliable than viewer.launch on macOS (mjpython).
    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer.sync()
        while viewer.is_running():
            step_start = time.time()
            if not args.paused:
                mujoco.mj_step(model, data)
            viewer.sync()
            # Real-time pacing.
            elapsed = time.time() - step_start
            remaining = model.opt.timestep - elapsed
            if remaining > 0:
                time.sleep(remaining)


if __name__ == "__main__":
    main()
