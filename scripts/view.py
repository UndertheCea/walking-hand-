"""Open a MuJoCo scene in the interactive viewer, starting from a keyframe.

Usage: python scripts/view.py <scene.xml> [--key NAME]

By default the first keyframe in the scene is loaded (for scene.xml that is
the "stand" pose), so the hand starts standing on its two fingertips instead
of dropping from the all-zero pose. Pass --key "" to start from qpos0.
"""
import argparse

import mujoco
import mujoco.viewer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("scene")
    ap.add_argument("--key", default=None, help="keyframe name (default: first keyframe)")
    args = ap.parse_args()

    model = mujoco.MjModel.from_xml_path(args.scene)
    data = mujoco.MjData(model)
    if args.key is None and model.nkey > 0:
        mujoco.mj_resetDataKeyframe(model, data, 0)
    elif args.key:
        mujoco.mj_resetDataKeyframe(model, data, model.key(args.key).id)
    mujoco.viewer.launch(model, data)


if __name__ == "__main__":
    main()
