"""Render a MuJoCo scene offscreen to a PNG (no display needed).

Usage: MUJOCO_GL=osmesa python scripts/render.py <scene.xml> <out.png>
           [--key NAME] [--seconds T] [--width W] [--height H]

Loads the first keyframe (or --key), simulates for --seconds, and saves one
frame from the scene's default free camera.
"""
import argparse
import struct
import zlib

import mujoco


def write_png(path, rgb):
    """Write an HxWx3 uint8 array as PNG using only the standard library."""
    h, w, _ = rgb.shape
    raw = b"".join(b"\x00" + rgb[y].tobytes() for y in range(h))

    def chunk(tag, payload):
        body = tag + payload
        return struct.pack(">I", len(payload)) + body + struct.pack(">I", zlib.crc32(body))

    with open(path, "wb") as f:
        f.write(b"\x89PNG\r\n\x1a\n")
        f.write(chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0)))
        f.write(chunk(b"IDAT", zlib.compress(raw, 6)))
        f.write(chunk(b"IEND", b""))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("scene")
    ap.add_argument("out")
    ap.add_argument("--key", default=None)
    ap.add_argument("--seconds", type=float, default=0.0)
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=480)
    args = ap.parse_args()

    model = mujoco.MjModel.from_xml_path(args.scene)
    model.vis.global_.offwidth = max(model.vis.global_.offwidth, args.width)
    model.vis.global_.offheight = max(model.vis.global_.offheight, args.height)
    data = mujoco.MjData(model)
    if args.key is None and model.nkey > 0:
        mujoco.mj_resetDataKeyframe(model, data, 0)
    elif args.key:
        mujoco.mj_resetDataKeyframe(model, data, model.key(args.key).id)
    mujoco.mj_forward(model, data)
    while data.time < args.seconds:
        mujoco.mj_step(model, data)

    with mujoco.Renderer(model, args.height, args.width) as renderer:
        renderer.update_scene(data)
        write_png(args.out, renderer.render())
    print(f"wrote {args.out} at t={data.time:.2f}s")


if __name__ == "__main__":
    main()
