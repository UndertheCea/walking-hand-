"""Search for a two-finger stand keyframe on the planar base.

Seats both tips on the floor with the projected CoM between them, then holds
position actuators and scores survival with palm-height / upright thresholds.

Usage:
    PYTHONPATH=. python scripts/search_two_finger_stand.py [--trials 3000] [--hold 20]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import mujoco
import numpy as np

SCENE = Path(__file__).resolve().parents[1] / "models" / "thing_hand" / "scene.xml"
ACT = ["index_abd", "index_mcp", "index_pip", "middle_abd", "middle_mcp", "middle_pip"]
COUPLE = 0.7


def seat(model, data, qadr, pitch, finger):
    x_adr = model.jnt_qposadr[model.joint("root_x").id]
    z_adr = model.jnt_qposadr[model.joint("root_z").id]
    p_adr = model.jnt_qposadr[model.joint("root_pitch").id]
    mujoco.mj_resetData(model, data)
    data.qpos[x_adr] = 0.0
    data.qpos[z_adr] = 0.15
    data.qpos[p_adr] = pitch
    data.qpos[qadr] = finger
    for name, pip in (("index_dip", finger[2]), ("middle_dip", finger[5])):
        data.qpos[model.jnt_qposadr[model.joint(name).id]] = COUPLE * pip
    mujoco.mj_forward(model, data)
    zi = data.site_xpos[model.site("index_tip").id][2]
    zm = data.site_xpos[model.site("middle_tip").id][2]
    data.qpos[z_adr] -= 0.5 * (zi + zm)
    mujoco.mj_forward(model, data)
    for _ in range(10):
        ti = data.site_xpos[model.site("index_tip").id][0]
        tm = data.site_xpos[model.site("middle_tip").id][0]
        comx = data.subtree_com[0][0]
        data.qpos[x_adr] += comx - 0.5 * (ti + tm)
        mujoco.mj_forward(model, data)
    data.qvel[:] = 0.0
    data.ctrl[:] = finger


def hold(model, data, hold_s, min_height, min_up):
    steps = int(hold_s / model.opt.timestep)
    minh, minu = 1e9, 1e9
    for i in range(steps):
        mujoco.mj_step(model, data)
        R = data.site_xmat[model.site("imu").id].reshape(3, 3)
        u = float(R[2, 2])
        h = float(data.site_xpos[model.site("imu").id][2])
        minh, minu = min(minh, h), min(minu, u)
        if h < min_height or u < min_up:
            return i * model.opt.timestep, minh, minu
    return hold_s, minh, minu


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=3000)
    ap.add_argument("--hold", type=float, default=20.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--min-height", type=float, default=0.05)
    ap.add_argument("--min-upright", type=float, default=0.45)
    args = ap.parse_args()

    model = mujoco.MjModel.from_xml_path(str(SCENE))
    data = mujoco.MjData(model)
    if "root_x" not in {model.joint(i).name for i in range(model.njnt)}:
        raise SystemExit("scene must use the planar base (root_x / root_z / root_pitch)")

    qadr = np.array([model.jnt_qposadr[model.joint(j).id] for j in ACT])
    lo = model.actuator_ctrlrange[:, 0].copy()
    hi = model.actuator_ctrlrange[:, 1].copy()
    rng = np.random.default_rng(args.seed)

    best_t = -1.0
    best = None
    for n in range(args.trials):
        finger = np.clip(
            np.array(
                [
                    rng.uniform(-0.2, 0.2),
                    rng.uniform(0.3, 1.15),
                    rng.uniform(0.35, 1.15),
                    rng.uniform(-0.2, 0.2),
                    rng.uniform(0.35, 1.25),
                    rng.uniform(0.3, 1.1),
                ]
            ),
            lo,
            hi,
        )
        pitch = rng.uniform(0.25, 0.9)
        seat(model, data, qadr, pitch, finger)
        ti = data.site_xpos[model.site("index_tip").id]
        tm = data.site_xpos[model.site("middle_tip").id]
        com = data.subtree_com[0]
        u0 = float(data.site_xmat[model.site("imu").id].reshape(3, 3)[2, 2])
        h0 = float(data.site_xpos[model.site("imu").id][2])
        if h0 < 0.055 or u0 < 0.55:
            continue
        if min(ti[2], tm[2]) > 0.012 or abs(ti[0] - tm[0]) < 0.01:
            continue
        lo_x, hi_x = sorted([ti[0], tm[0]])
        if not (lo_x + 0.002 <= com[0] <= hi_x - 0.002):
            continue

        q_save = data.qpos.copy()
        t_hold, minh, minu = hold(model, data, min(5.0, args.hold), args.min_height, args.min_upright)
        if t_hold > best_t:
            best_t = t_hold
            best = dict(
                t=t_hold,
                pitch=float(q_save[model.jnt_qposadr[model.joint("root_pitch").id]]),
                finger=finger.copy(),
                qpos=q_save,
                ctrl=finger.copy(),
                h0=h0,
                u0=u0,
                minh=minh,
                minu=minu,
            )
            print(
                f"[{n}] hold={t_hold:.2f}s pitch={best['pitch']:.3f} "
                f"h0={h0:.3f} u0={u0:.3f} minh={minh:.3f} finger={np.round(finger, 3)}"
            )
            if t_hold >= min(5.0, args.hold) - 1e-9:
                data.qpos[:] = q_save
                data.qvel[:] = 0.0
                data.ctrl[:] = finger
                mujoco.mj_forward(model, data)
                t_full, minh, minu = hold(model, data, args.hold, args.min_height, args.min_upright)
                print(f"  -> {args.hold:.0f}s check: {t_full:.2f}s minh={minh:.3f} minu={minu:.3f}")
                best["t"] = t_full
                best_t = t_full
                if t_full >= args.hold - 1e-9:
                    break

    if best is None:
        print("no candidate passed geometric filters")
        return
    print("\nBEST")
    print(f"  hold={best['t']:.2f}s pitch={best['pitch']:.4f}")
    print(f"  ctrl={best['ctrl'].round(4).tolist()}")
    print(f"  qpos={best['qpos'].round(4).tolist()}")


if __name__ == "__main__":
    main()
