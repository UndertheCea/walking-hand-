"""Jacobian IK for index/middle fingertips (DIP = 0.7 · PIP).

Solves for the 3 actuated joints per finger (abd, mcp, pip) so each tip
tracks a world-frame target. Uses MuJoCo site Jacobians and least squares.
"""
from __future__ import annotations

import mujoco
import numpy as np

from walking_hand.baseline.balance import CTRL_HI, CTRL_LO

COUPLE = 0.7
FINGERS = ("index", "middle")
# Actuated joint names per finger (abd, mcp, pip); DIP follows PIP.
_JOINTS = {
    "index": ("index_abd", "index_mcp", "index_pip"),
    "middle": ("middle_abd", "middle_mcp", "middle_pip"),
}


class FingerIK:
    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData, max_iters: int = 12, step: float = 0.6):
        self.model = model
        self.data = data
        self.max_iters = max_iters
        self.step = step
        self.site_id = {f: model.site(f"{f}_tip").id for f in FINGERS}
        self.dof = {
            f: np.array([model.jnt_dofadr[model.joint(j).id] for j in _JOINTS[f]]) for f in FINGERS
        }
        self.qadr = {
            f: np.array([model.jnt_qposadr[model.joint(j).id] for j in _JOINTS[f]]) for f in FINGERS
        }
        self.dip_qadr = {
            f: model.jnt_qposadr[model.joint(f"{f}_dip").id] for f in FINGERS
        }
        # Actuator order: index abd/mcp/pip, middle abd/mcp/pip.
        self.act_qadr = np.concatenate([self.qadr["index"], self.qadr["middle"]])

    def _sync_dip(self):
        d = self.data
        for f in FINGERS:
            pip = d.qpos[self.qadr[f][2]]
            d.qpos[self.dip_qadr[f]] = COUPLE * pip

    def tip_world(self, finger: str) -> np.ndarray:
        return self.data.site_xpos[self.site_id[finger]].copy()

    def solve(self, targets_world: dict[str, np.ndarray], q_seed: np.ndarray | None = None) -> np.ndarray:
        """Return 6-D actuated joint targets (index then middle) for tip goals.

        Temporarily writes ``data.qpos`` for forward kinematics, then restores
        the previous configuration so the live sim state is unchanged.
        """
        m, d = self.model, self.data
        qpos_save = d.qpos.copy()
        try:
            if q_seed is not None:
                d.qpos[self.act_qadr] = q_seed
                self._sync_dip()
            mujoco.mj_forward(m, d)

            jacp = np.zeros((3, m.nv))
            for _ in range(self.max_iters):
                total = 0.0
                for f, target in targets_world.items():
                    err = target - d.site_xpos[self.site_id[f]]
                    total += float(err @ err)
                    mujoco.mj_jacSite(m, d, jacp, None, self.site_id[f])
                    J = jacp[:, self.dof[f]]
                    dq, *_ = np.linalg.lstsq(J, err, rcond=None)
                    d.qpos[self.qadr[f]] += self.step * dq
                    lo = CTRL_LO[0:3] if f == "index" else CTRL_LO[3:6]
                    hi = CTRL_HI[0:3] if f == "index" else CTRL_HI[3:6]
                    d.qpos[self.qadr[f]] = np.clip(d.qpos[self.qadr[f]], lo, hi)
                    self._sync_dip()
                mujoco.mj_forward(m, d)
                if total < 1e-8:
                    break

            return np.concatenate(
                [d.qpos[self.qadr["index"]], d.qpos[self.qadr["middle"]]]
            ).copy()
        finally:
            d.qpos[:] = qpos_save
            mujoco.mj_forward(m, d)
