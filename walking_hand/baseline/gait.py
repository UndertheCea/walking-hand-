"""Phase-clock walking on index/middle tips with lean feedback.

Default ``mode="elevated"``: low-frequency IK with stance + swing tip targets so
the palm actually translates (~1 cm/s, ~2 cm/cycle) while staying ~10 cm high.

``mode="fast"``: aggressive joint crawl (~3 cm/s) that drops the palm — opt-in only.
"""
from __future__ import annotations

import mujoco
import numpy as np

from walking_hand.baseline.balance import (
    CTRL_HI,
    CTRL_LO,
    STAND_CTRL,
    BalanceHoldController,
    ctrl_to_action,
)
from walking_hand.baseline.ik import FINGERS, FingerIK


def _smooth01(s: float) -> float:
    s = float(np.clip(s, 0.0, 1.0))
    return s * s * (3.0 - 2.0 * s)


class WalkController:
    """Two-finger walk → 6-D gym action.

    ``mode="elevated"`` (default)
        Stance tip sweeps rear under the palm (push) while the swing tip lifts
        forward. Palm stays ~10 cm and advances ~1 cm/s at 0.5 Hz.

    ``mode="fast"``
        Joint crawl (~3 cm/s) that sinks the palm. Use only when height is OK to drop.
    """

    def __init__(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        dt: float,
        mode: str = "elevated",
        # Elevated IK: short stance/swing travel keeps palm high and moves forward.
        frequency: float = 0.5,
        step_length: float = 0.025,
        step_height: float = 0.012,
        duty: float = 0.55,
        max_joint_delta: float = 0.45,
        min_palm_height: float = 0.075,
        # Fast joint-crawl (palm drops) — opt-in via mode="fast".
        mcp_amp: float = 0.35,
        pip_amp: float = 0.25,
        abd_amp: float = 0.05,
        balance_scale: float = 1.0,
        balance: BalanceHoldController | None = None,
        use_ik: bool | None = None,
    ):
        self.model = model
        self.data = data
        self.dt = float(dt)
        if use_ik is True:
            mode = "elevated"
        elif use_ik is False:
            mode = "fast"
        self.mode = mode
        self.omega = 2.0 * np.pi * frequency
        self.step_length = step_length
        self.step_height = step_height
        self.duty = duty
        self.max_joint_delta = max_joint_delta
        self.min_palm_height = min_palm_height
        self.mcp_amp = mcp_amp
        self.pip_amp = pip_amp
        self.abd_amp = abd_amp
        self.balance_scale = balance_scale
        self.balance = balance or BalanceHoldController()
        self.stand_ctrl = STAND_CTRL.copy()
        self.phase = 0.0
        self.ik = FingerIK(model, data)
        self.palm_id = model.body("palm").id
        self.tip_palm0 = {f: None for f in FINGERS}
        self._initialized = False

    def reset(self):
        mujoco.mj_forward(self.model, self.data)
        R = self.data.xmat[self.palm_id].reshape(3, 3)
        p = self.data.xpos[self.palm_id]
        for f in FINGERS:
            tip = self.ik.tip_world(f)
            self.tip_palm0[f] = R.T @ (tip - p)
        self.phase = 0.0
        self._initialized = True

    def _leg_phase(self, phi: float) -> tuple[str, float]:
        """Return ('swing'|'stance', fraction in [0, 1]) for one finger."""
        phi = phi % (2.0 * np.pi)
        swing = (1.0 - self.duty) * 2.0 * np.pi
        if phi < swing:
            return "swing", phi / max(swing, 1e-9)
        return "stance", (phi - swing) / max(2.0 * np.pi - swing, 1e-9)

    def _hold_stand(self, obs: np.ndarray) -> np.ndarray:
        q = np.clip(self.stand_ctrl + self.balance.deltas(obs), CTRL_LO, CTRL_HI)
        return ctrl_to_action(q).astype(np.float64)

    def _elevated_ik(self, obs: np.ndarray) -> np.ndarray:
        # Never let the palm collapse: recover on the stand pose if too low.
        if float(obs[21]) < self.min_palm_height:
            self.phase += 0.15 * self.omega * self.dt
            return self._hold_stand(obs)

        R = self.data.xmat[self.palm_id].reshape(3, 3)
        p = self.data.xpos[self.palm_id]
        # Walk along world +x, expressed in the palm frame.
        ex = R.T @ np.array([1.0, 0.0, 0.0])
        L = self.step_length
        H = self.step_height

        targets: dict[str, np.ndarray] = {}
        mask = np.ones(6, dtype=np.float64)
        for f, offset, sl in (
            ("index", 0.0, slice(0, 3)),
            ("middle", np.pi, slice(3, 6)),
        ):
            mode, s = self._leg_phase(self.phase + offset)
            u = _smooth01(s)
            t0 = self.tip_palm0[f]
            if mode == "swing":
                # Rear → front + lift: place the next foothold ahead.
                tip_p = t0 - 0.5 * L * ex + L * u * ex
                tip_p = tip_p + np.array([0.0, 0.0, H * np.sin(np.pi * u)])
                mask[sl] = 0.2
            else:
                # Front → rear under the palm: stance push that translates the body.
                tip_p = t0 + 0.5 * L * ex - L * u * ex
                tip_p = tip_p.copy()
                tip_p[2] = t0[2]
            targets[f] = p + R @ tip_p

        q_ik = self.ik.solve(targets, q_seed=np.asarray(obs[0:6], dtype=np.float64))
        dq = np.clip(q_ik - self.stand_ctrl, -self.max_joint_delta, self.max_joint_delta)
        q = np.clip(
            self.stand_ctrl + dq + self.balance_scale * self.balance.deltas(obs) * mask,
            CTRL_LO,
            CTRL_HI,
        )
        self.phase += self.omega * self.dt
        return ctrl_to_action(q).astype(np.float64)

    def _fast_crawl(self, obs: np.ndarray) -> np.ndarray:
        # Hard stop if the palm is dropping — prefer upright stand over speed.
        if float(obs[21]) < self.min_palm_height:
            self.phase += 0.15 * self.omega * self.dt
            return self._hold_stand(obs)

        q = self.stand_ctrl.copy()
        s_i = np.sin(self.phase)
        s_m = np.sin(self.phase + np.pi)
        q[0] += self.abd_amp * s_i
        q[3] += self.abd_amp * s_m
        q[1] += self.mcp_amp * s_i
        q[4] += self.mcp_amp * s_m
        q[2] += self.pip_amp * max(0.0, s_i)
        q[5] += self.pip_amp * max(0.0, s_m)
        q = np.clip(q + self.balance_scale * self.balance.deltas(obs), CTRL_LO, CTRL_HI)
        self.phase += self.omega * self.dt
        return ctrl_to_action(q).astype(np.float64)

    def __call__(self, obs: np.ndarray) -> np.ndarray:
        if not self._initialized:
            self.reset()
        if self.mode == "fast":
            return self._fast_crawl(obs)
        return self._elevated_ik(obs)
