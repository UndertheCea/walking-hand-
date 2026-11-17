"""Lean-feedback stand controller for two-finger balance (no walking clock).

Holds the stand joint targets and adds small MCP/PIP/abd offsets from the
palm-frame gravity direction (and optional gyro damping) so the fingertips
stay under the mass. Designed for ``stub_contact=False``.

Observation layout matches ``ThingHandWalkEnv._get_obs`` (non-tactile):
    q (6), qd (6), g_palm (3), gyro (3), v_palm (3), height (1), prev_action (6)
"""
from __future__ import annotations

import numpy as np

# Tall two-finger stand ctrl from models/thing_hand/scene.xml (index then middle: abd, mcp, pip).
STAND_CTRL = np.array([-0.0727, 0.7260, 0.0632, -0.1129, 0.9953, 0.1237], dtype=np.float64)
# Gravity direction in the palm frame at that stand (pitch ≈ 0.84 rad). Lean feedback
# regulates around this, not [0, 0, -1], or the controller fights the nominal pose.
STAND_G_PALM = np.array([0.7448, 0.0, -0.6673], dtype=np.float64)

# Actuator ctrlrange for the six position actuators (same order).
CTRL_LO = np.array([-0.35, -0.3, 0.0, -0.35, -0.3, 0.0], dtype=np.float64)
CTRL_HI = np.array([0.35, 1.6, 1.6, 0.35, 1.6, 1.6], dtype=np.float64)

# Obs slices.
_G_PALM = slice(12, 15)
_GYRO = slice(15, 18)


def ctrl_to_action(ctrl: np.ndarray) -> np.ndarray:
    """Map joint targets in radians onto the env's [-1, 1] action space."""
    return 2.0 * (ctrl - CTRL_LO) / (CTRL_HI - CTRL_LO) - 1.0


def action_to_ctrl(action: np.ndarray) -> np.ndarray:
    a = np.clip(action, -1.0, 1.0)
    return CTRL_LO + (a + 1.0) * 0.5 * (CTRL_HI - CTRL_LO)


class BalanceHoldController:
    """Stand pose + lean feedback → 6-D gym action.

    Palm frame: +x fingers/forward, +y thumb side, +z back of hand.
    Lean is gravity-in-palm minus ``STAND_G_PALM`` (the pitched stand), plus gyro damping.
    """

    def __init__(
        self,
        stand_ctrl: np.ndarray | None = None,
        stand_g_palm: np.ndarray | None = None,
        # Pitch (forward/back): positive lean_x → flex MCP / ease PIP (tips under CoM).
        k_pitch_mcp: float = 0.45,
        k_pitch_pip: float = 0.25,
        # Roll (side): positive lean_y → open abd and load the opposite finger's MCP.
        k_roll_abd: float = 0.35,
        k_roll_mcp: float = 0.30,
        # Gyro damping (palm frame, rad/s).
        k_gyro_pitch: float = 0.08,
        k_gyro_roll: float = 0.08,
        # Clip corrective deltas in radians before adding to stand.
        max_delta: float = 0.35,
    ):
        self.stand_ctrl = np.asarray(
            STAND_CTRL if stand_ctrl is None else stand_ctrl, dtype=np.float64
        ).copy()
        self.stand_g_palm = np.asarray(
            STAND_G_PALM if stand_g_palm is None else stand_g_palm, dtype=np.float64
        ).copy()
        self.k_pitch_mcp = k_pitch_mcp
        self.k_pitch_pip = k_pitch_pip
        self.k_roll_abd = k_roll_abd
        self.k_roll_mcp = k_roll_mcp
        self.k_gyro_pitch = k_gyro_pitch
        self.k_gyro_roll = k_gyro_roll
        self.max_delta = max_delta
        self.nominal_action = ctrl_to_action(self.stand_ctrl)

    def deltas(self, obs: np.ndarray) -> np.ndarray:
        """Return 6-D joint-space correction (radians) from lean + gyro."""
        g = np.asarray(obs[_G_PALM], dtype=np.float64)
        w = np.asarray(obs[_GYRO], dtype=np.float64)
        lean = g - self.stand_g_palm
        # Pitch about palm y: lean_x and omega_y. Roll about palm x: lean_y and omega_x.
        pitch = lean[0] + self.k_gyro_pitch * w[1]
        roll = lean[1] + self.k_gyro_roll * w[0]

        d = np.zeros(6, dtype=np.float64)
        # index: 0 abd, 1 mcp, 2 pip; middle: 3 abd, 4 mcp, 5 pip
        d[1] += self.k_pitch_mcp * pitch
        d[4] += self.k_pitch_mcp * pitch
        d[2] -= self.k_pitch_pip * pitch
        d[5] -= self.k_pitch_pip * pitch

        d[0] += self.k_roll_abd * roll
        d[3] += self.k_roll_abd * roll
        # Differential MCP: lean toward +y (thumb) loads middle, unloads index.
        d[1] -= self.k_roll_mcp * roll
        d[4] += self.k_roll_mcp * roll

        return np.clip(d, -self.max_delta, self.max_delta)

    def __call__(self, obs: np.ndarray) -> np.ndarray:
        ctrl = np.clip(self.stand_ctrl + self.deltas(obs), CTRL_LO, CTRL_HI)
        return ctrl_to_action(ctrl).astype(np.float64)
