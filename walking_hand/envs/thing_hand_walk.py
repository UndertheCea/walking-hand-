"""Gymnasium env: the Thing hand walking forward on its index and middle fingers.

One class covers both RL variants in the proposal. They differ only in the
observation, so with the same seed and settings both see identical physics:

    tactile=False   joint state + palm IMU + previous action        (pure RL)
    tactile=True    the same + index/middle fingertip touch forces  (tactile RL)

Action: 6 values in [-1, 1], mapped linearly onto each position actuator's
ctrlrange (index/middle x abd, mcp, pip). DIP follows PIP through the model's
0.7 coupling. Control runs at 1 / (timestep * frame_skip) = 50 Hz by default.

World frame: +x is forward (the fingers point along +x at reset), +z is up.
"""
from __future__ import annotations

from pathlib import Path

import gymnasium as gym
import mujoco
import numpy as np
from gymnasium import spaces

DEFAULT_SCENE = Path(__file__).resolve().parents[2] / "models" / "thing_hand" / "scene.xml"

ACTUATED_JOINTS = ["index_abd", "index_mcp", "index_pip", "middle_abd", "middle_mcp", "middle_pip"]
TOUCH_SENSORS = ["index_touch", "middle_touch"]


class ThingHandWalkEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 50}

    def __init__(
        self,
        scene_path: str | Path = DEFAULT_SCENE,
        tactile: bool = False,
        frame_skip: int = 10,
        stub_contact: bool = False,
        # Reward weights (starting point, expect to tune).
        forward_reward_weight: float = 20.0,
        alive_bonus: float = 0.05,
        ctrl_cost_weight: float = 1e-3,
        action_rate_cost_weight: float = 1e-2,
        upright_cost_weight: float = 0.1,
        # Termination thresholds.
        min_palm_height: float = 0.03,
        min_upright: float = 0.3,
        # Randomization. friction_range=None keeps the scene's floor friction.
        friction_range: tuple[float, float] | None = None,
        reset_noise_scale: float = 0.02,
        # External pushes on the palm. push_force_range=None disables them.
        push_force_range: tuple[float, float] | None = None,
        push_interval_range: tuple[float, float] = (1.0, 3.0),
        push_duration: float = 0.1,
        tactile_scale: float = 1.0,
        render_mode: str | None = None,
        width: int = 640,
        height: int = 480,
    ):
        self.model = mujoco.MjModel.from_xml_path(str(scene_path))
        self.data = mujoco.MjData(self.model)
        m = self.model

        self.tactile = tactile
        self.frame_skip = frame_skip
        self.dt = m.opt.timestep * frame_skip
        self.metadata = {**self.metadata, "render_fps": int(round(1.0 / self.dt))}

        self.forward_reward_weight = forward_reward_weight
        self.alive_bonus = alive_bonus
        self.ctrl_cost_weight = ctrl_cost_weight
        self.action_rate_cost_weight = action_rate_cost_weight
        self.upright_cost_weight = upright_cost_weight
        self.min_palm_height = min_palm_height
        self.min_upright = min_upright
        self.friction_range = friction_range
        self.reset_noise_scale = reset_noise_scale
        self.push_force_range = push_force_range
        self.push_interval_range = push_interval_range
        self.push_duration = push_duration
        self.tactile_scale = tactile_scale

        # Indices into the model.
        self.palm_id = m.body("palm").id
        self.imu_site = m.site("imu").id
        self.floor_geom = m.geom("floor").id
        self.nominal_friction = m.geom_friction[self.floor_geom].copy()
        self.qadr = np.array([m.jnt_qposadr[m.joint(j).id] for j in ACTUATED_JOINTS])
        self.vadr = np.array([m.jnt_dofadr[m.joint(j).id] for j in ACTUATED_JOINTS])
        self.ctrl_lo = m.actuator_ctrlrange[:, 0].copy()
        self.ctrl_hi = m.actuator_ctrlrange[:, 1].copy()
        self.init_key = m.key("stand").id if m.nkey > 0 else None

        # Optional third support: let the forearm stub touch the floor.
        stub = m.geom("stub").id
        m.geom_contype[stub] = 1 if stub_contact else 0
        m.geom_conaffinity[stub] = 1 if stub_contact else 0

        self.action_space = spaces.Box(-1.0, 1.0, shape=(m.nu,), dtype=np.float32)
        obs_dim = 6 + 6 + 3 + 3 + 3 + 1 + m.nu + (len(TOUCH_SENSORS) if tactile else 0)
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(obs_dim,), dtype=np.float64)

        self.render_mode = render_mode
        self.width, self.height = width, height
        self._renderer = None
        self._viewer = None

        self.prev_action = np.zeros(m.nu)
        self.next_push_time = np.inf
        self.push_end_time = -1.0
        self.push_force = np.zeros(3)

    # ------------------------------------------------------------------ helpers
    def _action_to_ctrl(self, action):
        a = np.clip(action, -1.0, 1.0)
        return self.ctrl_lo + (a + 1.0) * 0.5 * (self.ctrl_hi - self.ctrl_lo)

    def _palm_rotation(self):
        return self.data.site_xmat[self.imu_site].reshape(3, 3)

    def _upright(self):
        """World-z component of the back-of-hand axis: 1 = palm facing the floor, 0 = on its side."""
        return float(self._palm_rotation()[2, 2])

    def _palm_height(self):
        return float(self.data.site_xpos[self.imu_site][2])

    def _touch(self):
        return np.array([self.data.sensor(s).data[0] for s in TOUCH_SENSORS])

    def _get_obs(self):
        d = self.data
        R = self._palm_rotation()
        parts = [
            d.qpos[self.qadr],                 # 6 actuated joint angles
            d.qvel[self.vadr],                 # 6 actuated joint velocities
            R.T @ np.array([0.0, 0.0, -1.0]),  # 3 gravity direction in the palm frame
            d.sensor("palm_gyro").data,        # 3 palm angular velocity (palm frame)
            R.T @ d.qvel[0:3],                 # 3 base linear velocity (palm frame)
            [self._palm_height()],             # 1 palm height
            self.prev_action,                  # 6 previous action
        ]
        if self.tactile:
            parts.append(np.log1p(self.tactile_scale * self._touch()))  # 2 fingertip forces
        return np.concatenate(parts)

    def _schedule_push(self):
        if self.push_force_range is None:
            self.next_push_time = np.inf
        else:
            self.next_push_time = self.data.time + self.np_random.uniform(*self.push_interval_range)

    def _apply_pushes(self):
        d = self.data
        if d.time >= self.next_push_time:
            mag = self.np_random.uniform(*self.push_force_range)
            ang = self.np_random.uniform(0, 2 * np.pi)
            self.push_force = mag * np.array([np.cos(ang), np.sin(ang), 0.0])
            self.push_end_time = d.time + self.push_duration
            self._schedule_push()
        if d.time < self.push_end_time:
            d.xfrc_applied[self.palm_id, :3] = self.push_force
        else:
            d.xfrc_applied[self.palm_id, :3] = 0.0

    # ---------------------------------------------------------------- gym API
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        options = options or {}
        m, d = self.model, self.data

        if self.init_key is not None:
            mujoco.mj_resetDataKeyframe(m, d, self.init_key)
        else:
            mujoco.mj_resetData(m, d)
        d.qpos[self.qadr] += self.np_random.uniform(-1, 1, size=len(self.qadr)) * self.reset_noise_scale

        # Floor friction: explicit option (for evaluation) > random range > nominal.
        friction = self.nominal_friction.copy()
        if "friction" in options:
            friction[0] = options["friction"]
        elif self.friction_range is not None:
            friction[0] = self.np_random.uniform(*self.friction_range)
        m.geom_friction[self.floor_geom] = friction

        d.xfrc_applied[:] = 0.0
        self.push_end_time = -1.0
        self._schedule_push()

        # Start from the actuators holding the reset pose.
        hold = np.clip(d.qpos[self.qadr], self.ctrl_lo, self.ctrl_hi)
        d.ctrl[:] = hold
        self.prev_action = 2.0 * (hold - self.ctrl_lo) / (self.ctrl_hi - self.ctrl_lo) - 1.0
        mujoco.mj_forward(m, d)

        if self.render_mode == "human":
            self.render()
        return self._get_obs(), self._get_info(friction[0])

    def step(self, action):
        action = np.asarray(action, dtype=np.float64)
        d = self.data
        x_before = d.qpos[0]
        d.ctrl[:] = self._action_to_ctrl(action)
        for _ in range(self.frame_skip):
            self._apply_pushes()
            mujoco.mj_step(self.model, d)
        x_vel = (d.qpos[0] - x_before) / self.dt

        upright = self._upright()
        height = self._palm_height()
        terminated = bool(height < self.min_palm_height or upright < self.min_upright)

        forward = self.forward_reward_weight * x_vel
        ctrl_cost = self.ctrl_cost_weight * float(np.sum(d.actuator_force ** 2))
        rate_cost = self.action_rate_cost_weight * float(np.sum((action - self.prev_action) ** 2))
        upright_cost = self.upright_cost_weight * (1.0 - upright)
        alive = 0.0 if terminated else self.alive_bonus
        reward = forward + alive - ctrl_cost - rate_cost - upright_cost
        self.prev_action = np.clip(action, -1.0, 1.0)

        info = self._get_info(self.model.geom_friction[self.floor_geom, 0])
        info.update(
            x_velocity=x_vel,
            reward_forward=forward,
            reward_alive=alive,
            cost_ctrl=ctrl_cost,
            cost_action_rate=rate_cost,
            cost_upright=upright_cost,
        )
        if self.render_mode == "human":
            self.render()
        return self._get_obs(), reward, terminated, False, info

    def _get_info(self, friction):
        return {
            "x_position": float(self.data.qpos[0]),
            "palm_height": self._palm_height(),
            "upright": self._upright(),
            "touch": self._touch(),
            "friction": float(friction),
            "pushing": bool(self.data.time < self.push_end_time),
        }

    def render(self):
        if self.render_mode == "rgb_array":
            if self._renderer is None:
                self.model.vis.global_.offwidth = max(self.model.vis.global_.offwidth, self.width)
                self.model.vis.global_.offheight = max(self.model.vis.global_.offheight, self.height)
                self._renderer = mujoco.Renderer(self.model, self.height, self.width)
            self._renderer.update_scene(self.data)
            return self._renderer.render()
        if self.render_mode == "human":
            if self._viewer is None:
                from mujoco import viewer  # needs a display; only loaded for "human"

                self._viewer = viewer.launch_passive(self.model, self.data)
            self._viewer.sync()
        return None

    def close(self):
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None
        if self._viewer is not None:
            self._viewer.close()
            self._viewer = None
