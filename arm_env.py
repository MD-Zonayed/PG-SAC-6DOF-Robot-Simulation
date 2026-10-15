"""Gymnasium environment for energy-aware control of a 6-actuated-joint arm.

This refactors the supplied research snippet around PyBullet's KUKA iiwa URDF.
The bundled URDF has seven revolute joints; this environment actuates joints 0-5
and actively holds joint 6 at zero to approximate the six-actuated-joint setup
used in the supplied code. This is a simulated setup, not a hardware controller.
"""
from __future__ import annotations

from collections import deque
from typing import Any

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pybullet as p
import pybullet_data


class ArmEnv(gym.Env):
    """Continuous-control task for an industrial-style manipulator in PyBullet."""

    metadata = {"render_modes": ["human"], "render_fps": 240}

    def __init__(
        self,
        render_mode: str | None = None,
        max_steps: int = 200,
        reward_weights: tuple[float, float, float] = (0.65, 0.20, 0.15),
        target_tolerance_m: float = 0.02,
    ) -> None:
        super().__init__()
        if render_mode not in (None, "human"):
            raise ValueError("render_mode must be None or 'human'.")
        if max_steps < 1:
            raise ValueError("max_steps must be >= 1")
        if len(reward_weights) != 3 or any(w < 0 for w in reward_weights):
            raise ValueError("reward_weights must contain three nonnegative values")

        self.render_mode = render_mode
        self.max_steps = int(max_steps)
        self.reward_weights = tuple(float(w) for w in reward_weights)
        self.target_tolerance_m = float(target_tolerance_m)
        self.dt = 1.0 / 240.0
        self.num_joints = 6
        self.locked_joint_index = 6
        self.end_effector_link_index = 6
        self.joint_limits = np.array(
            [
                [-2.96, 2.96],
                [-2.09, 2.09],
                [-2.96, 2.96],
                [-2.09, 2.09],
                [-2.96, 2.96],
                [-2.09, 2.09],
            ],
            dtype=np.float64,
        )
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(17,), dtype=np.float32
        )
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(6,), dtype=np.float32
        )

        connection_mode = p.GUI if render_mode == "human" else p.DIRECT
        self.client_id = p.connect(connection_mode)
        if self.client_id < 0:
            raise RuntimeError("Could not connect to PyBullet")
        p.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=self.client_id)

        self.robot_id: int | None = None
        self.target_pos = np.zeros(3, dtype=np.float64)
        self.step_count = 0
        self.energy_window: deque[float] = deque(maxlen=10)
        self.prev_torques = np.zeros(self.num_joints, dtype=np.float64)
        self.prev_joint_velocities = np.zeros(self.num_joints, dtype=np.float64)
        self.prev_joint_accelerations = np.zeros(self.num_joints, dtype=np.float64)
        self.prev_error: float | None = None
        self.total_energy = 0.0
        self.total_jerk = 0.0
        self.episode_success = False
        self._load_world()

    def _load_world(self) -> None:
        p.resetSimulation(physicsClientId=self.client_id)
        p.setGravity(0, 0, -9.81, physicsClientId=self.client_id)
        p.setTimeStep(self.dt, physicsClientId=self.client_id)
        p.loadURDF("plane.urdf", physicsClientId=self.client_id)
        self.robot_id = p.loadURDF(
            "kuka_iiwa/model.urdf",
            [0, 0, 0],
            useFixedBase=True,
            physicsClientId=self.client_id,
        )

    def _set_locked_joint(self) -> None:
        assert self.robot_id is not None
        p.setJointMotorControl2(
            self.robot_id,
            self.locked_joint_index,
            p.POSITION_CONTROL,
            targetPosition=0.0,
            force=200.0,
            maxVelocity=1.0,
            physicsClientId=self.client_id,
        )

    def _get_joint_states(self) -> list[tuple[Any, ...]]:
        assert self.robot_id is not None
        return [
            p.getJointState(self.robot_id, i, physicsClientId=self.client_id)
            for i in range(self.num_joints)
        ]

    def _get_ee_pos(self) -> np.ndarray:
        assert self.robot_id is not None
        state = p.getLinkState(
            self.robot_id,
            self.end_effector_link_index,
            computeForwardKinematics=True,
            physicsClientId=self.client_id,
        )
        return np.asarray(state[4], dtype=np.float64)  # world link-frame position

    def _sample_reachable_target(self) -> np.ndarray:
        """Sample a reachable end-effector position from a random joint pose."""
        assert self.robot_id is not None
        random_angles = self.np_random.uniform(
            self.joint_limits[:, 0] * 0.5, self.joint_limits[:, 1] * 0.5
        )
        for idx, angle in enumerate(random_angles):
            p.resetJointState(
                self.robot_id, idx, float(angle), physicsClientId=self.client_id
            )
        p.resetJointState(
            self.robot_id, self.locked_joint_index, 0.0, physicsClientId=self.client_id
        )
        p.performCollisionDetection(physicsClientId=self.client_id)
        return self._get_ee_pos()

    def _get_state(self) -> np.ndarray:
        joint_states = self._get_joint_states()
        positions = np.asarray([item[0] for item in joint_states], dtype=np.float64)
        velocities = np.asarray([item[1] for item in joint_states], dtype=np.float64)
        ee_pos = self._get_ee_pos()
        displacement = self.target_pos - ee_pos
        distance = float(np.linalg.norm(displacement))
        energy_history = float(np.mean(self.energy_window)) if self.energy_window else 0.0
        return np.concatenate(
            [positions, velocities, displacement, [distance, energy_history]]
        ).astype(np.float32)

    def reset(self, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        self._load_world()
        assert self.robot_id is not None

        # Set a reachable target, then place the robot at a different initial pose.
        self.target_pos = self._sample_reachable_target()
        starts = self.np_random.uniform(
            self.joint_limits[:, 0] * 0.3, self.joint_limits[:, 1] * 0.3
        )
        for idx, start in enumerate(starts):
            p.resetJointState(
                self.robot_id, idx, float(start), physicsClientId=self.client_id
            )
        p.resetJointState(
            self.robot_id, self.locked_joint_index, 0.0, physicsClientId=self.client_id
        )
        self._set_locked_joint()
        p.performCollisionDetection(physicsClientId=self.client_id)

        self.step_count = 0
        self.energy_window.clear()
        self.prev_torques = np.zeros(self.num_joints, dtype=np.float64)
        self.prev_joint_velocities = np.zeros(self.num_joints, dtype=np.float64)
        self.prev_joint_accelerations = np.zeros(self.num_joints, dtype=np.float64)
        self.prev_error = None
        self.total_energy = 0.0
        self.total_jerk = 0.0
        self.episode_success = False
        return self._get_state(), {"target_pos_m": self.target_pos.copy()}

    def step(self, action: np.ndarray):
        assert self.robot_id is not None
        action = np.asarray(action, dtype=np.float64).reshape(6)
        action = np.clip(action, self.action_space.low, self.action_space.high)
        states_before = self._get_joint_states()
        current_positions = np.asarray([item[0] for item in states_before])
        target_positions = np.clip(
            current_positions + action * 0.1,
            self.joint_limits[:, 0],
            self.joint_limits[:, 1],
        )

        for idx, target in enumerate(target_positions):
            p.setJointMotorControl2(
                self.robot_id,
                idx,
                p.POSITION_CONTROL,
                targetPosition=float(target),
                force=200.0,
                maxVelocity=2.0,
                physicsClientId=self.client_id,
            )
        self._set_locked_joint()
        p.stepSimulation(physicsClientId=self.client_id)
        self.step_count += 1

        joint_states = self._get_joint_states()
        torques = np.asarray([item[3] for item in joint_states], dtype=np.float64)
        velocities = np.asarray([item[1] for item in joint_states], dtype=np.float64)
        accelerations = (velocities - self.prev_joint_velocities) / self.dt
        joint_jerk = (accelerations - self.prev_joint_accelerations) / self.dt
        jerk_rad_s3 = float(np.mean(np.abs(joint_jerk)))
        # Mechanical work estimate: sum_i |tau_i * qdot_i| * dt.
        energy_j = float(np.sum(np.abs(torques * velocities)) * self.dt)
        self.energy_window.append(energy_j)
        self.total_energy += energy_j
        self.total_jerk += jerk_rad_s3

        self.prev_torques = torques.copy()
        self.prev_joint_velocities = velocities.copy()
        self.prev_joint_accelerations = accelerations.copy()

        ee_pos = self._get_ee_pos()
        error_m = float(np.linalg.norm(self.target_pos - ee_pos))
        error_mm = error_m * 1000.0
        w_tracking, w_energy, w_jerk = self.reward_weights

        # Scaled terms prevent the raw metrics from differing by many orders of magnitude.
        tracking_cost = error_m * 20.0
        energy_cost = energy_j * 10.0
        jerk_cost = jerk_rad_s3 * 1e-6
        progress_reward = 0.0 if self.prev_error is None else (self.prev_error - error_m) * 30.0
        success = error_m < 0.05
        reward = -(
            w_tracking * tracking_cost
            + w_energy * energy_cost
            + w_jerk * jerk_cost
        ) + progress_reward + (50.0 if error_m < self.target_tolerance_m else 0.0)
        self.prev_error = error_m
        self.episode_success = self.episode_success or success

        terminated = error_m < self.target_tolerance_m
        truncated = self.step_count >= self.max_steps
        info = {
            "tracking_error_mm": error_mm,
            "energy_J": energy_j,
            "jerk_rad_s3": jerk_rad_s3,
            "mean_abs_torque_Nm": float(np.mean(np.abs(torques))),
            "success": success,
            "episode_success": self.episode_success,
            "episode_energy_J": self.total_energy,
            "episode_mean_jerk_rad_s3": self.total_jerk / max(self.step_count, 1),
        }
        return self._get_state(), float(reward), terminated, truncated, info

    def render(self):
        # GUI rendering is provided by PyBullet when render_mode="human".
        return None

    def close(self) -> None:
        if getattr(self, "client_id", None) is not None and p.isConnected(self.client_id):
            p.disconnect(physicsClientId=self.client_id)
            self.client_id = None
