"""Run a simple PyBullet position-control diagnostic and save measured metrics.

This script is a standalone simulation diagnostic, not the SAC training loop.
It records measured PyBullet outputs rather than generating random graph data.

Usage:
    python -m src.simulation.run_pybullet_diagnostics --steps 1000
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import pybullet as p
import pybullet_data


def moving_average(values: np.ndarray, window: int = 15) -> np.ndarray:
    if len(values) == 0:
        return values
    window = max(1, min(int(window), len(values)))
    return np.convolve(values, np.ones(window) / window, mode="same")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--output", default="results/simulation_metrics.csv")
    parser.add_argument("--gui", action="store_true", help="Show PyBullet GUI")
    args = parser.parse_args()
    if args.steps < 1:
        raise ValueError("--steps must be positive")

    client = p.connect(p.GUI if args.gui else p.DIRECT)
    if client < 0:
        raise RuntimeError("Could not connect to PyBullet")

    dt = 1.0 / 240.0
    n_actuated = 6
    locked_joint = 6
    ee_link = 6
    target_pos = np.asarray([0.5, 0.2, 0.5], dtype=float)
    records: list[dict[str, float | int]] = []
    torque_history = np.zeros((args.steps, n_actuated), dtype=float)
    prev_vel = np.zeros(n_actuated, dtype=float)
    prev_acc = np.zeros(n_actuated, dtype=float)
    cumulative_energy = 0.0

    try:
        p.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=client)
        p.setGravity(0, 0, -9.81, physicsClientId=client)
        p.setTimeStep(dt, physicsClientId=client)
        p.loadURDF("plane.urdf", physicsClientId=client)
        robot = p.loadURDF("kuka_iiwa/model.urdf", [0, 0, 0], useFixedBase=True, physicsClientId=client)
        print("Simulation running...")

        for step in range(args.steps):
            ik = p.calculateInverseKinematics(robot, ee_link, target_pos.tolist(), physicsClientId=client)
            for joint_index in range(n_actuated):
                p.setJointMotorControl2(
                    robot, joint_index, p.POSITION_CONTROL,
                    targetPosition=float(ik[joint_index]), force=200.0,
                    maxVelocity=2.0, physicsClientId=client,
                )
            # The KUKA iiwa URDF has seven revolute joints. Hold the seventh at zero
            # to match the six-actuated-joint approximation in the research snippet.
            p.setJointMotorControl2(
                robot, locked_joint, p.POSITION_CONTROL,
                targetPosition=0.0, force=200.0, maxVelocity=1.0,
                physicsClientId=client,
            )
            p.stepSimulation(physicsClientId=client)

            joint_states = [p.getJointState(robot, i, physicsClientId=client) for i in range(n_actuated)]
            torques = np.asarray([state[3] for state in joint_states], dtype=float)
            velocities = np.asarray([state[1] for state in joint_states], dtype=float)
            accelerations = (velocities - prev_vel) / dt
            jerk = (accelerations - prev_acc) / dt
            mean_jerk = float(np.mean(np.abs(jerk)))
            energy = float(np.sum(np.abs(torques * velocities)) * dt)
            cumulative_energy += energy
            ee_state = p.getLinkState(robot, ee_link, computeForwardKinematics=True, physicsClientId=client)
            ee_pos = np.asarray(ee_state[4], dtype=float)
            error_mm = float(np.linalg.norm(target_pos - ee_pos) * 1000.0)
            torque_history[step] = torques
            records.append({
                "step": step,
                "time_s": step * dt,
                "tracking_error_mm": error_mm,
                "energy_step_J": energy,
                "cumulative_energy_J": cumulative_energy,
                "mean_abs_jerk_rad_s3": mean_jerk,
                **{f"joint_{i + 1}_torque_Nm": float(torques[i]) for i in range(n_actuated)},
            })
            prev_vel = velocities.copy()
            prev_acc = accelerations.copy()

        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=list(records[0].keys()))
            writer.writeheader()
            writer.writerows(records)
        print(f"Saved measured simulation metrics to {output}")
        print(f"Final tracking error: {records[-1]['tracking_error_mm']:.2f} mm")
        print(f"Cumulative mechanical energy estimate: {cumulative_energy:.4f} J")
    finally:
        p.disconnect(physicsClientId=client)


if __name__ == "__main__":
    main()
