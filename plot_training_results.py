"""Plot measured metrics saved by the training callback and PyBullet diagnostic.

Run from repository root:
    python -m src.analysis.plot_training_results

The script only plots real values read from CSV files; it does not synthesize data.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def rolling_mean(values: pd.Series, window: int = 20) -> pd.Series:
    return values.rolling(window=max(1, window), min_periods=1).mean()


def save_line_plot(x, y, title: str, xlabel: str, ylabel: str, out: Path, label: str | None = None) -> None:
    plt.figure(figsize=(8, 5))
    plt.plot(x, y, linewidth=2, label=label)
    if label:
        plt.legend()
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=160, bbox_inches="tight")
    plt.close()
    print(f"Saved {out}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-csv", default="logs/training_log.csv")
    parser.add_argument("--simulation-csv", default="results/simulation_metrics.csv")
    parser.add_argument("--optuna-csv", default="results/optuna_trials.csv")
    parser.add_argument("--output-dir", default="results/figures")
    parser.add_argument("--window", type=int, default=20)
    args = parser.parse_args()

    out = Path(args.output_dir)
    training_path = Path(args.training_csv)
    if training_path.exists():
        data = pd.read_csv(training_path)
        if {"episode", "episode_reward"}.issubset(data.columns):
            save_line_plot(
                data["episode"], rolling_mean(data["episode_reward"], args.window),
                "Reward Convergence (SAC Training)", "Episode", "Episode Reward",
                out / "reward_convergence.png", "Rolling mean",
            )
        if {"episode", "mean_tracking_error_mm"}.issubset(data.columns):
            save_line_plot(
                data["episode"], rolling_mean(data["mean_tracking_error_mm"], args.window),
                "Tracking Error During Training", "Episode", "Mean Tracking Error (mm)",
                out / "tracking_error.png", "Rolling mean",
            )
        if {"episode", "episode_energy_J"}.issubset(data.columns):
            save_line_plot(
                data["episode"], rolling_mean(data["episode_energy_J"], args.window),
                "Episode Energy Estimate", "Episode", "Mechanical Energy Estimate (J)",
                out / "episode_energy.png", "Rolling mean",
            )
        if {"episode", "mean_jerk_rad_s3"}.issubset(data.columns):
            save_line_plot(
                data["episode"], rolling_mean(data["mean_jerk_rad_s3"], args.window),
                "Mean Joint Jerk During Training", "Episode", "Mean Absolute Jerk (rad/s³)",
                out / "episode_jerk.png", "Rolling mean",
            )
    else:
        print(f"Training CSV not found; skipping training plots: {training_path}")

    simulation_path = Path(args.simulation_csv)
    if simulation_path.exists():
        data = pd.read_csv(simulation_path)
        if {"time_s", "tracking_error_mm"}.issubset(data.columns):
            save_line_plot(data["time_s"], data["tracking_error_mm"],
                           "End-Effector Tracking Error", "Time (s)", "Error (mm)",
                           out / "simulation_tracking_error.png")
        if {"time_s", "cumulative_energy_J"}.issubset(data.columns):
            save_line_plot(data["time_s"], data["cumulative_energy_J"],
                           "Cumulative Mechanical Energy Estimate", "Time (s)", "Energy (J)",
                           out / "simulation_energy.png")
        if {"time_s", "mean_abs_jerk_rad_s3"}.issubset(data.columns):
            save_line_plot(data["time_s"], data["mean_abs_jerk_rad_s3"],
                           "Joint Motion Smoothness", "Time (s)", "Mean Absolute Jerk (rad/s³)",
                           out / "simulation_jerk.png")

        torque_columns = [column for column in data.columns if column.endswith("_torque_Nm")]
        if torque_columns and "time_s" in data:
            plt.figure(figsize=(9, 5))
            for column in torque_columns:
                plt.plot(data["time_s"], data[column], linewidth=1.3, label=column.replace("_torque_Nm", ""))
            plt.axhline(200, color="black", linestyle="--", alpha=0.5, label="+200 Nm reference")
            plt.axhline(-200, color="black", linestyle="--", alpha=0.5, label="-200 Nm reference")
            plt.title("Joint Torque Profiles")
            plt.xlabel("Time (s)")
            plt.ylabel("Torque (Nm)")
            plt.legend(ncol=2)
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            out.mkdir(parents=True, exist_ok=True)
            plt.savefig(out / "joint_torque_profiles.png", dpi=160, bbox_inches="tight")
            plt.close()
            print(f"Saved {out / 'joint_torque_profiles.png'}")
    else:
        print(f"Simulation CSV not found; skipping simulation plots: {simulation_path}")

    optuna_path = Path(args.optuna_csv)
    if optuna_path.exists():
        data = pd.read_csv(optuna_path)
        if {"number", "w1"}.issubset(data.columns):
            plt.figure(figsize=(8, 5))
            plt.scatter(data["number"], data["w1"], alpha=0.65, label="Trial w1")
            if "best_w1_so_far" in data.columns:
                plt.plot(data["number"], data["best_w1_so_far"], linestyle="--", label="Best so far")
            plt.title("Optuna Reward-Weight Search")
            plt.xlabel("Trial")
            plt.ylabel("w1")
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            out.mkdir(parents=True, exist_ok=True)
            plt.savefig(out / "optuna_reward_weights.png", dpi=160, bbox_inches="tight")
            plt.close()
            print(f"Saved {out / 'optuna_reward_weights.png'}")


if __name__ == "__main__":
    main()
