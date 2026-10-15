"""Stable-Baselines3 callbacks that save episode metrics to CSV."""
from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback


class TrainingMetricsCallback(BaseCallback):
    """Record episode reward, tracking error, energy, jerk, and success."""

    def __init__(self, csv_path: str = "logs/training_log.csv", verbose: int = 0):
        super().__init__(verbose=verbose)
        self.csv_path = Path(csv_path)
        self.episode_rewards: list[float] = []
        self.episode_errors: list[float] = []
        self.episode_energy: list[float] = []
        self.episode_jerk: list[float] = []
        self.episode_successes: list[int] = []
        self._current_reward = 0.0
        self._errors: list[float] = []
        self._energy: list[float] = []
        self._jerk: list[float] = []
        self._success = False
        self._csv_file = None
        self._writer = None

    def _on_training_start(self) -> None:
        self.csv_path.parent.mkdir(parents=True, exist_ok=True)
        file_exists = self.csv_path.exists() and self.csv_path.stat().st_size > 0
        self._csv_file = self.csv_path.open("a", newline="", encoding="utf-8")
        self._writer = csv.DictWriter(
            self._csv_file,
            fieldnames=[
                "episode", "episode_reward", "mean_tracking_error_mm",
                "episode_energy_J", "mean_jerk_rad_s3", "success"
            ],
        )
        if not file_exists:
            self._writer.writeheader()

    def _on_step(self) -> bool:
        rewards = self.locals.get("rewards")
        infos = self.locals.get("infos")
        dones = self.locals.get("dones")
        if rewards is None or infos is None or dones is None:
            return True

        self._current_reward += float(rewards[0])
        info: dict[str, Any] = infos[0]
        if "tracking_error_mm" in info:
            self._errors.append(float(info["tracking_error_mm"]))
        if "energy_J" in info:
            self._energy.append(float(info["energy_J"]))
        if "jerk_rad_s3" in info:
            self._jerk.append(float(info["jerk_rad_s3"]))
        self._success = self._success or bool(info.get("success", False))

        if bool(dones[0]):
            self.episode_rewards.append(self._current_reward)
            mean_error = float(np.mean(self._errors)) if self._errors else float("nan")
            energy_total = float(np.sum(self._energy)) if self._energy else float("nan")
            mean_jerk = float(np.mean(self._jerk)) if self._jerk else float("nan")
            self.episode_errors.append(mean_error)
            self.episode_energy.append(energy_total)
            self.episode_jerk.append(mean_jerk)
            self.episode_successes.append(int(self._success))

            if self._writer:
                self._writer.writerow({
                    "episode": len(self.episode_rewards),
                    "episode_reward": self._current_reward,
                    "mean_tracking_error_mm": mean_error,
                    "episode_energy_J": energy_total,
                    "mean_jerk_rad_s3": mean_jerk,
                    "success": int(self._success),
                })
                self._csv_file.flush()

            if self.verbose and len(self.episode_rewards) % 50 == 0:
                print(
                    f"Episode {len(self.episode_rewards)} | "
                    f"reward={np.mean(self.episode_rewards[-50:]):.2f} | "
                    f"tracking_error={np.nanmean(self.episode_errors[-50:]):.1f} mm"
                )
            self._current_reward = 0.0
            self._errors.clear()
            self._energy.clear()
            self._jerk.clear()
            self._success = False
        return True

    def _on_training_end(self) -> None:
        if self._csv_file:
            self._csv_file.close()
            self._csv_file = None
            self._writer = None
