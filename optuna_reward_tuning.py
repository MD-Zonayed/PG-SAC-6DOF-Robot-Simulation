"""Tune reward weights by training and evaluating an SAC policy per Optuna trial.

This is an executable, compute-intensive starting point based on the supplied Optuna
snippet. Each trial trains a fresh SAC model with candidate weights and evaluates it
using a common fixed score. Trial values and best-so-far weights are saved as CSV.

Example (run from repository root):
    python -m src.optimization.optuna_reward_tuning --trials 50 --train-timesteps 10000

For a publication-grade rerun, match the manuscript's exact environment, reward
scaling, random seeds, trial count, and stopping criterion before interpreting results.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import optuna
from stable_baselines3 import SAC

from src.env.arm_env import ArmEnv


def evaluate(model: SAC, episodes: int, seed_base: int) -> tuple[float, float, float, float]:
    env = ArmEnv()
    errors: list[float] = []
    energies: list[float] = []
    jerks: list[float] = []
    successes: list[float] = []
    try:
        for episode in range(episodes):
            state, _ = env.reset(seed=seed_base + episode)
            terminated = truncated = False
            while not (terminated or truncated):
                action, _ = model.predict(state, deterministic=True)
                state, _, terminated, truncated, info = env.step(action)
                errors.append(float(info["tracking_error_mm"]))
                energies.append(float(info["energy_J"]))
                jerks.append(float(info["jerk_rad_s3"]))
                successes.append(float(info["success"]))
    finally:
        env.close()
    return (
        float(np.mean(errors)) if errors else float("inf"),
        float(np.sum(energies)) if energies else float("inf"),
        float(np.mean(jerks)) if jerks else float("inf"),
        float(np.mean(successes)) if successes else 0.0,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trials", type=int, default=50)
    parser.add_argument("--train-timesteps", type=int, default=10_000)
    parser.add_argument("--eval-episodes", type=int, default=3)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", default="results/optuna")
    args = parser.parse_args()
    if args.trials < 1 or args.train_timesteps < 1 or args.eval_episodes < 1:
        raise ValueError("trials, train-timesteps, and eval-episodes must be positive")

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    def objective(trial: optuna.Trial) -> float:
        w1 = trial.suggest_float("w1", 0.40, 0.90)
        # Keep w1 + w2 <= 0.95 so w3 remains >= 0.05.
        w2_upper = min(0.40, 0.95 - w1)
        if w2_upper < 0.05:
            raise optuna.TrialPruned("No valid range for w2")
        w2 = trial.suggest_float("w2", 0.05, w2_upper)
        w3 = 1.0 - w1 - w2
        if w3 < 0.05:
            raise optuna.TrialPruned("w3 must be at least 0.05")

        env = ArmEnv(reward_weights=(w1, w2, w3))
        try:
            model = SAC(
                "MlpPolicy", env,
                learning_rate=3e-4,
                buffer_size=100_000,
                batch_size=256,
                gamma=0.99,
                tau=0.005,
                ent_coef=0.2,
                seed=args.seed + trial.number,
                verbose=0,
                device=args.device,
            )
            model.learn(total_timesteps=args.train_timesteps, progress_bar=False)
        finally:
            env.close()

        mean_error_mm, total_energy_j, mean_jerk, success_rate = evaluate(
            model, args.eval_episodes, args.seed + 100_000 + trial.number * args.eval_episodes
        )
        # Common evaluation score, independent of the trial's training reward weights.
        score = (mean_error_mm / 1000.0) + (total_energy_j / 100.0) + (mean_jerk / 1_000_000.0) - (0.1 * success_rate)
        trial.set_user_attr("mean_tracking_error_mm", mean_error_mm)
        trial.set_user_attr("total_energy_J", total_energy_j)
        trial.set_user_attr("mean_jerk_rad_s3", mean_jerk)
        trial.set_user_attr("success_rate", success_rate)
        return float(score)

    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=args.seed))
    print(f"Starting Optuna search: {args.trials} trials; {args.train_timesteps} training timesteps/trial")
    study.optimize(objective, n_trials=args.trials, show_progress_bar=True)

    rows = []
    best_so_far = float("inf")
    best_w1_so_far = float("nan")
    for trial in study.trials:
        if trial.value is not None and trial.value < best_so_far:
            best_so_far = float(trial.value)
            best_w1_so_far = float(trial.params.get("w1", float("nan")))
        row = {
            "number": trial.number,
            "value": trial.value,
            "w1": trial.params.get("w1"),
            "w2": trial.params.get("w2"),
            "w3": 1.0 - trial.params["w1"] - trial.params["w2"] if "w1" in trial.params and "w2" in trial.params else None,
            "best_w1_so_far": best_w1_so_far,
            "mean_tracking_error_mm": trial.user_attrs.get("mean_tracking_error_mm"),
            "total_energy_J": trial.user_attrs.get("total_energy_J"),
            "mean_jerk_rad_s3": trial.user_attrs.get("mean_jerk_rad_s3"),
            "success_rate": trial.user_attrs.get("success_rate"),
            "state": trial.state.name,
        }
        rows.append(row)

    csv_path = output_dir / "optuna_trials.csv"
    fieldnames = list(rows[0].keys()) if rows else ["number", "value", "w1", "w2", "w3"]
    with csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    if study.best_trial:
        best = study.best_trial
        w1 = best.params["w1"]
        w2 = best.params["w2"]
        w3 = 1.0 - w1 - w2
        print("Best weights from this run:")
        print(f"w1 (tracking) = {w1:.3f}")
        print(f"w2 (energy)   = {w2:.3f}")
        print(f"w3 (jerk)     = {w3:.3f}")
        print(f"Objective     = {best.value:.5f}")
    print(f"Saved trial records to {csv_path}")
    print("Note: this training-per-trial setup is computationally expensive and should be aligned with the manuscript before reporting results.")


if __name__ == "__main__":
    main()
