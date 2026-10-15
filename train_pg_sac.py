"""Train a SAC policy on the PyBullet 6-actuated-joint ArmEnv.

Usage from the repository root:
    python -m src.training.train_pg_sac --timesteps 100000 --device auto
"""
from __future__ import annotations

import argparse
from pathlib import Path

from stable_baselines3 import SAC
from stable_baselines3.common.env_checker import check_env

from src.env.arm_env import ArmEnv
from src.training.callbacks import TrainingMetricsCallback


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--timesteps", type=int, default=100_000)
    parser.add_argument("--device", default="auto", help="auto, cpu, or cuda")
    parser.add_argument("--model-out", default="models/pg_sac_final")
    parser.add_argument("--log-csv", default="logs/training_log.csv")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--verbose", type=int, default=1)
    parser.add_argument("--check-env", action="store_true", help="Run Gymnasium env checker first")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.timesteps < 1:
        raise ValueError("--timesteps must be positive")

    env = ArmEnv()
    try:
        if args.check_env:
            check_env(env, warn=True)
        callback = TrainingMetricsCallback(args.log_csv, verbose=args.verbose)
        model = SAC(
            "MlpPolicy",
            env,
            learning_rate=3e-4,
            buffer_size=500_000,
            batch_size=256,
            gamma=0.99,
            tau=0.005,
            ent_coef=0.2,
            seed=args.seed,
            verbose=args.verbose,
            device=args.device,
        )
        print(f"Training SAC for {args.timesteps:,} timesteps...")
        model.learn(total_timesteps=args.timesteps, callback=callback, progress_bar=False)
        output = Path(args.model_out)
        output.parent.mkdir(parents=True, exist_ok=True)
        model.save(str(output))
        print(f"Saved model to: {output}.zip")
        print(f"Training metrics CSV: {args.log_csv}")
        if callback.episode_errors:
            print(f"Final 50-episode mean error: {sum(callback.episode_errors[-50:]) / len(callback.episode_errors[-50:]):.1f} mm")
    finally:
        env.close()


if __name__ == "__main__":
    main()
