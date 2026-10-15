"""Continue training an existing Stable-Baselines3 SAC checkpoint.

Example:
    python -m src.training.continue_training --model models/pg_sac_final.zip --timesteps 100000
"""
from __future__ import annotations

import argparse
from pathlib import Path

from stable_baselines3 import SAC

from src.env.arm_env import ArmEnv
from src.training.callbacks import TrainingMetricsCallback


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="Existing SAC model .zip path")
    parser.add_argument("--timesteps", type=int, default=100_000)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--model-out", default="models/pg_sac_continued")
    parser.add_argument("--log-csv", default="logs/continued_training_log.csv")
    parser.add_argument("--verbose", type=int, default=1)
    args = parser.parse_args()

    model_path = Path(args.model)
    if not model_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {model_path}")
    if args.timesteps < 1:
        raise ValueError("--timesteps must be positive")

    env = ArmEnv()
    try:
        model = SAC.load(str(model_path), env=env, device=args.device)
        callback = TrainingMetricsCallback(args.log_csv, verbose=args.verbose)
        model.learn(
            total_timesteps=args.timesteps,
            callback=callback,
            reset_num_timesteps=False,
            progress_bar=False,
        )
        output = Path(args.model_out)
        output.parent.mkdir(parents=True, exist_ok=True)
        model.save(str(output))
        print(f"Continued-training model saved to: {output}.zip")
        print(f"Metrics appended to: {args.log_csv}")
    finally:
        env.close()


if __name__ == "__main__":
    main()
