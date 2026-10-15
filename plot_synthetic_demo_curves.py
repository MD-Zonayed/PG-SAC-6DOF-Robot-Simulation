"""Generate illustrative synthetic curves similar to the supplied plotting snippets.

IMPORTANT: These plots are generated from random/synthetic values. They are for UI
or plotting tests only and must not be presented as measured research results.
Use plot_training_results.py for actual CSV-derived figures.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="results/demo_figures")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    time = np.linspace(0, 4, 1000)
    pgsac_power = np.abs(rng.normal(4.0, 0.5, 1000))
    pgsac_power[:100] += np.linspace(8, 0, 100)
    pid_power = np.abs(rng.normal(6.5, 1.0, 1000))
    pid_power[:100] += np.linspace(15, 0, 100)
    pgsac_energy = np.cumsum(pgsac_power) * (time[1] - time[0])
    pid_energy = np.cumsum(pid_power) * (time[1] - time[0])

    plt.figure(figsize=(8, 5))
    plt.plot(time, pgsac_energy, linewidth=2, label="Illustrative PG-SAC")
    plt.plot(time, pid_energy, linestyle="--", linewidth=2, label="Illustrative PID")
    plt.title("DEMO ONLY: Synthetic Cumulative Energy Curves")
    plt.xlabel("Time (s)")
    plt.ylabel("Illustrative Energy (arbitrary estimate, J-like scale)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output / "DEMO_synthetic_energy_comparison.png", dpi=150, bbox_inches="tight")
    plt.close()

    jerk = np.abs(rng.normal(18000, 2000, 500))
    jerk[:20] *= np.linspace(1.2, 1, 20)
    jerk = np.convolve(jerk, np.ones(15) / 15, mode="same")
    plt.figure(figsize=(8, 5))
    plt.plot(np.linspace(0, 4, len(jerk)), jerk, linewidth=2)
    plt.title("DEMO ONLY: Synthetic Jerk Profile")
    plt.xlabel("Time (s)")
    plt.ylabel("Synthetic Jerk (rad/s³)")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output / "DEMO_synthetic_jerk_profile.png", dpi=150, bbox_inches="tight")
    plt.close()

    print(f"Synthetic demo plots saved to {output.resolve()}")
    print("Reminder: these plots are random illustrations, not experimental results.")


if __name__ == "__main__":
    main()
