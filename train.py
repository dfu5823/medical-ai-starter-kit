#!/usr/bin/env python
"""Command-line entry point to launch a DermMNIST experiment.

To run this script, use:
    python train.py --config path/to/config.yaml

This file:
- Runs the entire pipeline for training and evaluating a model on DermMNIST.
- Parses a single `--config` argument pointing to a YAML config.
- Calls the pipeline-level `run_experiment` orchestrator with that config.
"""

import argparse

from src.pipeline.run import run_experiment


def parse_args():
    """Return parsed CLI arguments for the training script."""
    parser = argparse.ArgumentParser(description="DermMNIST workflow demo")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/baseline.yaml",
        help="Path to YAML config (default: configs/baseline.yaml).",
    )
    return parser.parse_args()


def main():
    """Load the config path from CLI and run the experiment."""
    args = parse_args()
    run_experiment(config_path=args.config)


if __name__ == "__main__":
    main()
