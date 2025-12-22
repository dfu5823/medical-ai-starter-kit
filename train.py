#!/usr/bin/env python
import argparse

from src.pipeline.run import run_experiment


def parse_args():
    parser = argparse.ArgumentParser(description="DermMNIST workflow demo")
    parser.add_argument("--config", type=str, default="configs/baseline.yaml",
                        help="Path to YAML config (default: configs/baseline.yaml).")
    return parser.parse_args()


def main():
    args = parse_args()
    run_experiment(config_path=args.config)


if __name__ == "__main__":
    main()
