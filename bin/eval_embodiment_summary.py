"""Summarize results of evaluation of embodiment mapping."""
import argparse
import glob
import json
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "pattern", type=str, help="Pattern to match result files.")
    return parser.parse_args()


def summary(args):
    filenames = list(glob.glob(args.pattern))
    metrics = {}

    for filename in filenames:
        with open(filename, "r") as f:
            result = json.load(f)
            for finger in result:
                if finger not in metrics:
                    metrics[finger] = []
                metrics[finger].append(result[finger])

    print(f"{len(filenames)} samples")
    for finger in metrics:
        print(f"{finger}:\t{np.mean(metrics[finger]):.4f} in "
              f"[{min(metrics[finger]):.4f}, {max(metrics[finger]):.4f}]")


if __name__ == "__main__":
    args = parse_args()
    summary(args)
