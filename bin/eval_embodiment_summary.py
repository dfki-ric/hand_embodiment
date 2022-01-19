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
    fingers = ["thumb", "index", "middle", "ring", "little"]
    metrics = {f: [] for f in fingers}

    for filename in filenames:
        with open(filename, "r") as f:
            result = json.load(f)
            for finger in fingers:
                metrics[finger].append(result[finger])

    print(f"{len(filenames)} samples")
    for finger in fingers:
        print(f"{finger}:\t{np.mean(metrics[finger]):.4f} in "
              f"[{min(metrics[finger]):.4f}, {max(metrics[finger]):.4f}]")


if __name__ == "__main__":
    args = parse_args()
    summary(args)
