"""Visualize embodied trajectories."""
import argparse
import numpy as np
import pandas as pd
import pytransform3d.visualizer as pv
import tqdm
from hand_embodiment.vis_utils import ARTISTS
from hand_embodiment.command_line import add_artist_argument


COLUMNS = ["base_x", "base_y", "base_z",
           "base_qw", "base_qx", "base_qy", "base_qz"]


def main():
    args = parse_args()

    fig = pv.figure()

    fig.plot_transform(np.eye(4), s=0.1)

    for filename in tqdm.tqdm(args.filenames):
        P = pd.read_csv(filename)[COLUMNS].to_numpy()
        fig.plot_trajectory(P, n_frames=2, s=0.03)

    if args.artist is not None:
        ARTISTS[args.artist](show_frame=False).add_artist(fig)

    fig.view_init()

    fig.show()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "filenames", type=str, nargs="+",
        help="Files that contain embodied trajectories.")
    add_artist_argument(parser)
    return parser.parse_args()


if __name__ == "__main__":
    main()
