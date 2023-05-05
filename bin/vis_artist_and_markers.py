"""Visualize markers and artist based on markers.

Example call:

python bin/vis_artist_and_markers.py ../motion_capture_grasp_poses/raw_data/insole_flipped/AprilInsoleModelCalibration.tsv --artist insole-flipped --markers edge_1 edge_2 edge_3 edge_4 edge_5 edge_6 edge_7
"""
import argparse
import numpy as np
import pytransform3d.visualizer as pv
from hand_embodiment.mocap_dataset import read_qualisys_tsv, match_columns
from hand_embodiment.command_line import add_artist_argument
from hand_embodiment.vis_utils import ARTISTS


def main():
    args = parse_args()

    artist = ARTISTS[args.artist]()

    trajectory = read_qualisys_tsv(filename=args.filename)
    columns = {}
    for marker_name in args.markers + list(artist.marker_names):
        column_names = match_columns(
            trajectory, [marker_name], keep_time=False)
        columns[marker_name] = list(sorted(column_names))

    fig = pv.figure()

    fig.plot_transform(np.eye(4), s=0.3)
    marker_spheres = fig.scatter(np.zeros((len(columns), 3)), s=0.006,
                                 c=(0.5, 0.5, 0.5))

    artist.add_artist(fig)

    fig.view_init()

    animation_callback = AnimationCallback(
        trajectory, args.markers, columns)
    fig.animate(animation_callback, n_frames=len(trajectory), loop=True,
                fargs=(marker_spheres, artist))
    fig.show()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "filename", type=str,
        help="File that contains marker positions.")
    add_artist_argument(parser)
    parser.add_argument(
        "--markers", type=str, nargs="*",
        help="Names of markers to display.")
    return parser.parse_args()


class AnimationCallback:
    def __init__(self, trajectory, markers, columns):
        self.trajectory = trajectory
        self.markers = markers
        self.columns = columns

    def __call__(self, step, marker_spheres, artist):
        t = step % len(self.trajectory)

        P = np.zeros((len(self.columns), 3))
        for marker_idx, marker in enumerate(self.columns):
            P[marker_idx] = self.trajectory[
                self.columns[marker]].iloc[t].to_numpy()
        marker_spheres.set_data(P)

        kwargs = {}
        for marker in artist.marker_names:
            kwargs[marker] = self.trajectory[
                self.columns[marker]].iloc[t].to_numpy()
        artist.set_data(**kwargs)

        return marker_spheres


if __name__ == "__main__":
    main()
