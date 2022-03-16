"""Visualize objects that will be used in datasets."""
import argparse
import numpy as np
from hand_embodiment.command_line import add_object_visualization_arguments
from hand_embodiment.vis_utils import Insole
from pytransform3d import visualizer as pv


def main():
    args = parse_args()

    fig = pv.figure()
    fig.plot_transform(np.eye(4), s=0.1)

    object_classes = []

    if args.insole:
        object_classes.append(Insole)

    if args.pillow:
        raise NotImplementedError()

    if args.electronic:
        raise NotImplementedError()

    if args.passport:
        raise NotImplementedError()

    if args.passport_closed:
        raise NotImplementedError()

    for ObjectClass in object_classes:
        markers = [p for p in ObjectClass.default_marker_positions.values()]
        fig.scatter(markers, s=0.006)

        pose = ObjectClass.pose_from_markers(
            **ObjectClass.default_marker_positions)
        fig.plot_transform(pose, s=0.1)

        artist = ObjectClass(**ObjectClass.default_marker_positions)
        artist.add_artist(fig)

        mesh = artist.load_mesh()
        fig.add_geometry(mesh)

    fig.show()


def parse_args():
    parser = argparse.ArgumentParser()
    add_object_visualization_arguments(parser)
    return parser.parse_args()


if __name__ == "__main__":
    main()
