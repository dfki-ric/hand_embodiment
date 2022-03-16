"""Visualize objects that will be used in datasets."""
import argparse
import numpy as np
from hand_embodiment.command_line import add_object_visualization_arguments
from hand_embodiment.vis_utils import Insole, PillowSmall
import pytransform3d.visualizer as pv
import pytransform3d.transformations as pt


MARKER_RADIUS = 0.006
MARKER_COLORS = (
    (1, 0, 0),
    (0, 1, 0),
    (0, 0, 1),
    (1, 1, 0),
    (0, 1, 1)
)


def main():
    args = parse_args()

    object_classes = []

    if args.insole:
        object_classes.append(Insole)

    if args.pillow:
        object_classes.append(PillowSmall)

    if args.electronic:
        raise NotImplementedError()

    if args.passport:
        raise NotImplementedError()

    if args.passport_closed:
        raise NotImplementedError()

    fig = pv.figure()
    fig.plot_transform(np.eye(4), s=0.1)

    for ObjectClass in object_classes:
        # Marker positions in marker frame
        markers = [p for p in ObjectClass.default_marker_positions.values()]
        fig.scatter(markers, s=MARKER_RADIUS)

        # Marker positions in mesh frame
        markers_inv = [
            pt.transform(ObjectClass.markers2mesh, pt.vector_to_point(p))[:3]
            for p in ObjectClass.default_marker_positions.values()]
        fig.scatter(markers_inv, s=MARKER_RADIUS, c=MARKER_COLORS)

        # Pose in marker frame
        pose = ObjectClass.pose_from_markers(
            **ObjectClass.default_marker_positions)
        fig.plot_transform(pose, s=0.1)

        # Mesh in marker frame
        artist = ObjectClass(**ObjectClass.default_marker_positions)
        artist.add_artist(fig)

        # Mesh in mesh frame
        mesh = artist.load_mesh()
        fig.add_geometry(mesh)

    fig.show()


def parse_args():
    parser = argparse.ArgumentParser()
    add_object_visualization_arguments(parser)
    return parser.parse_args()


if __name__ == "__main__":
    main()
