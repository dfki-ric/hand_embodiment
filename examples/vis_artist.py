"""Visualize objects that will be used in datasets."""
import argparse
import numpy as np
from hand_embodiment.command_line import add_object_visualization_arguments
from hand_embodiment.vis_utils import (
    Insole, PillowSmall, Passport, ElectronicObject, ElectronicTarget,
    PassportClosed, PassportBox)
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
        object_classes.append(ElectronicObject)
        object_classes.append(ElectronicTarget)

    if args.passport:
        object_classes.append(Passport)

    if args.passport_closed:
        object_classes.append(PassportClosed)
        object_classes.append(PassportBox)

    fig = pv.figure()
    fig.plot_transform(np.eye(4), s=0.1)

    for ObjectClass in object_classes:
        # Marker positions in marker frame
        markers = [p for p in ObjectClass.default_marker_positions.values()]
        if not args.hide_marker_frame:
            fig.scatter(markers, s=MARKER_RADIUS)

        # Marker positions in mesh frame
        markers_inv = [
            pt.transform(ObjectClass.markers2mesh, pt.vector_to_point(p))[:3]
            for p in ObjectClass.default_marker_positions.values()]
        if not args.hide_mesh_frame:
            fig.scatter(markers_inv, s=MARKER_RADIUS, c=MARKER_COLORS)

        # Pose in marker frame
        pose = ObjectClass.pose_from_markers(
            **ObjectClass.default_marker_positions)
        if not args.hide_marker_frame:
            fig.plot_transform(pose, s=0.1)

        # Mesh in marker frame
        artist = ObjectClass()
        if not args.hide_marker_frame:
            artist.add_artist(fig)

        # Mesh in mesh frame
        mesh = artist.load_mesh()
        if not args.hide_mesh_frame:
            fig.add_geometry(mesh)

    fig.show()


def parse_args():
    parser = argparse.ArgumentParser()
    add_object_visualization_arguments(parser)
    parser.add_argument("--hide-mesh-frame", action="store_true",
                        help="Hide everything that is in mesh frame.")
    parser.add_argument("--hide-marker-frame", action="store_true",
                        help="Hide everything that is in marker frame.")
    return parser.parse_args()


if __name__ == "__main__":
    main()
