"""Visualize objects that will be used in datasets."""
import argparse
import numpy as np
import pytransform3d.visualizer as pv
import pytransform3d.transformations as pt
from hand_embodiment.command_line import add_artist_argument
from hand_embodiment.vis_utils import ARTISTS


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

    fig = pv.figure()
    fig.plot_transform(np.eye(4), s=0.1)

    if args.artist is not None:
        ObjectClass = ARTISTS[args.artist]

        # Marker positions in marker frame
        markers = [p for p in ObjectClass.default_marker_positions.values()]
        if args.show_marker_frame:
            fig.scatter(markers, s=MARKER_RADIUS)

        # Marker positions in mesh frame
        markers_inv = [
            pt.transform(ObjectClass.markers2mesh, pt.vector_to_point(p))[:3]
            for p in ObjectClass.default_marker_positions.values()]
        if args.show_mesh_frame:
            fig.scatter(markers_inv, s=MARKER_RADIUS, c=MARKER_COLORS)

        # Pose in marker frame
        pose = ObjectClass.pose_from_markers(
            **ObjectClass.default_marker_positions)
        if args.show_marker_frame:
            fig.plot_transform(pose, s=0.1)

        # Mesh in marker frame
        artist = ObjectClass(show_frame=False)
        if args.show_marker_frame:
            artist.add_artist(fig)

        # Mesh in mesh frame
        mesh = artist.load_mesh()
        if args.show_mesh_frame:
            fig.add_geometry(mesh)

    fig.show()


def parse_args():
    parser = argparse.ArgumentParser()
    add_artist_argument(parser)
    parser.add_argument("--show-mesh-frame", action="store_true",
                        help="Show everything that is in mesh frame.")
    parser.add_argument("--show-marker-frame", action="store_true",
                        help="Show everything that is in marker frame.")
    return parser.parse_args()


if __name__ == "__main__":
    main()
