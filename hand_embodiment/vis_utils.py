import numpy as np
import open3d as o3d
from pytransform3d import visualizer as pv


def make_coordinate_system(s):
    coordinate_system = o3d.geometry.LineSet()
    points = []
    lines = []
    colors = []
    for d in range(3):
        color = [0, 0, 0]
        color[d] = 1

        start = [0, 0, 0]
        start[d] = -s
        end = [0, 0, 0]
        end[d] = s

        points.extend([start, end])
        lines.append([len(points) - 2, len(points) - 1])
        colors.append(color)
        for i, step in enumerate(np.arange(-s, s + 0.01, 0.01)):
            length = 0.05 if i % 5 == 0 else 0.01
            start = [0, 0, 0]
            start[d] = step
            start[(d + 2) % 3] = -length
            end = [0, 0, 0]
            end[d] = step
            end[(d + 2) % 3] = length
            points.extend([start, end])
            lines.append([len(points) - 2, len(points) - 1])
            colors.append(color)
        coordinate_system.points = o3d.utility.Vector3dVector(points)
        coordinate_system.lines = o3d.utility.Vector2iVector(lines)
        coordinate_system.colors = o3d.utility.Vector3dVector(colors)
    return coordinate_system


class ManoHand(pv.Artist):
    """Representation of hand mesh as artist for 3D visualization in Open3D."""
    def __init__(self, mbrm, show_mesh=True, show_vertices=False):
        self.mbrm = mbrm
        self.show_mesh = show_mesh
        self.show_vertices = show_vertices

    def set_data(self):
        pass

    @property
    def geometries(self):
        """Expose geometries.

        Returns
        -------
        geometries : list
            List of geometries that can be added to the visualizer.
        """
        geoms = []
        if self.show_mesh:
            geoms.append(self.mbrm.hand_state_.hand_mesh)
        if self.show_vertices:
            geoms.append(self.mbrm.hand_state_.hand_pointcloud)
        return geoms
