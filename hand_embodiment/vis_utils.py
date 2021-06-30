from pkg_resources import resource_filename
import numpy as np
import open3d as o3d
import pytransform3d.rotations as pr
import pytransform3d.transformations as pt
import pytransform3d.visualizer as pv


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


class Insole(pv.Artist):
    """Representation of insole mesh."""
    def __init__(self, insole_back=np.zeros(3), insole_front=np.array([1, 0, 0])):
        filename = resource_filename(
            "hand_embodiment", "model/objects/insole.stl")
        self.mesh = o3d.io.read_triangle_mesh(filename)
        self.insole_back = np.zeros(3)
        self.insole_front = np.array([1, 0, 0])
        self.insole_mesh2insole = pt.transform_from(
            R=pr.active_matrix_from_extrinsic_roll_pitch_yaw(np.deg2rad([180, 0, -7])),
            p=np.array([0.03, 0, 0]))
        self.insole2origin = np.copy(self.insole_mesh2insole)
        self.set_data(insole_back, insole_front)

    def set_data(self, insole_back, insole_front):
        if not any(np.isnan(insole_back)):
            self.insole_back = insole_back
        if not any(np.isnan(insole_front)):
            self.insole_front = insole_front

        self.mesh.transform(pt.invert_transform(pt.concat(self.insole_mesh2insole, self.insole2origin)))

        self.insole2origin = insole_pose(insole_back, insole_front)

        self.mesh.transform(pt.concat(self.insole_mesh2insole, self.insole2origin))

    @property
    def geometries(self):
        """Expose geometries.

        Returns
        -------
        geometries : list
            List of geometries that can be added to the visualizer.
        """
        return [self.mesh]


def insole_pose(insole_back, insole_front):
    """Compute pose of insole."""
    x_axis = pr.norm_vector(insole_front - insole_back)
    z_axis = np.copy(pr.unitz)
    y_axis = pr.norm_vector(pr.perpendicular_to_vectors(z_axis, x_axis))
    z_axis = pr.norm_vector(pr.perpendicular_to_vectors(x_axis, y_axis))
    R = np.column_stack((x_axis, y_axis, z_axis))
    return pt.transform_from(R=R, p=insole_back)
