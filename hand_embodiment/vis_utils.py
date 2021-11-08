import time

from pkg_resources import resource_filename
import numpy as np
import open3d as o3d
import pytransform3d.rotations as pr
import pytransform3d.transformations as pt
import pytransform3d.visualizer as pv


def make_coordinate_system(s, short_tick_length=0.01, long_tick_length=0.05):
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
            tick_length = long_tick_length if i % 5 == 0 else short_tick_length
            start = [0, 0, 0]
            start[d] = step
            start[(d + 2) % 3] = -tick_length
            end = [0, 0, 0]
            end[d] = step
            end[(d + 2) % 3] = tick_length
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
        self.mesh.paint_uniform_color(np.array([0.37, 0.28, 0.26]))
        self.mesh.compute_triangle_normals()
        self.insole_back = np.zeros(3)
        self.insole_front = np.array([1, 0, 0])
        self.insole_mesh2insole_markers = pt.transform_from(
            R=pr.active_matrix_from_extrinsic_roll_pitch_yaw(np.deg2rad([180, 0, -4.5])),
            p=np.array([0.04, 0.07, -0.007]))
        self.insole_markers2origin = np.copy(self.insole_mesh2insole_markers)
        self.set_data(insole_back, insole_front)

    def set_data(self, insole_back, insole_front):
        if not any(np.isnan(insole_back)):
            self.insole_back = insole_back
        if not any(np.isnan(insole_front)):
            self.insole_front = insole_front

        self.mesh.transform(pt.invert_transform(pt.concat(self.insole_mesh2insole_markers, self.insole_markers2origin)))

        self.insole_markers2origin = insole_pose(self.insole_back, self.insole_front)

        self.mesh.transform(pt.concat(self.insole_mesh2insole_markers, self.insole_markers2origin))

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


class PillowSmall(pv.Artist):
    """Representation of small pillow mesh."""
    def __init__(
            self, pillow_left=np.array([0, 1, 0]),
            pillow_right=np.array([0, 0, 0]), pillow_top=np.array([1, 0, 0])):
        filename = resource_filename(
            "hand_embodiment", "model/objects/pillow_small.stl")
        self.mesh = o3d.io.read_triangle_mesh(filename)
        self.mesh.compute_triangle_normals()
        self.pillow_left = pillow_left
        self.pillow_right = pillow_right
        self.pillow_top = pillow_top
        self.pillow_mesh2pillow_markers = pt.transform_from(
            R=pr.active_matrix_from_extrinsic_roll_pitch_yaw(np.deg2rad([0, 0, 0])),
            p=np.array([0.195, 0.195, 0.115]))
        self.pillow_markers2origin = np.copy(self.pillow_mesh2pillow_markers)
        self.set_data(pillow_left, pillow_right, pillow_top)

    def set_data(self, pillow_left, pillow_right, pillow_top):
        if not any(np.isnan(pillow_left)):
            self.pillow_left = pillow_left
        if not any(np.isnan(pillow_right)):
            self.pillow_right = pillow_right
        if not any(np.isnan(pillow_top)):
            self.pillow_top = pillow_top

        self.mesh.transform(pt.invert_transform(pt.concat(self.pillow_mesh2pillow_markers, self.pillow_markers2origin)))

        self.pillow_markers2origin = pillow_pose(self.pillow_left, self.pillow_right, self.pillow_top)

        self.mesh.transform(pt.concat(self.pillow_mesh2pillow_markers, self.pillow_markers2origin))

    @property
    def geometries(self):
        """Expose geometries.

        Returns
        -------
        geometries : list
            List of geometries that can be added to the visualizer.
        """
        return [self.mesh]


def pillow_pose(pillow_left, pillow_right, pillow_top):
    """Compute pose of pillow."""
    right2top = pillow_top - pillow_right
    right2left = pillow_left - pillow_right
    pose = np.eye(4)
    pose[:3, :3] = pr.matrix_from_two_vectors(right2top, right2left)
    pillow_middle = 0.5 * (pillow_left + pillow_right) + 0.5 * right2top
    pose[:3, 3] = pillow_middle
    return pose


class Electronic(pv.Artist):
    """Representation of electronic target component."""
    def __init__(
            self, target_top=np.zeros(3), target_bottom=np.array([1, 0, 0]),
            object_left=np.zeros(3), object_right=np.array([0, 1, 0]),
            object_top=np.array([1, 0, 0])):
        target_filename = resource_filename(
            "hand_embodiment", "model/objects/electronic_target.stl")
        self.target_mesh = o3d.io.read_triangle_mesh(target_filename)
        self.target_mesh.paint_uniform_color(np.array([0.21, 0.20, 0.46]))
        self.target_mesh.compute_triangle_normals()

        self.target_top = np.zeros(3)
        self.target_bottom = np.array([1, 0, 0])

        self.electronic_target2target_markers = pt.transform_from(
            R=pr.active_matrix_from_extrinsic_roll_pitch_yaw(np.deg2rad([0, 0, 0])),
            p=-np.array([0.0625 + 0.014, -0.057 + 0.006, 0.0]) / 2.0)
        self.target_markers2origin = np.copy(self.electronic_target2target_markers)

        target_filename = resource_filename(
            "hand_embodiment", "model/objects/electronic_object.stl")
        self.object_mesh = o3d.io.read_triangle_mesh(target_filename)
        self.object_mesh.paint_uniform_color(np.array([0.68, 0.45, 0.23]))
        self.object_mesh.compute_triangle_normals()

        self.object_left = np.zeros(3)
        self.object_right = np.array([0, 1, 0])
        self.object_top = np.array([1, 0, 0])

        self.electronic_object2object_markers = pt.transform_from(
            R=pr.active_matrix_from_extrinsic_roll_pitch_yaw(np.deg2rad([0, 0, 0])),
            p=np.array([0.0, 0.0, -0.01]))
        self.object_markers2origin = np.copy(self.electronic_object2object_markers)

        self.set_data(
            target_top, target_bottom, object_left, object_right, object_top)

    def set_data(self, target_top, target_bottom, object_left, object_right,
                 object_top):
        if not any(np.isnan(target_top)):
            self.target_top = target_top
        if not any(np.isnan(target_bottom)):
            self.target_bottom = target_bottom
        if not any(np.isnan(object_left)):
            self.object_left = object_left
        if not any(np.isnan(object_right)):
            self.object_right = object_right
        if not any(np.isnan(object_top)):
            self.object_top = object_top

        self.target_mesh.transform(pt.invert_transform(pt.concat(
            self.electronic_target2target_markers, self.target_markers2origin)))
        self.target_markers2origin = electronic_target_pose(
            self.target_top, self.target_bottom)
        self.target_mesh.transform(pt.concat(
            self.electronic_target2target_markers, self.target_markers2origin))

        self.object_mesh.transform(pt.invert_transform(pt.concat(
            self.electronic_object2object_markers, self.object_markers2origin)))
        self.object_markers2origin = electronic_object_pose(
            self.object_left, self.object_right, self.object_top)
        self.object_mesh.transform(pt.concat(
            self.electronic_object2object_markers, self.object_markers2origin))

    @property
    def geometries(self):
        """Expose geometries.

        Returns
        -------
        geometries : list
            List of geometries that can be added to the visualizer.
        """
        return [self.target_mesh, self.object_mesh]


def electronic_target_pose(target_top, target_bottom):
    """Compute pose of electronic target."""
    x_axis = pr.norm_vector(target_top - target_bottom)
    z_axis = np.copy(pr.unitz)
    y_axis = pr.norm_vector(pr.perpendicular_to_vectors(z_axis, x_axis))
    z_axis = pr.norm_vector(pr.perpendicular_to_vectors(x_axis, y_axis))
    R = np.column_stack((x_axis, y_axis, z_axis))
    return pt.transform_from(R=R, p=target_bottom)


def electronic_object_pose(object_left, object_right, object_top):
    """Compute pose of electronic object."""
    left2top = object_top - object_left
    left2right = object_left - object_right
    pose = np.eye(4)
    pose[:3, :3] = pr.matrix_from_two_vectors(left2right, left2top)
    object_middle = 0.5 * (object_left + object_right) + 0.5 * left2top
    pose[:3, 3] = object_middle
    return pose


class AnimationCallback:
    def __init__(self, fig, pipeline, args, show_robot=False):
        self.fig = fig
        self.args = args
        self.show_robot = show_robot

        self.show_mano = (hasattr(args, "hide_mano") and not args.hide_mano
                          or hasattr(args, "show_mano") and args.show_mano)
        if self.show_mano:
            self.hand = pipeline.make_hand_artist()
            self.hand.add_artist(self.fig)

        if self.args.insole:
            self.object_mesh = Insole()
            self.object_mesh.add_artist(self.fig)
            self.object_frame = pv.Frame(np.eye(4), s=0.1)
            self.object_frame.add_artist(self.fig)

        if self.args.pillow:
            self.object_mesh = PillowSmall()
            self.object_mesh.add_artist(self.fig)
            self.object_frame = pv.Frame(np.eye(4), s=0.1)
            self.object_frame.add_artist(self.fig)

        if self.args.electronic:
            self.object_mesh = Electronic()
            self.object_mesh.add_artist(self.fig)
            self.object_frame = pv.Frame(np.eye(4), s=0.1)
            self.object_frame.add_artist(self.fig)

        if show_robot:
            self.robot = pipeline.make_robot_artist()
            self.robot.add_artist(self.fig)

    def __call__(self, t, markers, dataset, pipeline):
        if t == 1:
            pipeline.reset()
            time.sleep(self.args.delay)

        markers.set_data(dataset.get_markers(t))

        artists = [markers]

        if self.args.insole:
            marker_names = dataset.config.get("additional_markers", ())
            additional_markers = dataset.get_additional_markers(t)
            insole_back = additional_markers[marker_names.index("insole_back")]
            insole_front = additional_markers[marker_names.index("insole_front")]
            self.object_mesh.set_data(insole_back, insole_front)
            artists.append(self.object_mesh)
            self.object_frame.set_data(insole_pose(insole_back, insole_front))
            artists.append(self.object_frame)

        if self.args.pillow:
            marker_names = dataset.config.get("additional_markers", ())
            additional_markers = dataset.get_additional_markers(t)
            pillow_left = additional_markers[marker_names.index("pillow_left")]
            pillow_right = additional_markers[marker_names.index("pillow_right")]
            pillow_top = additional_markers[marker_names.index("pillow_top")]
            self.object_mesh.set_data(pillow_left, pillow_right, pillow_top)
            artists.append(self.object_mesh)
            self.object_frame.set_data(pillow_pose(
                pillow_left, pillow_right, pillow_top))
            artists.append(self.object_frame)

        if self.args.electronic:
            marker_names = dataset.config.get("additional_markers", ())
            additional_markers = dataset.get_additional_markers(t)
            target_top = additional_markers[marker_names.index("target_top")]
            target_bottom = additional_markers[marker_names.index("target_bottom")]
            object_left = additional_markers[marker_names.index("object_left")]
            object_right = additional_markers[marker_names.index("object_right")]
            object_top = additional_markers[marker_names.index("object_top")]
            self.object_mesh.set_data(
                target_top, target_bottom, object_left, object_right,
                object_top)
            artists.append(self.object_mesh)
            if not any(np.isnan(np.hstack((target_bottom, target_top)))):
                self.object_frame.set_data(
                    electronic_target_pose(target_top, target_bottom))
                artists.append(self.object_frame)

        if self.show_mano or self.show_robot:
            pipeline.estimate_hand(
                dataset.get_hand_markers(t), dataset.get_finger_markers(t))

        if self.show_mano:
            self.hand.set_data()
            artists.append(self.hand)

        if self.show_robot:
            pipeline.estimate_robot()
            self.robot.set_data()
            artists.append(self.robot)

        return artists
