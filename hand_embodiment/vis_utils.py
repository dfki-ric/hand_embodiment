"""Visualization utilities."""
import time

from pkg_resources import resource_filename
import numpy as np
import pytransform3d.rotations as pr
import pytransform3d.transformations as pt
import pytransform3d.visualizer as pv

from .mocap_objects import (
    InsoleMarkers, PillowMarkers, PillowBigMarkers, OSAICaseMarkers,
    ElectronicTargetMarkers, ElectronicObjectMarkers, PassportMarkers,
    PassportClosedMarkers, PassportBoxMarkers)


def make_coordinate_system(s, short_tick_length=0.01, long_tick_length=0.05):
    """Make coordinate system.

    Parameters
    ----------
    s : float
        Scale.

    short_tick_length : float, optional (default: 0.01)
        Length of short ticks.

    long_tick_length : float, optional (default: 0.05)
        Length of long ticks (every 5th).

    Returns
    -------
    coordinate_system : o3d.geometry.LineSet
        Coordinate system.
    """
    import open3d as o3d
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


N_FINGER_MARKERS = 10  # TODO load from config?


class ManoHand(pv.Artist):
    """Representation of hand mesh as artist for 3D visualization in Open3D."""
    def __init__(self, mbrm, show_mesh=True, show_vertices=False, show_expected_markers=True):
        self.mbrm = mbrm
        self.show_mesh = show_mesh
        self.show_vertices = show_vertices
        self.show_expected_markers = show_expected_markers
        self.expected_markers = pv.PointCollection3D(
            np.zeros((N_FINGER_MARKERS, 3)), s=0.006, c=(1, 1, 1))

    def set_data(self):
        """Does nothing.

        The mesh will be updated when geometries are requested.
        """
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
        if self.show_expected_markers:
            expected_marker_positions = compute_expected_marker_positions(self.mbrm)
            if self.expected_markers is not None:
                expected_marker_positions = self.mbrm.mano2world_[:3, 3] + np.dot(
                    expected_marker_positions, self.mbrm.mano2world_[:3, :3].T)
                self.expected_markers.set_data(expected_marker_positions)
                geoms.extend(self.expected_markers.geometries)
        return geoms


def compute_expected_marker_positions(mbrm):
    expected_marker_positions = np.zeros((N_FINGER_MARKERS, 3))
    i = 0
    for fn in mbrm.mano_finger_kinematics_:
        finger_kinematics = mbrm.mano_finger_kinematics_[fn]
        if finger_kinematics.has_cached_forward_kinematics():
            positions = finger_kinematics.forward(return_cached_result=True)
            expected_marker_positions[i:i + len(positions)] = positions
            i += len(positions)
    return expected_marker_positions


class MoCapObjectMesh(pv.Artist):
    def __init__(self, mesh_filename, mesh_color=None, show_frame=True):
        self.mesh_filename = mesh_filename
        self.mesh_color = mesh_color
        self.mesh = self.load_mesh()

        self.marker_positions = {
            k: np.copy(v) for k, v in self.default_marker_positions.items()}

        self.markers2origin = np.copy(self.markers2mesh)

        if show_frame:
            self.frame = pv.Frame(np.eye(4), s=0.1)
        else:
            self.frame = None

        self.set_data(**self.marker_positions)

    def load_mesh(self):
        """Load mesh without specific pose.

        Returns
        -------
        mesh : open3d.geometry.TriangleMesh
            Object mesh.
        """
        import open3d as o3d
        mesh = o3d.io.read_triangle_mesh(self.mesh_filename)
        if self.mesh_color is not None:
            mesh.paint_uniform_color(self.mesh_color)
        mesh.compute_triangle_normals()
        return mesh

    def transform_from_mesh_to_origin(self, point_in_mesh):
        """Transform point from mesh frame to origin based on current pose.

        Parameters
        ----------
        point_in_mesh : array, shape (3,)
            Point in mesh coordinate system.

        Returns
        -------
        point_in_origin : array, shape (3,)
            Point in origin frame.
        """
        mesh2markers = pt.invert_transform(self.markers2mesh)
        mesh2origin = pt.concat(mesh2markers, self.markers2origin)
        return pt.transform(mesh2origin, pt.vector_to_point(point_in_mesh))[:3]

    def set_data(self, **kwargs):
        for marker_name, marker_position in kwargs.items():
            if not any(np.isnan(marker_position)):
                self.marker_positions[marker_name] = marker_position

        self.mesh.transform(pt.invert_transform(pt.concat(self.markers2mesh, self.markers2origin)))

        self.markers2origin = self.pose_from_markers(**self.marker_positions)

        self.mesh.transform(pt.concat(self.markers2mesh, self.markers2origin))

        if self.frame is not None:
            self.frame.set_data(self.markers2origin)

    @property
    def geometries(self):
        """Expose geometries.

        Returns
        -------
        geometries : list
            List of geometries that can be added to the visualizer.
        """
        g = [self.mesh]
        if self.frame is not None:
            g += self.frame.geometries
        return g


class Insole(MoCapObjectMesh, InsoleMarkers):
    """Representation of insole mesh.

    Parameters
    ----------
    show_frame : bool, optional (default: True)
        Show frame.
    """
    markers2mesh = pt.transform_from(
        R=pr.active_matrix_from_extrinsic_roll_pitch_yaw(np.deg2rad([180, 0, -4.5])),
        p=np.array([0.04, 0.07, -0.007]))

    def __init__(self, show_frame=True):
        super(Insole, self).__init__(
            mesh_filename=resource_filename("hand_embodiment", "model/objects/insole.stl"),
            mesh_color=np.array([0.37, 0.28, 0.26]),
            show_frame=show_frame)


class PillowSmall(MoCapObjectMesh, PillowMarkers):
    """Representation of small pillow mesh.

    Parameters
    ----------
    show_frame : bool, optional (default: True)
        Show frame.
    """
    markers2mesh = pt.transform_from(
        R=pr.active_matrix_from_extrinsic_roll_pitch_yaw(np.deg2rad([0, 0, 90])),
        p=np.array([0.0, -0.02, 0.095]))

    def __init__(self, show_frame=True):
        super(PillowSmall, self).__init__(
            mesh_filename=resource_filename("hand_embodiment", "model/objects/pillow_small.stl"),
            mesh_color=None,
            show_frame=show_frame)


class PillowBig(MoCapObjectMesh, PillowBigMarkers):
    """Representation of big pillow mesh.

    Parameters
    ----------
    show_frame : bool, optional (default: True)
        Show frame.
    """
    markers2mesh = pt.transform_from(
        R=pr.active_matrix_from_extrinsic_roll_pitch_yaw(np.deg2rad([0, 0, 0])),
        p=np.array([0.0, 0.0, 0.051]))

    def __init__(self, show_frame=True):
        super(PillowBig, self).__init__(
            mesh_filename=resource_filename("hand_embodiment", "model/objects/pillow_big.stl"),
            mesh_color=None,
            show_frame=show_frame)


class OSAICase(MoCapObjectMesh, OSAICaseMarkers):
    """Representation of OSAI case.

    Parameters
    ----------
    show_frame : bool, optional (default: True)
        Show frame.
    """
    markers2mesh = pt.transform_from(
        R=pr.active_matrix_from_extrinsic_roll_pitch_yaw(np.deg2rad([0, 0, 0])),
        p=np.array([0, 0, 0.006]))

    def __init__(self, show_frame=True):
        super(OSAICase, self).__init__(
            mesh_filename=resource_filename("hand_embodiment", "model/objects/electronic_target.stl"),
            mesh_color=np.array([0.21, 0.20, 0.46]),
            show_frame=show_frame)


class ElectronicTarget(MoCapObjectMesh, ElectronicTargetMarkers):
    """Representation of electronic target component.

    Parameters
    ----------
    show_frame : bool, optional (default: True)
        Show frame.
    """
    markers2mesh = pt.transform_from(
        R=pr.active_matrix_from_extrinsic_roll_pitch_yaw(np.deg2rad([0, 0, 0])),
        p=-np.array([0.0625 + 0.014, -0.057 + 0.006, 0.0]) / 2.0)

    def __init__(self, show_frame=True):
        super(ElectronicTarget, self).__init__(
            mesh_filename=resource_filename("hand_embodiment", "model/objects/electronic_target.stl"),
            mesh_color=np.array([0.21, 0.20, 0.46]),
            show_frame=show_frame)


class ElectronicObject(MoCapObjectMesh, ElectronicObjectMarkers):
    """Representation of electronic object component.

    Parameters
    ----------
    show_frame : bool, optional (default: True)
        Show frame.
    """
    markers2mesh = pt.transform_from(
        R=pr.active_matrix_from_extrinsic_roll_pitch_yaw(np.deg2rad([0, 0, 0])),
        p=np.array([0.0, 0.0, -0.01]))

    def __init__(self, show_frame=True):
        super(ElectronicObject, self).__init__(
            mesh_filename=resource_filename("hand_embodiment", "model/objects/electronic_object.stl"),
            mesh_color=np.array([0.68, 0.45, 0.23]),
            show_frame=show_frame)


class Passport(MoCapObjectMesh, PassportMarkers):
    """Representation of open passport.

    Parameters
    ----------
    show_frame : bool, optional (default: True)
        Show frame.
    """
    markers2mesh = pt.transform_from(
        R=pr.active_matrix_from_extrinsic_roll_pitch_yaw(np.deg2rad([0, 0, 0])),
        p=-np.array([0.0, 0.0, 0.0]))

    def __init__(self, show_frame=True):
        super(Passport, self).__init__(
            mesh_filename=resource_filename("hand_embodiment", "model/objects/passport_open.stl"),
            mesh_color=np.array([0.38, 0.48, 0.42]),
            show_frame=show_frame)


class PassportClosed(MoCapObjectMesh, PassportClosedMarkers):
    """Representation of passport.

    Parameters
    ----------
    show_frame : bool, optional (default: True)
        Show frame.
    """
    markers2mesh = pt.transform_from(
        R=pr.active_matrix_from_extrinsic_roll_pitch_yaw(np.deg2rad([0, 0, 0])),
        p=-np.array([0.0, 0.0, 0.008]))

    def __init__(self, show_frame=True):
        super(PassportClosed, self).__init__(
            mesh_filename=resource_filename("hand_embodiment", "model/objects/passport_closed.stl"),
            mesh_color=np.array([0.35, 0.14, 0.21]),
            show_frame=show_frame)


class PassportBox(MoCapObjectMesh, PassportBoxMarkers):
    """Representation of passport box.

    Parameters
    ----------
    show_frame : bool, optional (default: True)
        Show frame.
    """
    markers2mesh = pt.transform_from(
        R=pr.active_matrix_from_extrinsic_roll_pitch_yaw(np.deg2rad([0, 180, 0])),
        p=-np.array([0.0, 0.0, -0.046]))

    def __init__(self, show_frame=True):
        super(PassportBox, self).__init__(
            mesh_filename=resource_filename("hand_embodiment", "model/objects/passport_box.stl"),
            mesh_color=np.array([0.58, 0.46, 0.25]),
            show_frame=show_frame)


ARTISTS = {
    "insole": Insole,
    "pillow-small": PillowSmall,
    "pillow-big": PillowBig,
    "osai-case": OSAICase,
    "electronic-object": ElectronicObject,
    "electronic-target": ElectronicTarget,
    "passport": Passport,
    "passport-closed": PassportClosed,
    "passport-box": PassportBox
}


class AnimationCallback:
    """Animation callback.

    Parameters
    ----------
    fig : pytransform3d.visualizer.Figure
        Figure.

    pipeline : MoCapToRobot
        Pipeline.

    args : argparse.Namespace
        Command line arguments

    show_robot : bool, optional (default: False)
        Show robot.
    """
    def __init__(self, fig, pipeline, args, show_robot=False):
        self.fig = fig
        self.args = args
        self.show_robot = show_robot

        self.show_mano = (hasattr(args, "hide_mano") and not args.hide_mano
                          or hasattr(args, "show_mano") and args.show_mano)
        if self.show_mano:
            self.hand = pipeline.make_hand_artist(
                show_expected_markers=args.show_expected_markers)
            self.hand.add_artist(self.fig)

        self.object_meshes = []
        if self.args.insole:
            self.object_meshes.append(Insole())
        if self.args.pillow:
            self.object_meshes.append(PillowSmall())
        if self.args.pillow_big:
            self.object_meshes.append(PillowBig())
        if self.args.osai_case:
            self.object_meshes.append(OSAICase())
        if self.args.electronic:
            self.object_meshes.append(ElectronicTarget())
            self.object_meshes.append(ElectronicObject())
        if self.args.passport:
            self.object_meshes.append(Passport())
        if self.args.passport_closed:
            self.object_meshes.append(PassportClosed())
            self.object_meshes.append(PassportBox())
        for object_mesh in self.object_meshes:
            object_mesh.add_artist(self.fig)

        if show_robot:
            self.robot = pipeline.make_robot_artist()
            self.robot.add_artist(self.fig)

    def __call__(self, t, markers, dataset, pipeline):
        if t == 1:
            pipeline.reset()
            time.sleep(self.args.delay)

        markers.set_data(dataset.get_markers(t))

        artists = [markers]

        for object_mesh in self.object_meshes:
            marker_names = dataset.config.get("additional_markers", ())
            additional_markers = dataset.get_additional_markers(t)
            object_markers = {}
            for marker_name in object_mesh.marker_names:
                try:
                    marker_index = marker_names.index(marker_name)
                except ValueError as e:
                    raise e from ValueError(
                        f"Could not find index of one of the markers. Available "
                        f"marker names: {', '.join(marker_names)}. Required "
                        f"marker names for object: "
                        f"{', '.join(object_mesh.marker_names)}.")
                try:
                    object_markers[marker_name] = additional_markers[marker_index]
                except IndexError:
                    raise ValueError(
                        f"Mismatch between expected number of additional "
                        f"markers ({len(marker_names)}: "
                        f"{', '.join(marker_names)}) and actual markers. "
                        f"({len(additional_markers)}). Most likely the marker "
                        f"configuration does not match the MoCap recording.")
            object_mesh.set_data(**object_markers)
            artists.append(object_mesh)

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
