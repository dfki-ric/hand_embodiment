"""Visualization utilities."""
import time

from pkg_resources import resource_filename
import numpy as np
import open3d as o3d
import pytransform3d.rotations as pr
import pytransform3d.transformations as pt
import pytransform3d.visualizer as pv

from .mocap_objects import (
    InsoleMarkers, PillowMarkers, ElectronicTargetMarkers,
    ElectronicObjectMarkers, PassportMarkers, PassportClosedMarkers,
    PassportBoxMarkers)


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
        return geoms


class MeshToOriginMixin:
    """Provides transformation from mesh coordinates to origin frame."""
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


class Insole(pv.Artist, InsoleMarkers, MeshToOriginMixin):
    """Representation of insole mesh.

    Parameters
    ----------
    insole_back : array, shape (3,), optional
        Position of insole back marker.

    insole_front : array, shape (3,), optional
        Position of insole front marker.

    show_frame : bool, optional (default: True)
        Show frame.

    Attributes
    ----------
    markers2origin : array, shape (4, 4)
        Pose of marker frame.
    """
    markers2mesh = pt.transform_from(
        R=pr.active_matrix_from_extrinsic_roll_pitch_yaw(np.deg2rad([180, 0, -4.5])),
        p=np.array([0.04, 0.07, -0.007]))

    def __init__(
            self, insole_back=np.copy(InsoleMarkers.insole_back_default),
            insole_front=np.copy(InsoleMarkers.insole_front_default),
            show_frame=True):
        self.mesh_filename = resource_filename(
            "hand_embodiment", "model/objects/insole.stl")
        self.mesh = self.load_mesh()

        self.insole_back = np.copy(self.insole_back_default)
        self.insole_front = np.copy(self.insole_front_default)
        self.markers2origin = np.copy(self.markers2mesh)

        if show_frame:
            self.frame = pv.Frame(np.eye(4), s=0.1)
        else:
            self.frame = None

        self.set_data(insole_back, insole_front)

    def load_mesh(self):
        """Load mesh without specific pose.

        Returns
        -------
        mesh : open3d.geometry.TriangleMesh
            Object mesh.
        """
        mesh = o3d.io.read_triangle_mesh(self.mesh_filename)
        mesh.paint_uniform_color(np.array([0.37, 0.28, 0.26]))
        mesh.compute_triangle_normals()
        return mesh

    def set_data(self, insole_back, insole_front):
        if not any(np.isnan(insole_back)):
            self.insole_back = insole_back
        if not any(np.isnan(insole_front)):
            self.insole_front = insole_front

        self.mesh.transform(pt.invert_transform(pt.concat(self.markers2mesh, self.markers2origin)))

        self.markers2origin = self.pose_from_markers(self.insole_back, self.insole_front)

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


class PillowSmall(pv.Artist, PillowMarkers, MeshToOriginMixin):
    """Representation of small pillow mesh.

    Parameters
    ----------
    pillow_left : array, shape (3,)
        Position of left marker.

    pillow_right : array, shape (3,)
        Position of right marker.

    pillow_top : array, shape (3,)
        Position of top marker.

    show_frame : bool, optional (default: True)
        Show frame.

    Attributes
    ----------
    markers2origin : array, shape (4, 4)
        Pose of marker frame.
    """
    markers2mesh = pt.transform_from(
        R=pr.active_matrix_from_extrinsic_roll_pitch_yaw(np.deg2rad([0, 0, 90])),
        p=np.array([0.0, -0.02, 0.095]))

    def __init__(
            self, pillow_left=np.copy(PillowMarkers.pillow_left_default),
            pillow_right=np.copy(PillowMarkers.pillow_right_default),
            pillow_top=np.copy(PillowMarkers.pillow_top_default),
            show_frame=True):
        self.mesh_filename = resource_filename(
            "hand_embodiment", "model/objects/pillow_small.stl")
        self.mesh = self.load_mesh()

        self.pillow_left = np.copy(self.pillow_left_default)
        self.pillow_right = np.copy(self.pillow_right_default)
        self.pillow_top = np.copy(self.pillow_top_default)

        self.markers2origin = np.copy(self.markers2mesh)

        if show_frame:
            self.frame = pv.Frame(np.eye(4), s=0.1)
        else:
            self.frame = None

        self.set_data(pillow_left, pillow_right, pillow_top)

    def load_mesh(self):
        """Load mesh without specific pose.

        Returns
        -------
        mesh : open3d.geometry.TriangleMesh
            Object mesh.
        """
        mesh = o3d.io.read_triangle_mesh(self.mesh_filename)
        mesh.compute_triangle_normals()
        return mesh

    def set_data(self, pillow_left, pillow_right, pillow_top):
        if not any(np.isnan(pillow_left)):
            self.pillow_left = pillow_left
        if not any(np.isnan(pillow_right)):
            self.pillow_right = pillow_right
        if not any(np.isnan(pillow_top)):
            self.pillow_top = pillow_top

        self.mesh.transform(pt.invert_transform(
            pt.concat(self.markers2mesh, self.markers2origin)))

        self.markers2origin = self.pose_from_markers(
            self.pillow_left, self.pillow_right, self.pillow_top)

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


class ElectronicTarget(pv.Artist, ElectronicTargetMarkers, MeshToOriginMixin):
    """Representation of electronic object and target component."""
    markers2mesh = pt.transform_from(
        R=pr.active_matrix_from_extrinsic_roll_pitch_yaw(np.deg2rad([0, 0, 0])),
        p=-np.array([0.0625 + 0.014, -0.057 + 0.006, 0.0]) / 2.0)

    def __init__(
            self, target_top=np.copy(ElectronicTargetMarkers.target_top_default),
            target_bottom=np.copy(ElectronicTargetMarkers.target_bottom_default),
            show_frame=True):
        self.mesh_filename = resource_filename(
            "hand_embodiment", "model/objects/electronic_target.stl")
        self.mesh = self.load_mesh()

        self.target_top = np.copy(self.target_top_default)
        self.target_bottom = np.copy(self.target_bottom_default)

        self.markers2origin = np.copy(self.markers2mesh)

        if show_frame:
            self.frame = pv.Frame(np.eye(4), s=0.1)
        else:
            self.frame = None

        self.set_data(target_top, target_bottom)

    def load_mesh(self):
        mesh = o3d.io.read_triangle_mesh(self.mesh_filename)
        mesh.paint_uniform_color(np.array([0.21, 0.20, 0.46]))
        mesh.compute_triangle_normals()
        return mesh

    def set_data(self, target_top, target_bottom):
        if not any(np.isnan(target_top)):
            self.target_top = target_top
        if not any(np.isnan(target_bottom)):
            self.target_bottom = target_bottom

        self.mesh.transform(pt.invert_transform(pt.concat(
            self.markers2mesh, self.markers2origin)))
        self.markers2origin = self.pose_from_markers(
            self.target_top, self.target_bottom)
        self.mesh.transform(pt.concat(
            self.markers2mesh, self.markers2origin))

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


class ElectronicObject(pv.Artist, ElectronicObjectMarkers, MeshToOriginMixin):
    """Representation of electronic object and target component."""
    markers2mesh = pt.transform_from(
        R=pr.active_matrix_from_extrinsic_roll_pitch_yaw(np.deg2rad([0, 0, 0])),
        p=np.array([0.0, 0.0, -0.01]))

    def __init__(
            self, object_left=np.copy(ElectronicObjectMarkers.object_left_default),
            object_right=np.copy(ElectronicObjectMarkers.object_right_default),
            object_top=np.copy(ElectronicObjectMarkers.object_top_default),
            show_frame=True):
        self.mesh_filename = resource_filename(
            "hand_embodiment", "model/objects/electronic_object.stl")
        self.mesh = self.load_mesh()

        self.object_left = np.copy(ElectronicObjectMarkers.object_left_default)
        self.object_right = np.copy(ElectronicObjectMarkers.object_right_default)
        self.object_top = np.copy(ElectronicObjectMarkers.object_top_default)

        self.markers2origin = np.copy(self.markers2mesh)

        if show_frame:
            self.frame = pv.Frame(np.eye(4), s=0.1)
        else:
            self.frame = None

        self.set_data(object_left, object_right, object_top)

    def load_mesh(self):
        mesh = o3d.io.read_triangle_mesh(self.mesh_filename)
        mesh.paint_uniform_color(np.array([0.68, 0.45, 0.23]))
        mesh.compute_triangle_normals()
        return mesh

    def set_data(self, object_left, object_right, object_top):
        if not any(np.isnan(object_left)):
            self.object_left = object_left
        if not any(np.isnan(object_right)):
            self.object_right = object_right
        if not any(np.isnan(object_top)):
            self.object_top = object_top

        self.mesh.transform(pt.invert_transform(pt.concat(
            self.markers2mesh, self.markers2origin)))
        self.markers2origin = self.pose_from_markers(
            self.object_left, self.object_right, self.object_top)
        self.mesh.transform(pt.concat(
            self.markers2mesh, self.markers2origin))

        if self.frame:
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


class Passport(pv.Artist, PassportMarkers, MeshToOriginMixin):
    """Representation of open passport.

    Parameters
    ----------
    passport_left : array, shape (3,)
        Left passport marker (PL).

    passport_right : array, shape (3,)
        Right passport marker (PR).

    show_frame : bool, optional (default: True)
        Show frame.

    Attributes
    ----------
    markers2origin : array, shape (4, 4)
        Pose of marker frame.
    """
    markers2mesh = pt.transform_from(
        R=pr.active_matrix_from_extrinsic_roll_pitch_yaw(np.deg2rad([0, 0, 0])),
        p=-np.array([0.0, 0.0, 0.0]))

    def __init__(self, passport_left=np.copy(PassportMarkers.passport_left_default),
                 passport_right=np.copy(PassportMarkers.passport_right_default),
                 show_frame=True):
        self.mesh_filename = resource_filename(
            "hand_embodiment", "model/objects/passport_open.stl")
        self.mesh = self.load_mesh()

        self.passport_left = np.copy(self.passport_left_default)
        self.passport_right = np.copy(self.passport_right_default)

        self.markers2origin = np.copy(self.markers2mesh)

        if show_frame:
            self.frame = pv.Frame(np.eye(4), s=0.1)
        else:
            self.frame = None

        self.set_data(passport_left, passport_right)

    def load_mesh(self):
        """Load mesh without specific pose.

        Returns
        -------
        mesh : open3d.geometry.TriangleMesh
            Object mesh.
        """
        mesh = o3d.io.read_triangle_mesh(self.mesh_filename)
        mesh.paint_uniform_color(np.array([0.38, 0.48, 0.42]))
        mesh.compute_triangle_normals()
        return mesh

    def set_data(self, passport_left, passport_right):
        if not any(np.isnan(passport_left)):
            self.passport_left = passport_left
        if not any(np.isnan(passport_right)):
            self.passport_right = passport_right

        self.mesh.transform(pt.invert_transform(pt.concat(
            self.markers2mesh, self.markers2origin)))
        self.markers2origin = self.pose_from_markers(
            self.passport_left, self.passport_right)
        self.mesh.transform(pt.concat(
            self.markers2mesh, self.markers2origin))

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


class PassportClosed(pv.Artist, PassportClosedMarkers, MeshToOriginMixin):
    """Representation of passport."""
    markers2mesh = pt.transform_from(
        R=pr.active_matrix_from_extrinsic_roll_pitch_yaw(np.deg2rad([0, 0, 0])),
        p=-np.array([0.0, 0.0, 0.008]))

    def __init__(self, passport_top=np.copy(PassportClosedMarkers.passport_top_default),
                 passport_left=np.copy(PassportClosedMarkers.passport_left_default),
                 passport_right=np.copy(PassportClosedMarkers.passport_right_default),
                 show_frame=True):
        self.mesh_filename = resource_filename(
            "hand_embodiment", "model/objects/passport_closed.stl")
        self.mesh = self.load_mesh()

        self.passport_top = np.copy(PassportClosedMarkers.passport_top_default)
        self.passport_left = np.copy(PassportClosedMarkers.passport_left_default)
        self.passport_right = np.copy(PassportClosedMarkers.passport_right_default)

        self.markers2origin = np.copy(self.markers2mesh)

        if show_frame:
            self.frame = pv.Frame(np.eye(4), s=0.1)
        else:
            self.frame = None

        self.set_data(passport_top, passport_left, passport_right)

    def load_mesh(self):
        mesh = o3d.io.read_triangle_mesh(self.mesh_filename)
        mesh.paint_uniform_color(np.array([0.35, 0.14, 0.21]))
        mesh.compute_triangle_normals()
        return mesh

    def set_data(self, passport_top, passport_left, passport_right):
        if not any(np.isnan(passport_top)):
            self.passport_top = passport_top
        if not any(np.isnan(passport_left)):
            self.passport_left = passport_left
        if not any(np.isnan(passport_right)):
            self.passport_right = passport_right

        self.mesh.transform(pt.invert_transform(pt.concat(
            self.markers2mesh, self.markers2origin)))
        self.markers2origin = self.pose_from_markers(
            self.passport_top, self.passport_left, self.passport_right)
        self.mesh.transform(pt.concat(
            self.markers2mesh, self.markers2origin))

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


class PassportBox(pv.Artist, PassportBoxMarkers, MeshToOriginMixin):
    """Representation of passport box."""
    markers2mesh = pt.transform_from(
        R=pr.active_matrix_from_extrinsic_roll_pitch_yaw(np.deg2rad([0, 180, 0])),
        p=-np.array([0.0, 0.0, -0.046]))

    def __init__(self, box_top=np.copy(PassportBoxMarkers.box_top_default),
                 box_left=np.copy(PassportBoxMarkers.box_left_default),
                 box_right=np.copy(PassportBoxMarkers.box_right_default),
                 show_frame=True):
        self.mesh_filename = resource_filename(
            "hand_embodiment", "model/objects/passport_box.stl")
        self.mesh = self.load_mesh()

        self.box_top = np.copy(PassportBoxMarkers.box_top_default)
        self.box_left = np.copy(PassportBoxMarkers.box_left_default)
        self.box_right = np.copy(PassportBoxMarkers.box_right_default)

        self.markers2origin = np.copy(self.markers2mesh)

        if show_frame:
            self.frame = pv.Frame(np.eye(4), s=0.1)
        else:
            self.frames = None

        self.set_data(box_top, box_left, box_right)

    def load_mesh(self):
        mesh = o3d.io.read_triangle_mesh(self.mesh_filename)
        mesh.paint_uniform_color(np.array([0.58, 0.46, 0.25]))
        mesh.compute_triangle_normals()
        return mesh

    def set_data(self, box_top, box_left, box_right):
        if not any(np.isnan(box_top)):
            self.box_top = box_top
        if not any(np.isnan(box_left)):
            self.box_left = box_left
        if not any(np.isnan(box_right)):
            self.box_right = box_right

        self.mesh.transform(pt.invert_transform(pt.concat(
            self.markers2mesh, self.markers2origin)))
        self.markers2origin = self.pose_from_markers(
            self.box_top, self.box_left, self.box_right)
        self.mesh.transform(pt.concat(
            self.markers2mesh, self.markers2origin))

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


class AnimationCallback:
    """Animation callback.

    Parameters
    ----------
    fig : pytransform3d.visualizer.Figure
        Figure.

    pipeline : MoCapToRobot
        Pipeline.

    args : TODO result of ArgumentParser.parse_args()
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
            self.hand = pipeline.make_hand_artist()
            self.hand.add_artist(self.fig)

        self.object_meshes = []
        if self.args.insole:
            self.object_meshes.append(Insole())
        if self.args.pillow:
            self.object_meshes.append(PillowSmall())
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
            object_markers = {
                marker_name: additional_markers[marker_names.index(marker_name)]
                for marker_name in object_mesh.marker_names}
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
