import numpy as np
from pytransform3d import rotations as pr, transformations as pt


class InsoleMarkers:
    """Information about insole markers.

    Marker positions:

    .. code-block:: text

        IB-------------IF
    """
    insole_back_default = np.zeros(3)
    insole_front_default = np.array([0.19, 0, 0])
    default_marker_positions = {
        "insole_back": insole_back_default,
        "insole_front": insole_front_default
    }
    marker_names = tuple(default_marker_positions.keys())

    @staticmethod
    def pose_from_markers(insole_back, insole_front):
        """Compute pose of insole.

        Parameters
        ----------
        insole_back : array, shape (3,)
            Position of insole back marker (IB).

        insole_front : array, shape (3,)
            Position of insole front marker (IF).

        Returns
        -------
        pose : array, shape (4, 4)
            Pose of the insole.
        """
        return insole_pose(insole_back, insole_front)


class PillowMarkers:
    """Information about small pillow markers.

    Marker positions:

    .. code-block:: text

                        PT
                        |
                        |
                        |
                        |
        PL-------------PR
    """
    pillow_left_default = np.array([-0.11, 0.13, 0])
    pillow_right_default = np.array([-0.11, -0.13, 0])
    pillow_top_default = np.array([0.11, -0.13, 0])
    default_marker_positions = {
        "pillow_left": pillow_left_default,
        "pillow_right": pillow_right_default,
        "pillow_top": pillow_top_default
    }
    marker_names = tuple(default_marker_positions.keys())

    @staticmethod
    def pose_from_markers(pillow_left, pillow_right, pillow_top):
        """Compute pose of pillow.

        Parameters
        ----------
        pillow_left : array, shape (3,)
            Position of left marker (PL).

        pillow_right : array, shape (3,)
            Position of right marker (PR).

        pillow_top : array, shape (3,)
            Position of top marker (PT).

        Returns
        -------
        pose : array, shape (4, 4)
            Pose of the pillow.
        """
        return pillow_pose(pillow_left, pillow_right, pillow_top)


class ElectronicTargetMarkers:
    target_top_default = np.array([0, 0, 0])
    target_bottom_default = np.array([1, 0, 0])
    default_marker_positions = {
        "target_top": target_top_default,
        "target_bottom": target_bottom_default
    }
    marker_names = tuple(default_marker_positions.keys())

    @staticmethod
    def pose_from_markers(target_top, target_bottom):
        return electronic_target_pose(target_top, target_bottom)


class ElectronicObjectMarkers:
    object_left_default = np.zeros(3)
    object_right_default = np.array([1, 0, 0])
    object_top_default = np.array([1, 1, 0])
    default_marker_positions = {
        "object_left": object_left_default,
        "object_right": object_right_default,
        "object_top": object_top_default
    }
    marker_names = tuple(default_marker_positions.keys())

    @staticmethod
    def pose_from_markers(object_left, object_right, object_top):
        return electronic_object_pose(object_left, object_right, object_top)


class PassportMarkers:
    """Information about passport markers.

    Marker positions:

    .. code-block:: text

        PL  -------------  PR
            |           |
            |           |
            |           |
            -------------
    """
    passport_left_default = np.array([-0.103, 0.0, 0.0])
    passport_right_default = np.array([0.103, 0.0, 0.0])
    default_marker_positions = {
        "passport_left": passport_left_default,
        "passport_right": passport_right_default
    }
    marker_names = tuple(default_marker_positions.keys())

    @staticmethod
    def pose_from_markers(passport_left, passport_right):
        """Compute pose of passport.

        Parameters
        ----------
        passport_left : array, shape (3,)
            Left passport marker (PL).

        passport_right : array, shape (3,)
            Right passport marker (PR).

        Returns
        -------
        pose : array, shape (4, 4)
            Pose of the passport.
        """
        return passport_pose(passport_left, passport_right)


class PassportClosedMarkers:
    passport_top_default = np.array([0, 1, 0])
    passport_left_default = np.zeros(3)
    passport_right_default = np.array([1, 0, 0])
    default_marker_positions = {
        "passport_top": passport_top_default,
        "passport_left": passport_left_default,
        "passport_right": passport_right_default
    }
    marker_names = tuple(default_marker_positions.keys())

    @staticmethod
    def pose_from_markers(passport_top, passport_left, passport_right):
        return passport_closed_pose(passport_top, passport_left, passport_right)


class PassportBoxMarkers:
    box_top_default = np.array([0, 1, 0])
    box_left_default = np.zeros(3)
    box_right_default = np.array([1, 0, 0])
    default_marker_positions = {
        "box_top": box_top_default,
        "box_left": box_left_default,
        "box_right": box_right_default
    }
    marker_names = tuple(default_marker_positions.keys())

    @staticmethod
    def pose_from_markers(box_top, box_left, box_right):
        return box_pose(box_top, box_left, box_right)


def insole_pose(insole_back, insole_front):
    """Compute pose of insole.

    Parameters
    ----------
    insole_back : array, shape (3,)
        Position of insole back marker.

    insole_front : array, shape (3,)
        Position of insole front marker.

    Returns
    -------
    pose : array, shape (4, 4)
        Pose of the insole.
    """
    x_axis = pr.norm_vector(insole_front - insole_back)
    z_axis = np.copy(pr.unitz)
    y_axis = pr.norm_vector(pr.perpendicular_to_vectors(z_axis, x_axis))
    z_axis = pr.norm_vector(pr.perpendicular_to_vectors(x_axis, y_axis))
    R = np.column_stack((x_axis, y_axis, z_axis))
    return pt.transform_from(R=R, p=insole_back)


def pillow_pose(pillow_left, pillow_right, pillow_top):
    """Compute pose of pillow.

    Parameters
    ----------
    pillow_left : array, shape (3,)
        Position of left marker (PL).

    pillow_right : array, shape (3,)
        Position of right marker (PR).

    pillow_top : array, shape (3,)
        Position of top marker (PT).

    Returns
    -------
    pose : array, shape (4, 4)
        Pose of the pillow.
    """
    right2top = pillow_top - pillow_right
    right2left = pillow_left - pillow_right
    pose = np.eye(4)
    pose[:3, :3] = pr.matrix_from_two_vectors(right2top, right2left)
    pillow_middle = 0.5 * (pillow_left + pillow_right) + 0.5 * right2top
    pose[:3, 3] = pillow_middle
    return pose


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


def passport_pose(passport_left, passport_right):
    """Compute pose of passport.

    Parameters
    ----------
    passport_left : array, shape (3,)
        Left passport marker (PL).

    passport_right : array, shape (3,)
        Right passport marker (PR).

    Returns
    -------
    pose : array, shape (4, 4)
        Pose of the passport.
    """
    x_axis = pr.norm_vector(passport_right - passport_left)
    z_axis = np.copy(pr.unitz)
    y_axis = pr.norm_vector(pr.perpendicular_to_vectors(z_axis, x_axis))
    z_axis = pr.norm_vector(pr.perpendicular_to_vectors(x_axis, y_axis))
    R = np.column_stack((x_axis, y_axis, z_axis))
    return pt.transform_from(R=R, p=0.5 * (passport_right + passport_left))


def passport_closed_pose(passport_top, passport_left, passport_right):
    """Compute pose of closed passport."""
    left2top = passport_top - passport_left
    left2right = passport_left - passport_right
    pose = np.eye(4)
    pose[:3, :3] = pr.matrix_from_two_vectors(left2right, left2top)
    object_middle = passport_left + 0.5 * left2top
    pose[:3, 3] = object_middle
    return pose


def box_pose(box_top, box_left, box_right):
    """Compute pose of passport box."""
    left2top = box_top - box_left
    left2right = box_left - box_right
    pose = np.eye(4)
    pose[:3, :3] = pr.matrix_from_two_vectors(left2right, left2top)
    object_middle = 0.5 * (box_left + box_right) + 0.5 * left2top
    pose[:3, 3] = object_middle
    return pose