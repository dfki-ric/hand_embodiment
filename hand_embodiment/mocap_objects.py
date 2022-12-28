"""Information about objects used in motion capture recordings."""
import numpy as np
from pytransform3d import rotations as pr, transformations as pt


class InsoleMarkers:
    """Information about insole markers.

    Marker positions:

    .. code-block:: text

        IB-------------IF
    """
    default_marker_positions = {
        "insole_back": np.zeros(3),
        "insole_front": np.array([0.19, 0, 0])
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
        x_axis = pr.norm_vector(insole_front - insole_back)
        z_axis = np.copy(pr.unitz)
        y_axis = pr.norm_vector(pr.perpendicular_to_vectors(z_axis, x_axis))
        z_axis = pr.norm_vector(pr.perpendicular_to_vectors(x_axis, y_axis))
        R = np.column_stack((x_axis, y_axis, z_axis))
        return pt.transform_from(R=R, p=insole_back)


class InsoleBagMarkers:
    """Information about insole bag markers.

    Marker positions:

    .. code-block:: text

        ----------
        |        |
        |        |
        |       BX
        |        |
        |        |
        |        |
        BY------BO
    """
    default_marker_positions = {
        "bag_origin": np.array([0.0, 0.0, 0.0]),
        "bag_x": np.array([0.156, -0.006, 0.0]),
        "bag_y": np.array([0.002, 0.1, 0.0])
    }
    marker_names = tuple(default_marker_positions.keys())

    @staticmethod
    def pose_from_markers(bag_origin, bag_x, bag_y):
        """Compute pose of bag.

        Parameters
        ----------
        bag_origin : array, shape (3,)
            Position of the bag.

        bag_x : array, shape (3,)
            Defines direction of the x-axis.

        bag_y : array, shape (3,)
            Defines direction of the y-axis.

        Returns
        -------
        pose : array, shape (4, 4)
            Pose of the bag.
        """
        x_axis = pr.norm_vector(bag_x - bag_origin)
        y_axis = pr.norm_vector(bag_y - bag_origin)
        z_axis = np.cross(x_axis, y_axis)
        y_axis = pr.norm_vector(pr.perpendicular_to_vectors(z_axis, x_axis))
        z_axis = pr.norm_vector(pr.perpendicular_to_vectors(x_axis, y_axis))
        R = np.column_stack((x_axis, y_axis, z_axis))
        return pt.transform_from(R=R, p=bag_origin)


class ProtectorMarkers:
    """Information about protector markers.

    Marker positions:

    .. code-block:: text

        PL---PR
        |
        |
        |
        PB
    """
    default_marker_positions = {
        "protector_left": np.array([0.09, 0.0, 0.0]),
        "protector_right": np.array([0.073, -0.0205, 0.0]),
        "protector_bottom": np.array([0.0, 0.0, 0.0])
    }
    marker_names = tuple(default_marker_positions.keys())

    @staticmethod
    def pose_from_markers(protector_left, protector_right, protector_bottom):
        """Compute pose of insole.

        Parameters
        ----------
        protector_left : array, shape (3,)
            Position of protector left marker (PL).

        protector_right : array, shape (3,)
            Position of protector right marker (PR).

        protector_bottom : array, shape (3,)
            Position of protector bottom marker (PB).

        Returns
        -------
        pose : array, shape (4, 4)
            Pose of the insole.
        """
        x_axis = pr.norm_vector(protector_left - protector_bottom)
        side = protector_left - protector_right
        y_axis = pr.norm_vector(side - pr.vector_projection(side, x_axis))
        z_axis = pr.perpendicular_to_vectors(x_axis, y_axis)
        R = np.column_stack((x_axis, y_axis, z_axis))
        return pt.transform_from(R=R, p=protector_bottom)


class ProtectorInvertedMarkers:
    """Information about inverted protector markers.

    Marker positions:

    .. code-block:: text

        PT
        |
        |
        |
        PL---PR
    """
    default_marker_positions = {
        "protector_left": np.array([0.09, 0.0, 0.0]),
        "protector_right": np.array([0.073, -0.0205, 0.0]),
        "protector_top": np.array([0.0, 0.0, 0.0])
    }
    marker_names = tuple(default_marker_positions.keys())

    @staticmethod
    def pose_from_markers(protector_top, protector_left, protector_right):
        """Compute pose of insole.

        Parameters
        ----------
        protector_top : array, shape (3,)
            Position of protector top marker (PT).

        protector_left : array, shape (3,)
            Position of protector left marker (PL).

        protector_right : array, shape (3,)
            Position of protector right marker (PR).

        Returns
        -------
        pose : array, shape (4, 4)
            Pose of the insole.
        """
        x_axis = pr.norm_vector(protector_left - protector_top)
        side = protector_right - protector_left
        y_axis = pr.norm_vector(side - pr.vector_projection(side, x_axis))
        z_axis = pr.perpendicular_to_vectors(x_axis, y_axis)
        R = np.column_stack((x_axis, y_axis, z_axis))
        return pt.transform_from(R=R, p=protector_top)


class PillowMarkers:
    """Information about small pillow markers.

    Marker positions:

    .. code-block:: text

        ---------------PT
        |               |
        |               |
        |               |
        |               |
        PL-------------PR
    """
    default_marker_positions = {
        "pillow_left": np.array([-0.11, 0.13, 0]),
        "pillow_right": np.array([-0.11, -0.13, 0]),
        "pillow_top": np.array([0.11, -0.13, 0])
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
        right2top = pillow_top - pillow_right
        right2left = pillow_left - pillow_right
        pose = np.eye(4)
        pose[:3, :3] = pr.matrix_from_two_vectors(right2top, right2left)
        pillow_middle = 0.5 * (pillow_left + pillow_right) + 0.5 * right2top
        pose[:3, 3] = pillow_middle
        return pose


class PillowBigMarkers:
    """Information about big pillow markers.

    Marker positions:

    .. code-block:: text

        -------PT--------
        |               |
        |               |
        |               |
        |               |
        PL-------------PR
    """
    default_marker_positions = {
        "pillow_left": np.array([0.0, 0.0, 0.0]),
        "pillow_right": np.array([0.385, 0.008, 0.0]),
        "pillow_top": np.array([0.183, 0.295, 0.0])
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
        x_axis = pr.norm_vector(pillow_right - pillow_left)
        top_on_x_axis = pillow_left + np.dot(x_axis, pillow_top - pillow_left) * x_axis
        y_axis = pr.norm_vector(pillow_top - top_on_x_axis)
        z_axis = pr.norm_vector(np.cross(x_axis, y_axis))
        x_axis = pr.norm_vector(np.cross(y_axis, z_axis))
        R = np.column_stack((x_axis, y_axis, z_axis))
        return pt.transform_from(R=R, p=pillow_left)


class PillowSssaMarkers:
    """Information about small pillow markers.

    Marker positions:

    .. code-block:: text

        -----------------
        |       x3      |
        |               |
        | x2    x1      |
        |               |
        |               |
        -----------------
    """
    default_marker_positions = {
        "PEMU_1": np.array([0.0, 0.0, 0.0]),
        "PEMU_2": np.array([0.0, 0.05, 0.0]),
        "PEMU_3": np.array([0.05, 0.0, 0.0])
    }
    marker_names = tuple(default_marker_positions.keys())

    @staticmethod
    def pose_from_markers(PEMU_1, PEMU_2, PEMU_3):
        """Compute pose of pillow.

        Parameters
        ----------
        PEMU_1 : array, shape (3,)
            Center of the pillow.

        PEMU_2 : array, shape (3,)
            Positive along the y-axis.

        PEMU_3 : array, shape (3,)
            Positive along the x-axis.

        Returns
        -------
        pose : array, shape (4, 4)
            Pose of the pillow.
        """
        x_axis = pr.norm_vector(PEMU_2 - PEMU_1)
        y_axis = pr.norm_vector(PEMU_3 - PEMU_1)
        z_axis = np.cross(x_axis, y_axis)
        x_axis = np.cross(y_axis, z_axis)
        return pt.transform_from(
            R=pr.norm_matrix(np.column_stack((x_axis, y_axis, z_axis))),
            p=PEMU_1
        )


class OSAICaseMarkers:
    """Information about OSAI case markers.

    Marker positions:

    .. code-block:: text

        OSAI_3-----------------
        |                     |
        |                     |
        |                     |
        |                     |
        OSAI_1-----------OSAI_2
    """
    default_marker_positions = {
        "OSAI_1": np.array([-0.031, -0.028, 0.0]),
        "OSAI_2": np.array([0.031, -0.028, 0.0]),
        "OSAI_3": np.array([-0.031, 0.028, 0.0])
    }
    marker_names = tuple(default_marker_positions.keys())

    @staticmethod
    def pose_from_markers(OSAI_1, OSAI_2, OSAI_3):
        """Compute pose of OSAI case.

        Parameters
        ----------
        OSAI_1 : array, shape (3,)
            Position first marker.

        OSAI_2 : array, shape (3,)
            Position of second marker.

        OSAI_3 : array, shape (3,)
            Position of third marker.

        Returns
        -------
        pose : array, shape (4, 4)
            Pose of the electronic target.
        """
        x_axis = pr.norm_vector(OSAI_2 - OSAI_1)
        y_axis = pr.norm_vector(OSAI_3 - OSAI_1)
        z_axis = pr.norm_vector(np.cross(x_axis, y_axis))
        y_axis = pr.norm_vector(np.cross(z_axis, x_axis))
        R = np.column_stack((x_axis, y_axis, z_axis))
        center = OSAI_1 + 0.031 * x_axis + 0.028 * y_axis
        return pt.transform_from(R=R, p=center)


class OSAICaseSmallMarkers:
    """Information about small OSAI case markers.

    Marker positions:

    .. code-block:: text

        OSAI_3----------------------
        |                          |
        |                          |
        |                          |
        |                          |
        OSAI_1----------------OSAI_2
    """
    default_marker_positions = {
        "OSAI_1": np.array([-0.029 + 0.006, -0.017 + 0.006, 0.0]),
        "OSAI_2": np.array([0.029 - 0.007, -0.017 + 0.007, 0.0]),
        "OSAI_3": np.array([-0.029 + 0.006, 0.017 - 0.007, 0.0])
    }
    marker_names = tuple(default_marker_positions.keys())

    @staticmethod
    def pose_from_markers(OSAI_1, OSAI_2, OSAI_3):
        """Compute pose of OSAI case.

        Parameters
        ----------
        OSAI_1 : array, shape (3,)
            Position first marker.

        OSAI_2 : array, shape (3,)
            Position of second marker.

        OSAI_3 : array, shape (3,)
            Position of third marker.

        Returns
        -------
        pose : array, shape (4, 4)
            Pose of the electronic target.
        """
        x_axis = pr.norm_vector(OSAI_2 - OSAI_1)
        y_axis = pr.norm_vector(OSAI_3 - OSAI_1)
        z_axis = pr.norm_vector(np.cross(x_axis, y_axis))
        y_axis = pr.norm_vector(np.cross(z_axis, x_axis))
        R = np.column_stack((x_axis, y_axis, z_axis))
        center = OSAI_1 + 0.023 * x_axis + 0.011 * y_axis
        return pt.transform_from(R=R, p=center)


class ElectronicTargetMarkers:
    """Information about electronic target markers.

    Marker positions:

    .. code-block:: text

        TT-------------TB
    """
    default_marker_positions = {
        "target_top": np.array([0.076, 0, 0]),
        "target_bottom": np.array([0.0, 0, 0])
    }
    marker_names = tuple(default_marker_positions.keys())

    @staticmethod
    def pose_from_markers(target_top, target_bottom):
        """Compute pose of electronic target.

        Parameters
        ----------
        target_top : array, shape (3,)
            Position of top marker (TT).

        target_bottom : array, shape (3,)
            Position of bottom marker (TB).

        Returns
        -------
        pose : array, shape (4, 4)
            Pose of the electronic target.
        """
        x_axis = pr.norm_vector(target_top - target_bottom)
        z_axis = np.copy(pr.unitz)
        y_axis = pr.norm_vector(pr.perpendicular_to_vectors(z_axis, x_axis))
        z_axis = pr.norm_vector(pr.perpendicular_to_vectors(x_axis, y_axis))
        R = np.column_stack((x_axis, y_axis, z_axis))
        return pt.transform_from(R=R, p=target_bottom)


class ElectronicObjectMarkers:
    """Information about the electronic object markers.

    Marker positions:

    .. code-block:: text

        OT---------------
        |               |
        |               |
        |               |
        |               |
        OL-------------OR
    """
    default_marker_positions = {
        "object_left": np.array([0.025, -0.03, 0]),
        "object_right": np.array([-0.025, -0.03, 0]),
        "object_top": np.array([0.025, 0.03, 0])
    }
    marker_names = tuple(default_marker_positions.keys())

    @staticmethod
    def pose_from_markers(object_left, object_right, object_top):
        """Compute pose of electronic object.

        Parameters
        ----------
        object_left : array, shape (3,)
            Position of left marker (OL).

        object_right : array, shape (3,)
            Position of right marker (OR).

        object_top : array, shape (3,)
            Position of top marker (OT).

        Returns
        -------
        pose : array, shape (4, 4)
            Pose of the electronic object.
        """
        left2top = object_top - object_left
        left2right = object_left - object_right
        pose = np.eye(4)
        pose[:3, :3] = pr.matrix_from_two_vectors(left2right, left2top)
        object_middle = 0.5 * (object_left + object_right) + 0.5 * left2top
        pose[:3, 3] = object_middle
        return pose


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
    default_marker_positions = {
        "passport_left": np.array([-0.103, 0.0, 0.0]),
        "passport_right": np.array([0.103, 0.0, 0.0])
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
        x_axis = pr.norm_vector(passport_right - passport_left)
        z_axis = np.copy(pr.unitz)
        y_axis = pr.norm_vector(pr.perpendicular_to_vectors(z_axis, x_axis))
        z_axis = pr.norm_vector(pr.perpendicular_to_vectors(x_axis, y_axis))
        R = np.column_stack((x_axis, y_axis, z_axis))
        return pt.transform_from(R=R, p=0.5 * (passport_right + passport_left))


class PassportClosedMarkers:
    """Information about closed passport markers.

    Marker positions:

    .. code-block:: text

        ---------PT
        |         |
        |         |
        |         |
        PL-------PR
    """
    default_marker_positions = {
        "passport_top": np.array([-0.043, 0.06, 0]),
        "passport_left": np.array([0.043, -0.06, 0]),
        "passport_right": np.array([-0.043, -0.06, 0])
    }
    marker_names = tuple(default_marker_positions.keys())

    @staticmethod
    def pose_from_markers(passport_top, passport_left, passport_right):
        """Compute pose of passport.

        Parameters
        ----------
        passport_top : array, shape (3,)
            Top passport marker (PT).

        passport_left : array, shape (3,)
            Left passport marker (PL).

        passport_right : array, shape (3,)
            Right passport marker (PR).

        Returns
        -------
        pose : array, shape (4, 4)
            Pose of the passport.
        """
        left2top = passport_top - passport_left
        left2right = passport_left - passport_right
        pose = np.eye(4)
        pose[:3, :3] = pr.matrix_from_two_vectors(left2right, left2top)
        object_middle = passport_left + 0.5 * left2top
        pose[:3, 3] = object_middle
        return pose


class PassportBoxMarkers:
    """Information about passport box markers.

    Marker positions:

    .. code-block:: text

        BT---------
        |         |
        |         |
        |         |
        BL-------BR
    """
    default_marker_positions = {
        "box_top": np.array([0.065, 0.08, 0]),
        "box_left": np.array([0.065, -0.08, 0]),
        "box_right": np.array([-0.065, -0.08, 0])
    }
    marker_names = tuple(default_marker_positions.keys())

    @staticmethod
    def pose_from_markers(box_top, box_left, box_right):
        """Compute pose of box.

        Parameters
        ----------
        box_top : array, shape (3,)
            Top box marker (BT).

        box_left : array, shape (3,)
            Left box marker (BL).

        box_right : array, shape (3,)
            Right box marker (BR).

        Returns
        -------
        pose : array, shape (4, 4)
            Pose of the box.
        """
        left2top = box_top - box_left
        left2right = box_left - box_right
        pose = np.eye(4)
        pose[:3, :3] = pr.matrix_from_two_vectors(left2right, left2top)
        object_middle = 0.5 * (box_left + box_right) + 0.5 * left2top
        pose[:3, 3] = object_middle
        return pose


MOCAP_OBJECTS = {
    "insole": InsoleMarkers,
    "insolebag": InsoleBagMarkers,
    "protector": ProtectorMarkers,
    "protector-inverted": ProtectorInvertedMarkers,
    "pillow": PillowMarkers,
    "pillow-small": PillowMarkers,
    "pillow-big": PillowBigMarkers,
    "pillow-sssa": PillowSssaMarkers,
    "osai-case": OSAICaseMarkers,
    "osai-case-small": OSAICaseSmallMarkers,
    "electronic-object": ElectronicObjectMarkers,
    "electronic-target": ElectronicTargetMarkers,
    "passport": PassportMarkers,
    "passport-closed": PassportClosedMarkers,
    "passport-box": PassportBoxMarkers
}


def extract_mocap_origin2object_generic(args, dataset):
    deprecated_names = [
        "insole_hack", "pillow_hack", "pillow_big_hack",
        "osai_case_hack", "electronic_object_hack", "electronic_target_hack",
        "passport_hack", "passport_closed_hack", "passport_box_hack"]
    deprecated_artist_arguments(args, deprecated_names)
    if args.base_frame is not None:
        mocap_origin2origin = extract_mocap_origin2object(
            dataset, MOCAP_OBJECTS[args.base_frame])
        if args.constant_base_frame:
            mocap_origin2origin = mocap_origin2origin[0]
    else:
        mocap_origin2origin = None
    return mocap_origin2origin


def deprecated_artist_arguments(args, names):
    for name in names:
        if getattr(args, name):
            object_name = name[:-5]
            object_name = object_name.replace("_", "-")
            raise ValueError(
                f"Deprecated command line argument: '--{name}'. Please use "
                f"the new style of indicating object base frames: "
                f"'--base-frame {object_name}'")


def extract_mocap_origin2object(dataset, object_info):
    """Extract transformation from MoCap origin to object.

    Parameters
    ----------
    dataset : hand_embodiment.mocap_dataset.MotionCaptureDatasetBase
        MoCap dataset.

    object_info : class
        MoCap object information.

    Returns
    -------
    mocap_origin2object : array, shape (n_steps, 4, 4)
        Transform from MoCap origin to object frame.
    """
    mocap_origin2object = np.empty((dataset.n_steps, 4, 4))
    marker_positions = {
        k: np.copy(v) for k, v in object_info.default_marker_positions.items()}
    for t in range(dataset.n_steps):
        additional_markers = dataset.get_additional_markers(t)
        marker_names = dataset.config.get("additional_markers", ())
        for marker_name in object_info.marker_names:
            if not any(np.isnan(additional_markers[marker_names.index(marker_name)])):
                marker_positions[marker_name] = additional_markers[marker_names.index(marker_name)]
        object2mocap_origin = object_info.pose_from_markers(**marker_positions)
        mocap_origin2object[t] = pt.invert_transform(object2mocap_origin)
    return mocap_origin2object
