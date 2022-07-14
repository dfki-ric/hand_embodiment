"""Load transformations from URDF files.

See :doc:`transform_manager` for more information.
"""
import math
import numba
import numpy as np
from pytransform3d.rotations import norm_vector
from pytransform3d.urdf import parse_urdf, initialize_urdf_transform_manager
from .tree import TransformManager


class UrdfTransformManager(TransformManager):
    """Transformation manager that can load URDF files.

    URDF is the `Unified Robot Description Format <http://wiki.ros.org/urdf>`_.
    URDF allows to define joints between links that can be rotated about one
    axis. This transformation manager allows to set the joint angles after
    joints have been added or loaded from an URDF.

    This version has efficient numba-accelerated code to update joints.

    .. warning::

        Note that this module requires the Python package beautifulsoup4.

    .. note::

        Joint angles must be given in radians.
    """
    def __init__(self):
        super(UrdfTransformManager, self).__init__()
        self.visuals = []
        self.collision_objects = []
        self._joints = {}
        self.virtual_joints = {}

    def add_joint(self, joint_name, from_frame, to_frame, child2parent, axis,
                  limits=(float("-inf"), float("inf")), joint_type="revolute"):
        """Add joint.

        Parameters
        ----------
        joint_name : str
            Name of the joint

        from_frame : Hashable
            Child link of the joint

        to_frame : Hashable
            Parent link of the joint

        child2parent : array-like, shape (4, 4)
            Transformation from child to parent

        axis : array-like, shape (3,)
            Rotation axis of the joint (defined in the child frame)

        limits : pair of float, optional (default: (-inf, inf))
            Lower and upper joint angle limit

        joint_type : str, optional (default: 'revolute')
            Joint type: revolute or prismatic (continuous is the same as
            revolute)
        """
        self.add_transform(from_frame, to_frame, child2parent)
        self._joints[joint_name] = (
            from_frame, to_frame, child2parent, norm_vector(axis), limits,
            joint_type)

    def set_joint(self, joint_name, value):
        """Set joint position.

        Note that joint values are clipped to their limits.

        Parameters
        ----------
        joint_name : string
            Name of the joint

        value : float
            Joint angle in radians in case of revolute joints or position
            in case of prismatic joint.
        """
        if joint_name in self.virtual_joints:
            callback = self.virtual_joints[joint_name]
            actual_joint_states = callback(value)
            for actual_joint_name, actual_value in actual_joint_states.items():
                self.set_joint(actual_joint_name, actual_value)
            return

        from_frame, to_frame, child2parent, axis, limits, joint_type = self._joints[joint_name]
        value = min(max(value, limits[0]), limits[1])
        if joint_type == "revolute":
            joint2A = _fast_matrix_from_axis_angle(axis, value)
        else:
            joint2A = np.eye(4)
            joint2A[:3, 3] = value * axis
        self.transforms[(from_frame, to_frame)] = child2parent.dot(joint2A)

    def get_joint_limits(self, joint_name):
        """Get limits of a joint.

        Parameters
        ----------
        joint_name : str
            Name of the joint

        Returns
        -------
        limits : pair of float
            Lower and upper joint angle limit

        Raises
        ------
        KeyError
            If joint_name is unknown
        """
        if joint_name not in self._joints:
            raise KeyError("Joint '%s' is not known" % joint_name)
        return self._joints[joint_name][4]

    def load_urdf(self, urdf_xml, mesh_path=None, package_dir=None):
        """Load URDF file into transformation manager.

        Parameters
        ----------
        urdf_xml : str
            Robot definition in URDF

        mesh_path : str, optional (default: None)
            Path in which we search for meshes that are defined in the URDF.
            Meshes will be ignored if it is set to None and no 'package_dir'
            is given.

        package_dir : str, optional (default: None)
            Some URDFs start file names with 'package://' to refer to the ROS
            package in which these files (textures, meshes) are located. This
            variable defines to which path this prefix will be resolved.
        """
        robot_name, links, joints = parse_urdf(
            urdf_xml, mesh_path, package_dir, False)
        initialize_urdf_transform_manager(self, robot_name, links, joints)

    def get_ee2base(self, ee_index, base_index):
        """Request a transform.

        Parameters
        ----------
        ee_index : int
            Index of the end-effector node

        base_index : int
            Index of the base node

        Returns
        -------
        ee2base : array-like, shape (4, 4)
            Homogeneous matrix that represents the transform from ee to base
        """
        return self._path_transform(self._shortest_path(ee_index, base_index))

    def add_virtual_joint(self, joint_name, callback):
        """Add virtual joint.

         A virtual joint is a wrapper that controls multiple other joints.

        Parameters
        ----------
        joint_name : str
            Name of the new virtual joint.

        callback : callable
            A callable object that provides the function 'make_virtual_joint'
            to initialize the transform manager. The function call operator
            will be used to set the joint angle of the virtual joint.
        """
        self.virtual_joints[joint_name] = callback
        self._joints[joint_name] = callback.make_virtual_joint(
            joint_name, self)


@numba.jit(nopython=True, cache=True)
def _fast_matrix_from_axis_angle(axis, angle):
    """Compute transformation matrix from axis-angle.

    Parameters
    ----------
    a : array-like, shape (4,)
        Axis of rotation and rotation angle: (x, y, z, angle)

    Returns
    -------
    A2B : array-like, shape (4, 4)
        Transformation matrix
    """
    ux, uy, uz = axis
    c = math.cos(angle)
    s = math.sin(angle)
    ci = 1.0 - c
    ciux = ci * ux
    ciuy = ci * uy
    ciuz = ci * uz
    return np.array([
        [ciux * ux + c, ciux * uy - uz * s, ciux * uz + uy * s, 0.0],
        [ciuy * ux + uz * s, ciuy * uy + c, ciuy * uz - ux * s, 0.0],
        [ciuz * ux - uy * s, ciuz * uy + ux * s, ciuz * uz + c, 0.0],
        [0.0, 0.0, 0.0, 1.0]
    ])
