"""Load transformations from URDF files."""
import math
import numpy as np
cimport numpy as np
cimport cython
from pytransform3d.urdf import UrdfTransformManager


@cython.cclass
class FastUrdfTransformManager:
    """Transformation manager that can load URDF files.

    URDF is the `Unified Robot Description Format <http://wiki.ros.org/urdf>`_.
    URDF allows to define joints between links that can be rotated about one
    axis. This transformation manager allows to set the joint angles after
    joints have been added or loaded from an URDF.

    .. warning::

        Note that this module requires the Python package beautifulsoup4.

    .. note::

        Joint angles must be given in radians.
    """
    _tm : object
    _compiled : bool
    _virtual_joints : dict
    _cached_shortest_paths : dict
    _transforms : dict

    def __init__(self):
        self._tm = UrdfTransformManager()
        self._compiled = False
        self._virtual_joints = {}

        self._cached_shortest_paths = {}
        self._transforms = {}

    @property
    def visuals(self):
        return self._tm.visuals

    @property
    def collision_objects(self):
        return self._tm.collision_objects

    @property
    def nodes(self):
        return self._tm.nodes

    def compile(self, joint_names, base_frame, ee_frames):
        """Compile kinematic tree.

        Parameters
        ----------
        joint_names : list
            Names of joints that should be used

        base_frame : str
            Name of the base link

        ee_frames : list of str
            Name of the end-effector links
        """
        self._transforms.update(self._tm.transforms)
        for ee_frame in ee_frames:
            self._compute_path(base_frame, ee_frame)

        self._compiled = True

    @cython.ccall
    def _compute_path(self, base_frame, ee_frame):
        i = self._tm.nodes.index(ee_frame)
        j = self._tm.nodes.index(base_frame)
        path = []
        k = i
        while k != -9999:
            path.append(self._tm.nodes[k])
            k = self._tm.predecessors[j, k]
        self._cached_shortest_paths[(i, j)] = path

    @cython.ccall
    def _path_transform(self, path):
        A2B = np.eye(4)
        for i in range(len(path) - 1):
            from_f = path[i]
            to_f = path[i + 1]
            if (from_f, to_f) in self._transforms:
                B2C = self._transforms[(from_f, to_f)]
            else:
                B2C = _invert_transform(self._transforms[to_f, from_f])
            A2B = np.dot(B2C, A2B)
        return A2B

    def set_joint_limits(self, joint_name, lower=None, upper=None):
        joint_info = self._tm._joints[joint_name]
        new_limits = [joint_info[4][0], joint_info[4][1]]
        if lower is not None:
            new_limits[0] = lower
        if upper is not None:
            new_limits[1] = upper
        joint_info = (joint_info[0], joint_info[1], joint_info[2], joint_info[3],
                      tuple(new_limits), joint_info[5])
        self._tm._joints[joint_name] = joint_info

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
        return self._tm.get_joint_limits(joint_name)

    def add_transform(self, from_frame, to_frame, A2B):
        """Update existing transform.

        Parameters
        ----------
        from_frame : Hashable
            Name of the frame for which the transformation is added in the
            to_frame coordinate system

        to_frame : Hashable
            Name of the frame in which the transformation is defined

        A2B : array-like, shape (4, 4)
            Homogeneous matrix that represents the transformation from
            'from_frame' to 'to_frame'

        Returns
        -------
        self : TransformManager
            This object for chaining
        """
        if self._compiled:
            assert (from_frame, to_frame) in self._tm.transforms
            self._transforms[(from_frame, to_frame)] = A2B
        else:
            self._tm.add_transform(from_frame, to_frame, A2B)
        return self

    def get_transform(self, from_frame, to_frame):
        """Request a transformation.

        Parameters
        ----------
        from_frame : Hashable
            Name of the frame for which the transformation is requested in the
            to_frame coordinate system

        to_frame : Hashable
            Name of the frame in which the transformation is defined

        Returns
        -------
        A2B : array-like, shape (4, 4)
            Homogeneous matrix that represents the transformation from
            'from_frame' to 'to_frame'
        """
        i = self.nodes.index(from_frame)
        j = self.nodes.index(to_frame)
        A2B = self._path_transform(self._tm._shortest_path(i, j))
        return A2B

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
        self._tm.add_joint(
            joint_name, from_frame, to_frame, child2parent, axis, limits,
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
        if joint_name in self._virtual_joints:
            callback = self._virtual_joints[joint_name]
            actual_joint_states = callback(value)
            for actual_joint_name, actual_value in actual_joint_states.items():
                self.set_joint(actual_joint_name, actual_value)
            return

        from_frame, to_frame, child2parent, axis, limits, joint_type = self._tm._joints[joint_name]
        value = min(max(value, limits[0]), limits[1])
        if joint_type == "revolute":
            joint2A = _fast_matrix_from_axis_angle(axis, value)
        else:
            joint2A = np.eye(4)
            joint2A[:3, 3] = value * axis
        self.add_transform(from_frame, to_frame, child2parent.dot(joint2A))

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
        self._tm.load_urdf(urdf_xml, mesh_path, package_dir)

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
        return self._path_transform(self._cached_shortest_paths[ee_index, base_index])

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
        self._virtual_joints[joint_name] = callback
        self._tm._joints[joint_name] = callback.make_virtual_joint(
            joint_name, self)

    def _whitelisted_nodes(self, whitelist):
        """Get whitelisted nodes.

        Parameters
        ----------
        whitelist : list or None
            Whitelist of frames

        Returns
        -------
        nodes : set
            Existing whitelisted nodes

        Raises
        ------
        KeyError
            Will be raised if an unknown node is in the whitelist.
        """
        nodes = set(self.nodes)
        if whitelist is not None:
            whitelist = set(whitelist)
            nodes = nodes.intersection(whitelist)
            nonwhitlisted_nodes = whitelist.difference(nodes)
            if nonwhitlisted_nodes:
                raise KeyError("Whitelist contains unknown nodes: '%s'"
                               % nonwhitlisted_nodes)
        return nodes


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cpdef np.ndarray[double, ndim=2] _fast_matrix_from_axis_angle(np.ndarray[double, ndim=1] axis, double angle):
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


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cpdef np.ndarray[double, ndim=2] _invert_transform(np.ndarray[double, ndim=2] A2B):
    """Invert transform.

    Parameters
    ----------
    A2B : array-like, shape (4, 4)
        Transform from frame A to frame B

    Returns
    -------
    B2A : array-like, shape (4, 4)
        Transform from frame B to frame A
    """
    B2A = np.empty((4, 4))
    RT = A2B[:3, :3].T
    B2A[:3, :3] = RT
    B2A[:3, 3] = -np.dot(RT, A2B[:3, 3])
    B2A[3, :3] = 0.0
    B2A[3, 3] = 1.0
    return B2A
