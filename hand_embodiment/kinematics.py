"""Robot kinematics.

Forward and inverse kinematics for robotic hands.
"""
import math
from typing import NamedTuple, List, Optional, Callable

import os
os.environ.setdefault("XLA_FLAGS", "--xla_gpu_enable_command_buffer=")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("JAX_COMPILATION_CACHE_DIR",
                      os.path.expanduser("~/.cache/hand_embodiment/jax_cache"))
import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from pytransform3d import urdf
from scipy.optimize import minimize


class FastUrdfTransformManager(urdf.UrdfTransformManager):
    """Fast transformation manager that can load URDF files.

    This version has efficient numba-accelerated code to update joints.
    """
    def __init__(self):
        super(FastUrdfTransformManager, self).__init__(check=False)
        self.virtual_joints = {}

    def load_urdf(self, urdf_xml, mesh_path=None, package_dir=None, scale=1.0):
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

        scale : float, optional (default: 1)
            Scaling factor.
        """
        robot_name, links, joints = urdf.parse_urdf(
            urdf_xml, mesh_path, package_dir, self.strict_check)
        self._scale_urdf(links, joints, scale)
        urdf.initialize_urdf_transform_manager(self, robot_name, links, joints)

    def _scale_urdf(self, links, joints, scale):
        for link in links:
            for transform in link.transforms:
                transform[2][:3, 3] *= scale
            for visual in link.visuals:
                self._scale_object(visual, scale)
            for collision_object in link.collision_objects:
                self._scale_object(collision_object, scale)
        for joint in joints:
            joint.child2parent[:3, 3] *= scale

    def _scale_object(self, visual, scale):
        if isinstance(visual, urdf.Sphere):
            visual.radius *= scale
        elif isinstance(visual, urdf.Cylinder):
            visual.radius *= scale
            visual.length *= scale
        elif isinstance(visual, urdf.Box):
            visual.size *= scale
        else:
            assert isinstance(visual, urdf.Mesh)
            visual.scale *= scale

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


def _fast_matrix_from_axis_angle(axis, angle):
    """Compute transformation matrix from axis-angle."""
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


class Kinematics:
    """Robot kinematics.

    Parameters
    ----------
    urdf : str
        URDF description of a robot

    mesh_path : str, optional (default: None)
        Path in which we search for meshes that are defined in the URDF.
        Meshes will be ignored if it is set to None.

    package_dir : str, optional (default: None)
        Path to corresponding ROS package

    scale : float, optional (default: 1)
        Scaling factor.
    """
    def __init__(self, urdf, mesh_path=None, package_dir=None, scale=1.0):
        self.tm = FastUrdfTransformManager()
        self.tm.load_urdf(
            urdf, mesh_path=mesh_path, package_dir=package_dir, scale=scale)

    def create_chain(self, joint_names, base_frame, ee_frame, verbose=0):
        """Create kinematic chain.

        Parameters
        ----------
        joint_names : list
            Names of joints that should be used

        base_frame : str
            Name of the base link

        ee_frame : str
            Name of the end-effector link

        verbose : int, optional (default: 0)
            Verbosity level

        Returns
        -------
        chain : Chain
            Kinematic chain
        """
        return Chain(self.tm, joint_names, base_frame, ee_frame, verbose)

    def create_multi_chain(self, joint_names, base_frame, ee_frames, verbose=0):
        """Create kinematic chain with multiple tips.

        Parameters
        ----------
        joint_names : list
            Names of joints that should be used

        base_frame : str
            Name of the base link

        ee_frames : list of str
            Name of the end-effector links

        verbose : int, optional (default: 0)
            Verbosity level

        Returns
        -------
        chain : MultiChain
            Kinematic chain
        """
        return MultiChain(self.tm, joint_names, base_frame, ee_frames, verbose)


class Chain:
    """Kinematic chain.

    Parameters
    ----------
    tm : FastUrdfTransformManager
        Transformation manager

    joint_names : list
        Names of joints that should be used

    base_frame : str
        Name of the base link

    ee_frame : str
        Name of the end-effector link

    verbose : int, optional (default: 0)
        Verbosity level
    """
    def __init__(self, tm, joint_names, base_frame, ee_frame, verbose=0):
        self.tm = tm
        self.joint_names = joint_names
        self.base_frame = base_frame
        self.ee_frame = ee_frame
        self.verbose = verbose

        self.joint_limits = np.array([self.tm._joints[jn][4] for jn in self.joint_names])
        for i in range(len(self.joint_limits)):
            if np.isinf(self.joint_limits[i, 0]):
                self.joint_limits[i, 0] = -math.pi
            if np.isinf(self.joint_limits[i, 1]):
                self.joint_limits[i, 1] = math.pi

        self.n_joints = len(self.joint_names)
        assert len(self.joint_limits) == self.n_joints

        self.ee_index = self.tm.nodes.index(ee_frame)
        self.base_index = self.tm.nodes.index(base_frame)

    def forward(self, joint_angles):
        """Forward kinematics.

        Parameters
        ----------
        joint_angles : array, shape (n_joints,)
            Joint angles

        Returns
        -------
        ee2base : array, shape (4, 4)
            Transformation from end-effector to base frame
        """
        for i in range(self.n_joints):
            self.tm.set_joint(self.joint_names[i], joint_angles[i])
        return self.tm.get_ee2base(self.ee_index, self.base_index)

    def ee_pos_error(self, joint_angles, desired_pos):
        """Compute position error.

        Parameters
        ----------
        joint_angles : array-like, shape (n_joints,)
            Actual joint angles for which we compute forward kinematics.

        desired_pos : array-like, shape (3,)
            Desired position.

        Returns
        -------
        pos_error : float
            Position error.
        """
        return np.linalg.norm(desired_pos - self.forward(joint_angles)[:3, 3])

    def ee_pose_error(self, joint_angles, desired_pose, orientation_weight=1.0, position_weight=1.0):
        """Compute pose error.

        Parameters
        ----------
        joint_angles : array-like, shape (n_joints,)
            Actual joint angles for which we compute forward kinematics.

        desired_pose : array-like, shape (4, 4)
            Desired pose.

        orientation_weight : float, optional (default: 1.0)
            Should be between 0.0 and 1.0 and represent the weighting for
            minimizing the orientation error.

        position_weight : float, optional (default: 1.0)
            Should be between 0.0 and 1.0 and represent the weighting for
            minimizing the position error.

        Returns
        -------
        pose_error : float
            Weighted error between actual pose and desired pose.
        """
        return pose_dist(desired_pose, self.forward(joint_angles),
                         orientation_weight, position_weight)

    def inverse_position(self, desired_pos, initial_joint_angles, return_error=False, bounds=None):
        """Inverse kinematics.

        Parameters
        ----------
        desired_pos : array, shape (3,)
            Desired position of end-effector in base frame

        initial_joint_angles : array, shape (n_joints,)
            Initial guess for joint angles

        return_error : bool, optional (default: False)
            Return error in addition to joint angles

        bounds : array, shape (n_joints, 2), optional (default: joint limits)
            Bounds for joint angle optimization

        Returns
        -------
        joint_angles : array, shape (n_joints,)
            Solution

        error : float, optional
            Pose error
        """
        if bounds is None:
            bounds = self.joint_limits
        res = minimize(self.ee_pos_error, initial_joint_angles, (desired_pos,), method="SLSQP", bounds=bounds)

        if self.verbose >= 2:
            print("Error: %g" % res["fun"])
        if return_error:
            return res["x"], res["fun"]
        else:
            return res["x"]

    def inverse(self, desired_pose, initial_joint_angles, return_error=False, bounds=None):
        """Inverse kinematics.

        Parameters
        ----------
        desired_pose : array, shape (4, 4)
            Desired transformation from end-effector to base frame

        initial_joint_angles : array, shape (n_joints,)
            Initial guess for joint angles

        return_error : bool, optional (default: False)
            Return error in addition to joint angles

        bounds : array, shape (n_joints, 2), optional (default: joint limits)
            Bounds for joint angle optimization

        Returns
        -------
        joint_angles : array, shape (n_joints,)
            Solution

        error : float, optional
            Pose error
        """
        if bounds is None:
            bounds = self.joint_limits
        res = minimize(
            self.ee_pose_error, initial_joint_angles, (desired_pose,),
            method="SLSQP", bounds=bounds)

        if self.verbose >= 2:
            print("Error: %g" % res["fun"])
        if return_error:
            return res["x"], res["fun"]
        else:
            return res["x"]

    def inverse_with_random_restarts(
            self, desired_pose, n_restarts=10, tolerance=1e-3,
            random_state=None):
        """Compute inverse kinematics with multiple random restarts.

        Parameters
        ----------
        desired_pose : array-like, shape (4, 4)
            Desired pose.

        n_restarts : int, optional (default: 10)
            Maximum number of allowed restarts.

        tolerance : float, optional (default: 1e-3)
            Required tolerance to abort.

        random_state : np.random.RandomState, optional (default: np.random)
            Random state.

        Returns
        -------
        joint_angles : array, shape (n_joints,)
            Solution
        """
        if random_state is None:
            random_state = np.random
        assert n_restarts >= 1
        Q = []
        errors = []
        for _ in range(n_restarts):
            q, error = self.inverse(
                desired_pose, self._sample_joints_uniform(random_state),
                return_error=True)
            Q.append(q)
            errors.append(error)
            if error <= tolerance:
                break
        if self.verbose:
            print(np.round(errors, 4))
        return Q[np.argmin(errors)]

    def local_inverse_with_random_restarts(
            self, desired_pose, joint_angles, interval, n_restarts=10,
            tolerance=1e-3, random_state=None):
        """Compute inverse kinematics with multiple random restarts.

        Parameters
        ----------
        desired_pose : array-like, shape (4, 4)
            Desired pose.

        joint_angles : array-like, shape (n_joints,)
            Initial guess for joint angles.

        interval : float
            We will search for a solution within the range
            [joint_angles - interval, joint_angles + interval].

        n_restarts : int, optional (default: 10)
            Maximum number of allowed restarts.

        tolerance : float, optional (default: 1e-3)
            Required tolerance to abort.

        random_state : np.random.RandomState, optional (default: np.random)
            Random state.

        Returns
        -------
        joint_angles : array, shape (n_joints,)
            Solution
        """
        if random_state is None:
            random_state = np.random
        assert n_restarts >= 1
        Q = []
        errors = []
        bounds = np.empty((self.n_joints, 2))
        bounds[:, 0] = joint_angles - interval
        bounds[:, 1] = joint_angles + interval
        q = joint_angles  # start with previous state
        for _ in range(n_restarts):
            q, error = self.inverse(desired_pose, q, return_error=True)
            Q.append(q)
            errors.append(error)
            if error <= tolerance:
                break
            q = self._sample_joints_uniform(random_state, bounds=bounds)
        return Q[np.argmin(errors)]

    def _sample_joints_uniform(self, random_state, bounds=None):
        if bounds is None:
            bounds = self.joint_limits
        return random_state.rand(len(bounds)) * (bounds[:, 1] - bounds[:, 0]) + bounds[:, 0]

    def forward_trajectory(self, Q):
        H = np.empty((len(Q), 4, 4))
        for t in range(len(Q)):
            H[t] = self.forward(Q[t])
        return H

    def inverse_trajectory(
            self, H, initial_joint_angles=None, interval=0.1 * math.pi,
            random_restarts=True, random_state=None):
        """Compute inverse kinematics for a trajectory.

        Parameters
        ----------
        H : array-like, shape (n_steps, 4, 4)
            Desired end-effector poses.

        initial_joint_angles : array-like, shape (n_joints,), optional (default: None)
            Initial guess for joint angles.

        interval : float
            We will search for a solution within the range
            [joint_angles - interval, joint_angles + interval] in each step.

        random_restarts : bool, optional (default: True)
            Allow random restarts if no solution is found.

        random_state : np.random.RandomState, optional (default: np.random)
            Random state.

        Returns
        -------
        Q : array, shape (n_steps, n_joints)
            Solution
        """
        Q = np.empty((len(H), len(self.joint_names)))

        if initial_joint_angles is not None:
            Q[0] = self.inverse(H[0], initial_joint_angles)
        else:
            Q[0] = self.inverse_with_random_restarts(
                H[0], random_state=random_state)

        for t in range(1, len(H)):
            if self.verbose >= 2:
                print("Step: %d" % (t + 1))
            if random_restarts:
                Q[t] = self.local_inverse_with_random_restarts(
                    H[t], Q[t - 1], interval, random_state=random_state)
            else:
                bounds = np.empty((self.n_joints, 2))
                bounds[:, 0] = Q[t - 1] - interval
                bounds[:, 1] = Q[t - 1] + interval
                Q[t] = self.inverse(H[t], Q[t - 1], False, bounds)
        return Q


class MultiChain:
    """Kinematic chain with multiple end effectors.

    Parameters
    ----------
    tm : FastUrdfTransformManager
        Transformation manager

    joint_names : list
        Names of joints that should be used

    base_frame : str
        Name of the base link

    ee_frames : list of str
        Name of the end-effector links

    verbose : int, optional (default: 0)
        Verbosity level
    """
    def __init__(self, tm, joint_names, base_frame, ee_frames, verbose=0):
        self.tm = tm
        self.joint_names = joint_names
        self.base_frame = base_frame
        self.ee_frames = ee_frames
        self.verbose = verbose

        self.joint_limits = np.array([self.tm._joints[jn][4] for jn in self.joint_names])
        for i in range(len(self.joint_limits)):
            if np.isinf(self.joint_limits[i, 0]):
                self.joint_limits[i, 0] = -math.pi
            if np.isinf(self.joint_limits[i, 1]):
                self.joint_limits[i, 1] = math.pi

        self.n_joints = len(self.joint_names)
        assert len(self.joint_limits) == self.n_joints

        self.ee_indices = [self.tm.nodes.index(ee_frame)
                           for ee_frame in self.ee_frames]
        self.base_index = self.tm.nodes.index(base_frame)

        self._jax_params = _build_jax_fk_params(
            tm, joint_names, base_frame, ee_frames, self.joint_limits)

        jax_params_ = self._jax_params
        lower_ = jnp.array(self.joint_limits[:, 0], dtype=jnp.float64)
        upper_ = jnp.array(self.joint_limits[:, 1], dtype=jnp.float64)
        n_joints_ = self.n_joints

        def _fk(q):
            return _jax_multi_chain_forward_all(jax_params_, q)

        # LM schedule: λ from 0.1 (gradient-descent-like, stable far from solution)
        # down to 1e-4 (Gauss-Newton-like, fast near solution).
        # Scale relative to typical diag(J^T J) ~ 0.01 for finger chains in metres.
        _lm_lambdas = jnp.exp(
            jnp.linspace(jnp.log(0.1), jnp.log(1e-4), 20))

        def _ik_fn(q0, des):
            def step(q, lam):
                r = (des - _fk(q)).ravel()
                J = jax.jacfwd(_fk)(q).reshape(-1, n_joints_)
                A = J.T @ J + lam * jnp.eye(n_joints_)
                delta = jnp.linalg.solve(A, J.T @ r)
                return jnp.clip(q + delta, lower_, upper_), None
            q_final, _ = jax.lax.scan(step, q0, _lm_lambdas)
            error = jnp.sum(jnp.linalg.norm(des - _fk(q_final), axis=-1))
            return q_final, error

        self._jit_ik = jax.jit(_ik_fn)

        def _scan_step(q_prev, des_t):
            def step(q, lam):
                r = (des_t - _fk(q)).ravel()
                J = jax.jacfwd(_fk)(q).reshape(-1, n_joints_)
                A = J.T @ J + lam * jnp.eye(n_joints_)
                delta = jnp.linalg.solve(A, J.T @ r)
                return jnp.clip(q + delta, lower_, upper_), None
            q_new, _ = jax.lax.scan(step, q_prev, _lm_lambdas)
            return q_new, q_new

        self._ik_scan_step = jax.jit(_scan_step)

    def forward(self, joint_angles):
        """Forward kinematics.

        Parameters
        ----------
        joint_angles : array, shape (n_joints,)
            Joint angles

        Returns
        -------
        ee2base : array, shape (4, 4)
            Transformation from end-effector to base frame
        """
        for i in range(self.n_joints):
            self.tm.set_joint(self.joint_names[i], joint_angles[i])
        return [self.tm.get_ee2base(ee_index, self.base_index)
                for ee_index in self.ee_indices]

    def ee_pos_error(self, joint_angles, desired_positions):
        """Compute position error.

        Parameters
        ----------
        joint_angles : array-like, shape (n_joints,)
            Actual joint angles for which we compute forward kinematics.

        desired_positions : array-like, shape (n_end_effectors, 3)
            Desired position.

        Returns
        -------
        pos_error : float
            Position error.
        """
        actual_positions = self.forward(joint_angles)
        return np.linalg.norm(
            [desired_pos - actual_pos[:3, 3]
             for desired_pos, actual_pos in zip(
                desired_positions, actual_positions)]).sum()  # TODO why norm().sum()?

    def inverse_position(self, desired_positions, initial_joint_angles, return_error=False, bounds=None):
        """Inverse kinematics.

        Parameters
        ----------
        desired_positions : array, shape (n_ee_frames, 3)
            Desired positions of end-effectors in base frame

        initial_joint_angles : array, shape (n_joints,)
            Initial guess for joint angles

        return_error : bool, optional (default: False)
            Return error in addition to joint angles

        bounds : array, shape (n_joints, 2), optional (default: joint limits)
            Bounds for joint angle optimization

        Returns
        -------
        joint_angles : array, shape (n_joints,)
            Solution

        error : float, optional
            Pose error
        """
        q_np, error = self._jit_ik(
            jnp.array(initial_joint_angles, dtype=jnp.float64),
            jnp.array(desired_positions, dtype=jnp.float64))
        q_np = np.array(q_np)

        if self.verbose >= 2:
            print("Error: %g" % float(error))
        if return_error:
            return q_np, float(error)
        else:
            return q_np


def pose_dist(ee2base_desired, ee2base_actual, orientation_weight, position_weight):
    ee_actual2ee_desired = np.linalg.inv(ee2base_actual).dot(ee2base_desired)
    trace = ee_actual2ee_desired[0, 0] + ee_actual2ee_desired[1, 1] + ee_actual2ee_desired[2, 2]
    angle = math.acos(min(max((trace - 1.0) / 2.0, -1.0), 1.0))
    orientation_error = min(angle, 2.0 * math.pi - angle)
    position_error = math.sqrt(ee_actual2ee_desired[0, 3] ** 2 + ee_actual2ee_desired[1, 3] ** 2 + ee_actual2ee_desired[2, 3] ** 2)
    return orientation_weight * orientation_error + position_weight * position_error


class JaxFKParams(NamedTuple):
    """Static FK parameters for a kinematic chain, extracted at init time."""
    all_path_entries: List  # one list of entries per EE frame
    joint_limits: object    # jnp array (n_joints, 2)


def _build_jax_fk_params(tm, joint_names, base_frame, ee_frames,
                         joint_limits_np=None):
    """Extract static FK parameters from pytransform3d for JAX forward kinematics.

    Walks the kinematic path for each EE frame and classifies each step as a
    variable joint (in chain) or a fixed transform (not in chain, evaluated at
    θ=0). Virtual joints are expanded to their real joints with JAX-compatible
    angle extractors.

    Parameters
    ----------
    tm : FastUrdfTransformManager
    joint_names : list of str
    base_frame : str
    ee_frames : list of str

    Returns
    -------
    JaxFKParams
    """
    joints_by_frames = {(v[0], v[1]): k for k, v in tm._joints.items()}
    base_idx = tm.nodes.index(base_frame)

    # Build mapping: real_joint_name → (chain_idx, jax_angle_fn)
    real_to_chain = {}
    for i, jn in enumerate(joint_names):
        if jn in tm.virtual_joints:
            cb = tm.virtual_joints[jn]
            real_joints = cb(0.0)
            for real_jn in real_joints:
                if hasattr(cb, 'jax_angle_for'):
                    angle_fn = cb.jax_angle_for(real_jn)
                else:
                    # Fallback: use pure_callback (not differentiable)
                    def angle_fn(v, _cb=cb, _rjn=real_jn):
                        import jax
                        return jax.pure_callback(
                            lambda v_: np.float32(_cb(float(v_))[_rjn]),
                            jnp.float32, v)
                real_to_chain[real_jn] = (i, angle_fn)
        else:
            real_to_chain[jn] = (i, lambda v: v)

    def build_entries(ee_frame):
        path = tm._shortest_path(tm.nodes.index(ee_frame), base_idx)
        entries = []
        for a, b in zip(path[:-1], path[1:]):
            if (a, b) in joints_by_frames:
                jn = joints_by_frames[(a, b)]
                _, _, child2parent, axis, _, _ = tm._joints[jn]
                if jn in real_to_chain:
                    chain_idx, angle_fn = real_to_chain[jn]
                    entries.append(('joint',
                                    jnp.array(child2parent, dtype=jnp.float64),
                                    np.array(axis, dtype=np.float64),
                                    chain_idx, angle_fn))
                else:
                    entries.append(('fixed', jnp.array(child2parent, dtype=jnp.float64)))
            elif (a, b) in tm.transforms:
                entries.append(('fixed', jnp.array(tm.transforms[(a, b)], dtype=jnp.float64)))
            elif (b, a) in tm.transforms:
                entries.append(('fixed', jnp.array(
                    np.linalg.inv(tm.transforms[(b, a)]), dtype=jnp.float64)))
            else:
                raise RuntimeError(f"No transform found for path step {a!r} → {b!r}")
        return entries

    all_path_entries = [build_entries(ee) for ee in ee_frames]
    if joint_limits_np is None:
        joint_limits_np = np.array([tm._joints[jn][4] for jn in joint_names],
                                   dtype=np.float64)
    joint_limits = jnp.array(joint_limits_np, dtype=jnp.float64)

    return JaxFKParams(all_path_entries=all_path_entries, joint_limits=joint_limits)


def _rot4_jax(axis, angle):
    """4×4 rotation matrix from axis (3-vector, unit) and angle (scalar)."""
    ux, uy, uz = axis
    c = jnp.cos(angle)
    s = jnp.sin(angle)
    ci = 1.0 - c
    return jnp.array([
        [ci*ux*ux + c,      ci*ux*uy - uz*s,  ci*ux*uz + uy*s,  0.0],
        [ci*uy*ux + uz*s,   ci*uy*uy + c,     ci*uy*uz - ux*s,  0.0],
        [ci*uz*ux - uy*s,   ci*uz*uy + ux*s,  ci*uz*uz + c,     0.0],
        [0.0,               0.0,               0.0,               1.0],
    ])


def _jax_multi_chain_forward(jax_params, joint_angles, ee_idx=0):
    """Pure JAX FK: returns position of one EE frame in base frame.

    Parameters
    ----------
    jax_params : JaxFKParams
    joint_angles : jnp array, shape (n_joints,)
    ee_idx : int
        Index into jax_params.all_path_entries selecting which EE frame.

    Returns
    -------
    pos : jnp array, shape (3,)
    """
    T = jnp.eye(4, dtype=jnp.float64)
    for entry in jax_params.all_path_entries[ee_idx]:
        if entry[0] == 'joint':
            _, child2parent, axis, chain_idx, angle_fn = entry
            theta = angle_fn(joint_angles[chain_idx])
            R4 = _rot4_jax(axis, theta)
            step = child2parent @ R4
        else:
            step = entry[1]
        T = step @ T
    return T[:3, 3]


def _jax_multi_chain_forward_all(jax_params, joint_angles):
    """Returns positions of ALL EE frames stacked as (n_ee, 3)."""
    return jnp.stack([
        _jax_multi_chain_forward(jax_params, joint_angles, i)
        for i in range(len(jax_params.all_path_entries))])
