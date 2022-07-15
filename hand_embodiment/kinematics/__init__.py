"""Robot kinematics.

Forward and inverse kinematics for robotic hands.
"""
import numpy as np
import math
import numba
from .urdf import FastUrdfTransformManager
from scipy.optimize import minimize


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
    """
    def __init__(self, urdf, mesh_path=None, package_dir=None):
        self.tm = FastUrdfTransformManager()
        self.tm.load_urdf(urdf, mesh_path=mesh_path, package_dir=package_dir)

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
        self.tm.compile(joint_names, base_frame, [ee_frame])

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

        self.joint_limits = np.array([
            self.tm.get_joint_limits(jn) for jn in self.joint_names])
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
        self.tm.compile(joint_names, base_frame, ee_frames)

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
        if bounds is None:
            bounds = self.joint_limits
        res = minimize(
            self.ee_pos_error, initial_joint_angles,
            (desired_positions,), method="SLSQP", bounds=bounds)

        if self.verbose >= 2:
            print("Error: %g" % res["fun"])
        if return_error:
            return res["x"], res["fun"]
        else:
            return res["x"]


@numba.jit(nopython=True, cache=True)
def pose_dist(ee2base_desired, ee2base_actual, orientation_weight, position_weight):
    ee_actual2ee_desired = np.linalg.inv(ee2base_actual).dot(ee2base_desired)
    trace = ee_actual2ee_desired[0, 0] + ee_actual2ee_desired[1, 1] + ee_actual2ee_desired[2, 2]
    angle = math.acos(min((trace - 1.0) / 2.0, 1.0))
    orientation_error = min(angle, 2.0 * math.pi - angle)
    position_error = math.sqrt(ee_actual2ee_desired[0, 3] ** 2 + ee_actual2ee_desired[1, 3] ** 2 + ee_actual2ee_desired[2, 3] ** 2)
    return orientation_weight * orientation_error + position_weight * position_error
