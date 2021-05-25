import time

import numpy as np
from mocap.mano import HandState, hand_vertices, apply_shape_parameters
from pytransform3d import transformations as pt, rotations as pr, visualizer as pv
from scipy.optimize import minimize


# TODO this probably has to be redefined and we have to make sure that this
#      is the same for all tests
MANO2HAND_MARKERS = pt.invert_transform(pt.transform_from(
    R=pr.active_matrix_from_intrinsic_euler_xyz(np.deg2rad([-5, 97, 0])),
    p=np.array([0.0, -0.03, 0.065])))


MANO_CONFIG = {
    "pose_parameters_per_finger":
        {
            "thumb": np.arange(39, 48),
            "index": np.arange(3, 12),
            "middle": np.arange(12, 21),
            "ring": np.arange(30, 39),
            "little": np.arange(21, 30),
        },
    "vertex_index_per_finger":
        {
            "thumb": 744,  # 729#727
            "index": 320,
            "middle": 445,
            "little": 673,
            "ring": 554,  # improvement possible...
        },
    "joint_indices_per_finger":
        {
            "thumb": (13, 14, 15),
            "index": (1, 2, 3),
            "middle": (4, 5, 6),
            "ring": (10, 11, 12),
            "little": (7, 8, 9)
        }
}


def make_finger_kinematics(hand_state, finger_name, action_weight=0.2):
    pppf = MANO_CONFIG["pose_parameters_per_finger"]
    vipf = MANO_CONFIG["vertex_index_per_finger"]
    jipf = MANO_CONFIG["joint_indices_per_finger"]
    return ManoFingerKinematics(
        hand_state, pppf[finger_name], vipf[finger_name], jipf[finger_name],
        action_weight)


class MarkerBasedRecordMapping:
    """Estimates pose of hand and finger configuration based on markers.

    We estimate the pose parameters of a MANO hand model from a marker-based
    motion capture system such as the Qualisys system.

    Parameters
    ----------
    left : bool, optional (default: False)
        Left hand. Right hand otherwise.

    action_weight : float, optional (default: 0.02)
        Default weight of action penalty in error function for fingers.

    mano2hand_markers : array-like, shape (4, 4)
        Transform from MANO model to hand markers.

    shape_parameters : array-like, shape (10,)
        Shape parameters for MANO hand.

    verbose : int, optional (default: 0)
        Verbosity level

    Attributes
    ----------
    hand_state_ : mocap.mano.HandState
        MANO hand state. This state will be updated by the record mapping
        and should be used to perform a subsequent embodiment mapping based
        on the current state.

    mano_finger_kinematics_ : dict (str to ManoFingerKinematics)
        Maps finger names to their kinematic chain in the MANO model.

    mano2world_ : array-like, shape (4, 4)
        MANO base pose in world frame.
    """
    def __init__(self, left=False, action_weight=0.02, mano2hand_markers=None,
                 shape_parameters=None, verbose=0):
        self.hand_state_ = HandState(left=left)
        if shape_parameters is not None:
            self.hand_state_.pose_parameters["J"], self.hand_state_.pose_parameters["v_template"] = \
                apply_shape_parameters(betas=shape_parameters, **self.hand_state_.shape_parameters)
        self.verbose = verbose
        self.mano_finger_kinematics_ = {
            "thumb": make_finger_kinematics(
                self.hand_state_, "thumb", action_weight),
            "index": make_finger_kinematics(
                self.hand_state_, "index", action_weight),
            "middle": make_finger_kinematics(
                self.hand_state_, "middle", action_weight),
            "ring": make_finger_kinematics(
                self.hand_state_, "ring", action_weight),
        }

        if mano2hand_markers is None:
            self.mano2hand_markers = MANO2HAND_MARKERS
        else:
            self.mano2hand_markers = mano2hand_markers
        self.current_hand_markers2world = np.eye(4)
        self.mano2world_ = pt.concat(
            self.mano2hand_markers, self.current_hand_markers2world)
        self.finger_markers_in_mano = {
            finger_name: np.eye(4)
            for finger_name in self.mano_finger_kinematics_.keys()}

    def estimate(self, hand_markers, finger_markers):
        """Estimate hand state from positions of hand markers and finger markers.

        Parameters
        ----------
        hand_markers : list
            Markers on hand in order 'hand_top', 'hand_left', 'hand_right'.

        finger_markers : dict (str to array-like)
            Positions of markers on fingers.
        """
        self.current_hand_markers2world = estimate_hand_pose(*hand_markers)
        self.mano2world_ = pt.concat(
            self.mano2hand_markers, self.current_hand_markers2world)

        for finger_name in finger_markers.keys():
            self.finger_markers_in_mano[finger_name] = pt.invert_transform(
                self.mano2world_).dot(
                pt.vector_to_point(finger_markers[finger_name]))[:3]

        if self.verbose:
            start = time.time()

        for finger_name in finger_markers.keys():
            fe = self.mano_finger_kinematics_[finger_name]
            finger_pose = fe.inverse(self.finger_markers_in_mano[finger_name])
            self.hand_state_.pose[fe.finger_pose_param_indices] = finger_pose

        """# joblib parallelization, not faster because of overhead for data transfer
        import joblib
        def estimate_finger_pose(finger_estimator, measurement):
            finger_pose = finger_estimator.estimate(measurement)
            return finger_estimator.finger_pose_param_indices, finger_pose

        results = joblib.Parallel(n_jobs=-1)(
            joblib.delayed(estimate_finger_pose)(self.finger_estimators[finger_name],
                                                 self.finger_markers_in_mano[finger_name])
            for finger_name in self.finger_estimators.keys())
        for pose_indices, pose in results:
            self.hand_state.pose[pose_indices] = pose
        #"""

        if self.verbose:
            stop = time.time()
            duration = stop - start
            print(f"[{type(self).__name__}] Time for optimization: "
                  f"{duration:.4f} s")

        self.hand_state_.recompute_mesh(self.mano2world_)


def estimate_hand_pose(hand_top, hand_left, hand_right):
    """Estimate pose of the hand from markers on the back of the hand."""
    hand_middle = 0.5 * (hand_left + hand_right)
    hand_pose = np.eye(4)
    middle2top = hand_top - hand_middle
    middle2right = hand_right - hand_middle
    hand_pose[:3, :3] = pr.matrix_from_two_vectors(
        middle2top, middle2right).dot(
        pr.active_matrix_from_intrinsic_euler_xyz([0, 0.5 * np.pi, -0.5 * np.pi]))
    hand_pose[:3, 3] = hand_middle
    return hand_pose


class ManoFingerKinematics:
    """Estimates the state of a finger.

    Parameters
    ----------
    hand_state : HandState
        State of the hand mesh

    finger_pose_param_indices : array, shape (n_finger_joints * 3,)
        Indices of pose parameters of this finger

    finger_vertex_index : int
        Index of the vertex of which we will optimize the position

    finger_joint_indices : array, shape (n_finger_joints,)
        Indices of joints that correspond to this finger

    action_weight : float, optional (default: 0.02)
        Default weight of action penalty in error function for fingers.
    """
    def __init__(self, hand_state, finger_pose_param_indices,
                 finger_vertex_index, finger_joint_indices, action_weight):
        self.finger_pose_param_indices = finger_pose_param_indices
        self.finger_vertex_index = finger_vertex_index
        self.finger_joint_indices = np.asarray([0] + list(finger_joint_indices), dtype=int)

        self._search_similar_vertices(finger_pose_param_indices, hand_state)

        self.finger_pose_params, self.finger_opt_vertex_index = \
            self.reduce_pose_parameters(hand_state)
        self.finger_error = FingerError(self.forward, action_weight)

        self.current_pose = np.zeros_like(
            self.finger_pose_param_indices, dtype=float)

        self._optimizer_pose = np.zeros(len(self.current_pose) + 3)

        self.last_forward_result = None

    def _search_similar_vertices(self, finger_pose_param_indices, hand_state):
        # TODO mapping to indices of weights might be correct by accident
        # search for vertices that are influenced by the same pose parameters
        self.finger_vertex_indices = np.unique(np.nonzero(
            hand_state.pose_parameters["weights"][:, np.unique(finger_pose_param_indices // 3)])[0])
        # extract those with a close index
        self.finger_vertex_indices = self.finger_vertex_indices[
            abs(self.finger_vertex_indices - self.finger_vertex_index) < 2]

    def reduce_pose_parameters(self, hand_state):  # TODO we should introduce our own vertex at marker's position
        finger_opt_vertex_index = np.where(self.finger_vertex_indices == self.finger_vertex_index)[0][0]
        pose_dir_joint_indices = np.hstack([np.arange(i, i + 9) for i in self.finger_joint_indices[1:]]).astype(int)
        pose_params = {
            "J": hand_state.pose_parameters["J"][self.finger_joint_indices],
            "weights": hand_state.pose_parameters["weights"][self.finger_vertex_indices][:, self.finger_joint_indices],
            "kintree_table": hand_state.pose_parameters["kintree_table"][:, self.finger_joint_indices],  # TODO maybe this does not work in general
            "v_template": hand_state.pose_parameters["v_template"][self.finger_vertex_indices],
            "posedirs": hand_state.pose_parameters["posedirs"][self.finger_vertex_indices][:, :, pose_dir_joint_indices]
        }
        return pose_params, finger_opt_vertex_index

    def forward(self, pose=None, return_cached_result=False):
        """Compute position at the tip of the finger for given joint parameters."""
        if return_cached_result:
            assert self.last_forward_result is not None
            return self.last_forward_result

        self._optimizer_pose[3:] = pose
        vertices = hand_vertices(pose=self._optimizer_pose, **self.finger_pose_params)
        self.last_forward_result = vertices[self.finger_opt_vertex_index]
        return self.last_forward_result

    def inverse(self, position):
        """Estimate finger joint parameters from position."""
        res = minimize(self.finger_error, self.current_pose, args=(position,), method="SLSQP")
        self.current_pose[:] = res["x"]
        return self.current_pose


class FingerError:
    """Compute error function for finger.

    Parameters
    ----------
    forward_kinematics : callable
        Forward kinematics

    action_weight : float, optional (default: 0.02)
        Default weight of action penalty in error function for fingers.
    """
    def __init__(self, forward_kinematics, action_weight):
        self.forward_kinematics = forward_kinematics
        self.action_weight = action_weight

    def __call__(self, finger_pose, desired_finger_pos):
        tip_position = self.forward_kinematics(finger_pose)
        return (np.linalg.norm(desired_finger_pos - tip_position) +
                self.action_weight * np.linalg.norm(finger_pose))


class ManoHand(pv.Artist):
    """Representation of hand mesh as artist for 3D visualization in Open3D."""
    def __init__(self, hse):
        self.hse = hse

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
        return [self.hse.hand_state_.hand_mesh]
