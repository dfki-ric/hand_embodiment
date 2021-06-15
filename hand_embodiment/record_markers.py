import time

import numpy as np
from mocap.mano import HandState, hand_vertices, apply_shape_parameters
from pytransform3d import transformations as pt, rotations as pr
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
    "vertex_indices_per_finger":
        {
            "thumb": [724, 706],
            "index": [314],
            "middle": [426],
            "little": [651],
            "ring": [534],
        },
    "joint_indices_per_finger":
        {
            "thumb": (13, 14, 15),
            "index": (1, 2, 3),
            "middle": (4, 5, 6),
            "ring": (10, 11, 12),
            "little": (7, 8, 9)
        },
    "tip_vertex_offsets_per_finger":
        {
            "thumb": [np.array([0.006, 0.006, 0.003]),
                      np.array([0.006, 0.006, 0.003])],
            "index": [np.array([0, 0.006, 0])],
            "middle": [np.array([0, 0.006, 0])],
            "ring": [np.array([0, 0.006, 0])],
            "little": [np.array([0, 0.006, 0])]
        },
    "action_weights_per_finger":
        {
            "thumb":     # roll -l/+r -ext/+flex
                np.array([[0.01, 0.01, 0.01,  # + positive
                           0.01, 0.01, 0.01,
                           0.01, 0.05, 0.05],
                          [0.01, 0.01, 0.01,  # - negative
                           0.01, 0.01, 0.01,
                           0.01, 0.05, 0.05]]),
            "index":
                np.array([[0.1, 0.001, 0.001,  # close to palm
                           0.1, 0.05, 0.001,  # middle joint
                           0.1, 0.05, 0.001],  # tip joint
                          [0.1, 0.001, 0.001,
                           0.1, 0.05, 0.003,
                           0.1, 0.05, 0.005]]),
            "middle":
                np.array([[0.1, 0.001, 0.001,
                           0.1, 0.01, 0.001,
                           0.1, 0.01, 0.001],
                          [0.1, 0.001, 0.001,
                           0.1, 0.05, 0.005,
                           0.1, 0.05, 0.005]]),
            "ring":
                np.array([[0.1, 0.001, 0.001,
                           0.1, 0.01, 0.001,
                           0.1, 0.01, 0.001],
                          [0.1, 0.001, 0.001,
                           0.1, 0.05, 0.005,
                           0.1, 0.05, 0.005]]),
            "little":
                np.array([[0.1, 0.001, 0.001,
                           0.1, 0.01, 0.001,
                           0.1, 0.01, 0.001],
                          [0.1, 0.001, 0.001,
                           0.1, 0.05, 0.005,
                           0.1, 0.05, 0.005]]),
        }
}


def make_finger_kinematics(hand_state, finger_name):
    return ManoFingerKinematics(
        hand_state,
        MANO_CONFIG["pose_parameters_per_finger"][finger_name],
        MANO_CONFIG["vertex_indices_per_finger"][finger_name],
        MANO_CONFIG["joint_indices_per_finger"][finger_name],
        MANO_CONFIG["action_weights_per_finger"][finger_name],
        MANO_CONFIG["tip_vertex_offsets_per_finger"][finger_name])


class MarkerBasedRecordMapping:
    """Estimates pose of hand and finger configuration based on markers.

    We estimate the pose parameters of a MANO hand model from a marker-based
    motion capture system such as the Qualisys system.

    Parameters
    ----------
    left : bool, optional (default: False)
        Left hand. Right hand otherwise.

    mano2hand_markers : array-like, shape (4, 4)
        Transform from MANO model to hand markers.

    shape_parameters : array-like, shape (10,)
        Shape parameters for MANO hand.

    hand_state : mocap.mano.HandState, optional (default: None)
        If there is already a hand state object, this can be reused for the
        record mapping. Otherwise we will create a new one.

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

    mano2hand_markers_ : array-like, shape (4, 4)
        Transformation from MANO base frame to marker base frame.

    mano2world_ : array-like, shape (4, 4)
        MANO base pose in world frame.
    """
    def __init__(self, left=False, mano2hand_markers=None,
                 shape_parameters=None, hand_state=None, verbose=0):
        if hand_state is None:
            self.hand_state_ = HandState(left=left)
            if shape_parameters is not None:
                self.hand_state_.betas[:] = shape_parameters
                self.hand_state_.pose_parameters["J"], \
                    self.hand_state_.pose_parameters["v_template"] = \
                    apply_shape_parameters(betas=shape_parameters,
                                           **self.hand_state_.shape_parameters)
        else:
            self.hand_state_ = hand_state

        self.verbose = verbose

        self.mano_finger_kinematics_ = {
            "thumb": make_finger_kinematics(self.hand_state_, "thumb"),
            "index": make_finger_kinematics(self.hand_state_, "index"),
            "middle": make_finger_kinematics(self.hand_state_, "middle"),
            "ring": make_finger_kinematics(self.hand_state_, "ring"),
            "little": make_finger_kinematics(self.hand_state_, "little"),
        }

        if mano2hand_markers is None:
            self.mano2hand_markers_ = MANO2HAND_MARKERS
        else:
            self.mano2hand_markers_ = mano2hand_markers
        self.current_hand_markers2world = np.eye(4)
        self.mano2world_ = pt.concat(
            self.mano2hand_markers_, self.current_hand_markers2world)
        self.markers_in_mano = {
            finger_name: None for finger_name in self.mano_finger_kinematics_}

    def reset(self):
        """Reset current joint poses of MANO."""
        for finger_name in self.mano_finger_kinematics_:
            self.mano_finger_kinematics_[finger_name].reset()

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
            self.mano2hand_markers_, self.current_hand_markers2world)

        world2mano = pt.invert_transform(self.mano2world_, check=False)
        for finger_name in finger_markers:
            markers_in_world = np.atleast_2d(finger_markers[finger_name])
            self.markers_in_mano[finger_name] = np.dot(
                pt.vectors_to_points(markers_in_world), world2mano.T)[:, :3]

        if self.verbose:
            start = time.time()

        for finger_name in finger_markers.keys():
            fe = self.mano_finger_kinematics_[finger_name]
            finger_pose = fe.inverse(self.markers_in_mano[finger_name])
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
        State of the hand mesh.

    finger_pose_param_indices : array, shape (n_finger_joints * 3,)
        Indices of pose parameters of this finger.

    finger_vertex_indices : list of int
        Indices of the vertices of which we will optimize the position.

    finger_joint_indices : array, shape (n_finger_joints,)
        Indices of joints that correspond to this finger.

    action_weights : array, shape (2, n_finger_joints * 3)
        Default weight of action penalty in error function for fingers.

    tip_vertex_offsets : list of array
        Offsets of vertex with respect to original vertex in MANO base frame.
    """
    def __init__(self, hand_state, finger_pose_param_indices,
                 finger_vertex_indices, finger_joint_indices, action_weights,
                 tip_vertex_offsets):
        self.finger_pose_param_indices = finger_pose_param_indices
        self.finger_vertex_indices = finger_vertex_indices
        self.finger_joint_indices = np.asarray(
            [0] + list(finger_joint_indices)).astype(dtype=int)
        self.tip_vertex_offsets = tip_vertex_offsets

        self.all_finger_vertex_indices = self._search_similar_vertices(
            finger_pose_param_indices, hand_state)

        self.finger_pose_params, self.finger_opt_vertex_indices = \
            self.reduce_pose_parameters(hand_state)
        self.finger_error = FingerError(self.forward, action_weights)

        self.current_pose = np.zeros_like(
            self.finger_pose_param_indices).astype(dtype=float)

        self._optimizer_pose = np.zeros(len(self.current_pose) + 3)
        self.bounds = np.array([
            [-0.4 * np.pi, 0.4 * np.pi]] * len(self.current_pose))

        self.last_forward_result = None

    def _search_similar_vertices(self, finger_pose_param_indices, hand_state):
        # search for vertices that are influenced by the same pose parameters
        # TODO mapping to indices of weights might be correct by accident
        return np.unique(np.nonzero(
            hand_state.pose_parameters["weights"][
                :, np.unique(finger_pose_param_indices // 3)])[0])

    def reset(self):
        self.current_pose[:] = 0.0

    def reduce_pose_parameters(self, hand_state):
        finger_opt_vertex_indices = []
        for idx in self.finger_vertex_indices:
            match = np.where(self.all_finger_vertex_indices == idx)[0]
            if not match:
                raise ValueError(
                    f"Vertex with index {idx} does not belong to this finger. "
                    f"Possible options: {self.all_finger_vertex_indices}")
            finger_opt_vertex_indices.append(match[0])
        pose_dir_joint_indices = np.hstack([np.arange(i, i + 9) for i in self.finger_joint_indices[1:]]).astype(int)

        v_template = hand_state.pose_parameters["v_template"][self.finger_vertex_indices].copy()
        for i in range(len(v_template)):
            v_template[i] += self.tip_vertex_offsets[i]

        pose_params = {
            "J": hand_state.pose_parameters["J"][self.finger_joint_indices],
            "weights": hand_state.pose_parameters["weights"][self.finger_vertex_indices][:, self.finger_joint_indices],
            "kintree_table": hand_state.pose_parameters["kintree_table"][:, self.finger_joint_indices],  # TODO maybe this does not work in general
            "v_template": v_template,
            "posedirs": hand_state.pose_parameters["posedirs"][self.finger_vertex_indices][:, :, pose_dir_joint_indices]
        }
        return pose_params, finger_opt_vertex_indices

    def forward(self, pose=None, return_cached_result=False):
        """Compute position at the tip of the finger for given joint parameters."""
        if return_cached_result:
            assert self.last_forward_result is not None
            return self.last_forward_result

        self._optimizer_pose[3:] = pose
        self.last_forward_result = hand_vertices(
            pose=self._optimizer_pose, **self.finger_pose_params)
        return self.last_forward_result

    def inverse(self, position):
        """Estimate finger joint parameters from position."""
        res = minimize(self.finger_error, self.current_pose, args=(position,),
                       method="SLSQP", bounds=self.bounds)  # SLSQP, COBYLA
        self.current_pose[:] = res["x"]
        return self.current_pose


class FingerError:
    """Compute error function for finger.

    Parameters
    ----------
    forward_kinematics : callable
        Forward kinematics

    action_weights : array, shape (2, n_joints * 3)
        Weight of action penalty in error function for fingers.
    """
    def __init__(self, forward_kinematics, action_weights):
        self.forward_kinematics = forward_kinematics
        self.action_weights = action_weights

    def __call__(self, finger_pose, desired_finger_pos):
        positions = self.forward_kinematics(finger_pose)
        desired_finger_pos = np.atleast_2d(desired_finger_pos)

        pos_finger_pose = np.maximum(0.0, finger_pose)
        neg_finger_pose = -np.minimum(0.0, finger_pose)

        # squared cost improves result and speed drastically in comparison
        # to non-squared cost
        errors = np.linalg.norm(desired_finger_pos - positions, axis=1) ** 2
        error = np.nansum(errors)

        regularization = (
                np.dot(self.action_weights[0], pos_finger_pose) ** 2
                + np.dot(self.action_weights[1], neg_finger_pose) ** 2)

        return error + regularization
