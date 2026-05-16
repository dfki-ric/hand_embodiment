"""Record mapping for a marker-based motion capture system.

Estimates MANO states from marker positions.
"""
import warnings

import numpy as np
from pytransform3d import transformations as pt, rotations as pr
from scipy.optimize import minimize
from .mano import HandState, hand_vertices, apply_shape_parameters
from .timing import TimeableMixin


# TODO this probably has to be redefined and we have to make sure that this
#      is the same for all tests
MANO2HAND_MARKERS = pt.invert_transform(pt.transform_from(
    R=pr.active_matrix_from_intrinsic_euler_xyz(np.deg2rad([-5, 97, 0])),
    p=np.array([0.0, -0.03, 0.065])))


VERTEX_OFFSET = 0.007  # marker radius: 0.006
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
            "thumb": [724,  # tip
                      706,  # middle
                      ],
            "index": [314,  # tip
                      261,  # middle
                      ],
            "middle": [426,  # tip
                       399,  # middle
                       ],
            "little": [651,  # tip
                       627,  # middle
                       ],
            "ring": [534,  # tip
                     509,  # middle
                     ],
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
            "index": [np.array([0, VERTEX_OFFSET, 0]),
                      np.array([0, VERTEX_OFFSET, 0])],
            "middle": [np.array([0, VERTEX_OFFSET, 0]),
                       np.array([0, VERTEX_OFFSET, 0])],
            "ring": [np.array([0, VERTEX_OFFSET, 0]),
                     np.array([0, VERTEX_OFFSET, 0])],
            "little": [np.array([0, VERTEX_OFFSET, 0]),
                       np.array([0, VERTEX_OFFSET, 0])]
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


def make_finger_kinematics(hand_state, finger_name, mano_config=MANO_CONFIG):
    return ManoFingerKinematics(
        hand_state,
        mano_config["pose_parameters_per_finger"][finger_name],
        mano_config["vertex_indices_per_finger"][finger_name],
        mano_config["joint_indices_per_finger"][finger_name],
        mano_config["action_weights_per_finger"][finger_name],
        mano_config["tip_vertex_offsets_per_finger"][finger_name])


class MarkerBasedRecordMapping(TimeableMixin):
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

    record_mapping_config : dict, optional (default: None)
        Configuration of record mapping.

    use_fingers : tuple of str, optional (default: ('thumb', 'index', 'middle', 'ring', 'little'))
        Fingers for which we compute the record mapping.

    verbose : int, optional (default: 0)
        Verbosity level

    measure_time : bool
        Measure computation time for each frame.

    Attributes
    ----------
    finger_names_ : set of str
        Fingers for which we compute the record mapping.

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
    def __init__(
            self, left=False, mano2hand_markers=None, shape_parameters=None,
            hand_state=None, record_mapping_config=None,
            use_fingers=("thumb", "index", "middle", "ring", "little"),
            verbose=0, measure_time=False):
        super(MarkerBasedRecordMapping, self).__init__(verbose or measure_time)
        self.finger_names_ = set(use_fingers)

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

        if record_mapping_config is None:
            record_mapping_config = MANO_CONFIG

        self.mano_finger_kinematics_ = {
            finger_name: make_finger_kinematics(
                self.hand_state_, finger_name, record_mapping_config)
            for finger_name in self.finger_names_
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

        self._build_vmap_mano_ik()

    def reset(self):
        """Reset current joint poses of MANO."""
        for finger_name in self.mano_finger_kinematics_:
            self.mano_finger_kinematics_[finger_name].reset()

    def _build_vmap_mano_ik(self):
        """Build a vmapped JAX LM inverse for all fingers simultaneously."""
        import jax
        jax.config.update("jax_enable_x64", True)
        import jax.numpy as jnp
        from .mano_jax import build_hand_vertices_fn

        FINGER_ORDER = ["thumb", "index", "middle", "ring", "little"]
        finger_names = [f for f in FINGER_ORDER if f in self.mano_finger_kinematics_]
        n_f = len(finger_names)
        fks = [self.mano_finger_kinematics_[f] for f in finger_names]
        self._vmap_finger_names = finger_names
        self._vmap_fks = fks

        fwd_fn = build_hand_vertices_fn(fks[0].finger_pose_params["kintree_table"])

        J_all = jnp.stack([jnp.array(fk.finger_pose_params["J"]) for fk in fks])
        w_all = jnp.stack([jnp.array(fk.finger_pose_params["weights"]) for fk in fks])
        vt_all = jnp.stack([jnp.array(fk.finger_pose_params["v_template"]) for fk in fks])
        pd_all = jnp.stack([jnp.array(fk.finger_pose_params["posedirs"]) for fk in fks])
        aw_all = jnp.stack([jnp.array(fk.finger_error.action_weights) for fk in fks])

        n_pose = 9
        lower = jnp.full(n_pose, -0.4 * np.pi, dtype=jnp.float64)
        upper = jnp.full(n_pose, 0.4 * np.pi, dtype=jnp.float64)
        lm_lams = jnp.exp(jnp.linspace(jnp.log(0.05), jnp.log(0.001), 15))
        max_step = 0.2
        smooth_w = jnp.float64(2e-3)

        def _inverse_one(pose0, des, des_mask, J_f, w_f, vt_f, pd_f, aw_f):
            def _fwd(p):
                return fwd_fn(J_f, w_f, vt_f, pd_f,
                              jnp.concatenate([jnp.zeros(3), p]))

            def _residual(pose):
                verts = _fwd(pose)
                diff = ((verts - des) * des_mask[:, None]).ravel()
                reg = jnp.concatenate([
                    aw_f[0] * jnp.maximum(0.0, pose),
                    aw_f[1] * jnp.maximum(0.0, -pose)])
                return jnp.concatenate([diff, reg])

            def step(pose, lam):
                r = _residual(pose)
                J = jax.jacfwd(_residual)(pose)
                A = J.T @ J + (lam + smooth_w) * jnp.eye(n_pose, dtype=jnp.float64)
                delta = jnp.linalg.solve(A, -(J.T @ r) - smooth_w * (pose - pose0))
                norm = jnp.linalg.norm(delta)
                delta = jnp.where(norm > max_step, delta * (max_step / norm), delta)
                return jnp.clip(pose + delta, lower, upper), None

            pose_f, _ = jax.lax.scan(step, pose0, lm_lams)
            return pose_f, _fwd(pose_f)

        _vmap_fn = jax.vmap(_inverse_one, in_axes=(0, 0, 0, 0, 0, 0, 0, 0))

        def _batch_ik(pose0s, dess, des_masks):
            return _vmap_fn(pose0s, dess, des_masks, J_all, w_all, vt_all, pd_all, aw_all)

        self._jit_vmap_mano_inverse = jax.jit(_batch_ik)

        self._jit_vmap_mano_inverse(
            jnp.zeros((n_f, n_pose), dtype=jnp.float64),
            jnp.zeros((n_f, 2, 3), dtype=jnp.float64),
            jnp.ones((n_f, 2), dtype=jnp.float64))

    def estimate(self, hand_markers, finger_markers):
        """Estimate hand state from positions of hand markers and finger markers.

        Parameters
        ----------
        hand_markers : list
            Markers on hand in order 'hand_top', 'hand_left', 'hand_right'.

        finger_markers : dict (str to array-like)
            Positions of markers on fingers.
        """
        current_hand_markers2world = estimate_hand_pose(*hand_markers)
        if np.any(np.isnan(current_hand_markers2world)):
            warnings.warn(
                "[MarkerBasedRecordMapping] Cannot estimate hand pose. "
                "Detected NaN.")
        else:
            self.current_hand_markers2world = current_hand_markers2world
        self.mano2world_ = pt.concat(
            self.mano2hand_markers_, self.current_hand_markers2world)

        available_fingers = self.finger_names_.intersection(
            finger_markers.keys())

        world2mano = pt.invert_transform(self.mano2world_, check=False)
        for finger_name in available_fingers:
            markers_in_world = np.atleast_2d(finger_markers[finger_name])
            self.markers_in_mano[finger_name] = np.dot(
                pt.vectors_to_points(markers_in_world), world2mano.T)[:, :3]

        self.start_measurement()

        import jax.numpy as jnp
        finger_names = self._vmap_finger_names
        fks = self._vmap_fks
        n_f = len(finger_names)
        pose0s = np.zeros((n_f, 9), dtype=float)
        dess = np.zeros((n_f, 2, 3), dtype=float)
        des_masks = np.zeros((n_f, 2), dtype=float)
        has_markers = np.zeros(n_f, dtype=bool)

        for fi, (fname, fk) in enumerate(zip(finger_names, fks)):
            pose0s[fi] = np.where(np.isnan(fk.current_pose), 0.0, fk.current_pose)
            markers = self.markers_in_mano.get(fname)
            if (markers is not None and fname in available_fingers
                    and not np.any(np.isnan(markers))):
                n_real = markers.shape[0]
                dess[fi, :n_real] = markers
                des_masks[fi, :n_real] = 1.0
                if n_real < 2 and fk.last_forward_result is not None:
                    dess[fi, n_real:] = fk.last_forward_result[n_real:]
                has_markers[fi] = True

        all_poses_jax, all_verts_jax = self._jit_vmap_mano_inverse(
            jnp.array(pose0s, dtype=jnp.float64),
            jnp.array(dess, dtype=jnp.float64),
            jnp.array(des_masks, dtype=jnp.float64))

        all_poses = np.array(all_poses_jax)
        all_verts = np.array(all_verts_jax)

        for fi, (fname, fk) in enumerate(zip(finger_names, fks)):
            if not has_markers[fi]:
                continue
            pose_f = all_poses[fi]
            if not np.any(np.isnan(pose_f)):
                fk.current_pose[:] = pose_f
                fk.last_forward_result = all_verts[fi]
                self.hand_state_.pose[fk.finger_pose_param_indices] = pose_f
            elif fk.last_forward_result is None:
                fk.forward(fk.current_pose)

        self.stop_measurement()
        if self.verbose:
            print(f"[{type(self).__name__}] Time for optimization: "
                  f"{self.last_timing():.4f} s")

        self.hand_state_.recompute_mesh(self.mano2world_)


def estimate_hand_pose(hand_top, hand_left, hand_right):
    """Estimate pose of the hand from markers on the back of the hand.

    To estimate the pose of the MANO frame in world frame, we first derive
    the pose of the hand based on three labeled markers on the back of the
    hand. In accordance with the two-vector representation (see Corke:
    Robotics, Vision and Control), we define the hand frame orientation by
    the approach vector (direction from right to front hand marker) and
    the orientation vector (normal of the plane defined by the three
    markers). The origin of the hand frame can be any point in the plane of
    the three markers. We choose the right marker.

    Parameters
    ----------
    hand_top : array, shape (3,)
        Position of hand_top marker.

    hand_left : array, shape (3,)
        Position of hand_left marker.

    hand_right : array, shape (3,)
        Position of hand_right marker.

    Returns
    -------
    hand_markers2world : array, shape (4, 4)
        Pose of hand marker frame.
    """
    right2left = hand_left - hand_right
    approach = pr.norm_vector(hand_top - hand_right)
    orientation = pr.norm_vector(np.cross(approach, right2left))
    normal = np.cross(orientation, approach)
    hand_pose = np.eye(4)
    hand_pose[:3, :3] = np.column_stack((normal, orientation, approach))
    hand_pose[:3, 3] = hand_right
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

        self._build_jax_inverse(action_weights)

    def _search_similar_vertices(self, finger_pose_param_indices, hand_state):
        # search for vertices that are influenced by the same pose parameters
        # TODO mapping to indices of weights might be correct by accident
        return np.unique(np.nonzero(
            hand_state.pose_parameters["weights"][
                :, np.unique(finger_pose_param_indices // 3)])[0])

    def reset(self):
        """Set all joint angles of the finger to 0."""
        self.current_pose[:] = 0.0

    def reduce_pose_parameters(self, hand_state):
        """Reduce parameters of the MANO model to this finger.

        Parameters
        ----------
        hand_state : HandState
            State of the hand mesh.

        Returns
        -------
        pose_params : dict
            Reduced set of pose parameters of MANO. Contains fields 'J',
            'weights', 'kintree_table', 'v_template', 'posedirs'.

        finger_opt_vertex_indices : list
            Indices of vertices that will be used during numerical inverse
            kinematics of this finger.
        """
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

    def has_cached_forward_kinematics(self):
        """Check if this object has a cached forward kinematics result."""
        return self.last_forward_result is not None

    def forward(self, pose=None, return_cached_result=False):
        """Compute position at the tip of the finger for given joint parameters.

        Parameters
        ----------
        pose : array, shape (n_finger_joints * 3,), optional (default: None)
            Joint angles.

        return_cached_result : bool, optional (default: False)
            Return cached result of previous forward kinematics calculation.

        Returns
        -------
        pos : array, shape (n_markers_per_finger, 3)
            Vertex positions.
        """
        if return_cached_result:
            assert self.last_forward_result is not None
            return self.last_forward_result

        self._optimizer_pose[3:] = pose
        self.last_forward_result = hand_vertices(
            pose=self._optimizer_pose, **self.finger_pose_params)
        return self.last_forward_result

    def _build_jax_inverse(self, action_weights):
        """Build and warm-up a JAX LM inverse for this finger."""
        import jax
        jax.config.update("jax_enable_x64", True)
        import jax.numpy as jnp
        from .mano_jax import build_hand_vertices_fn

        pp = self.finger_pose_params
        fwd_fn = build_hand_vertices_fn(pp["kintree_table"])
        J_j = jnp.array(pp["J"])
        w_j = jnp.array(pp["weights"])
        vt_j = jnp.array(pp["v_template"])
        pd_j = jnp.array(pp["posedirs"])

        n_pose = len(self.current_pose)
        n_markers = len(self.finger_vertex_indices)
        aw = jnp.array(action_weights)

        lower = jnp.full(n_pose, -0.4 * np.pi)
        upper = jnp.full(n_pose, 0.4 * np.pi)
        # MANO FK Jacobian scale: dposition/dangle ~ 0.02-0.08 m/rad,
        # so J^T J diagonal ~ 0.001-0.006.  λ starts at 0.05 (gradient-descent
        # steps ~0.1 rad) and decays to 0.001 for Newton-like fine convergence.
        # Step clamped to 0.2 rad to prevent divergence for unreachable targets.
        # Temporal smoothing (smooth_w) penalises deviation from pose0 (the
        # previous frame's solution), preventing jumps between local minima.
        lm_lams = jnp.exp(jnp.linspace(jnp.log(0.05), jnp.log(0.001), 15))
        max_step = 0.2
        smooth_w = jnp.float64(2e-3)

        def _residual(pose, des):
            # des.shape[0] is static at JIT trace time — retraces per unique shape.
            n_des = des.shape[0]
            full_pose = jnp.concatenate([jnp.zeros(3), pose])
            verts = fwd_fn(J_j, w_j, vt_j, pd_j, full_pose)
            diff = (verts[:n_des] - des).ravel()
            pos_pos = jnp.maximum(0.0, pose)
            neg_pos = -jnp.minimum(0.0, pose)
            # Component-wise regularisation: each DOF gets its own residual term
            # aw[i]*max(0,±pose[i]).  Avoids the cross-coupling of the dot-product
            # form (dot(aw, pos_pose)), which amplifies every DOF's gradient by the
            # sum of all other DOFs — causing the thumb to over-regularise and
            # jump between local minima despite temporal smoothing.
            reg = jnp.concatenate([aw[0] * pos_pos, aw[1] * neg_pos])
            return jnp.concatenate([diff, reg])

        def _fwd_all(pose):
            """Return all n_markers vertex positions for cache update."""
            full_pose = jnp.concatenate([jnp.zeros(3), pose])
            return fwd_fn(J_j, w_j, vt_j, pd_j, full_pose)

        def _inverse_jax(pose0, des):
            def step(pose, lam):
                r = _residual(pose, des)
                J = jax.jacfwd(lambda p: _residual(p, des))(pose)
                # Temporal smoothing: augments Hessian diagonal by smooth_w and
                # adds smooth_w*(pose-pose0) to gradient, keeping the solution
                # near the previous frame to prevent local-minima jumps.
                A = J.T @ J + (lam + smooth_w) * jnp.eye(n_pose)
                delta = jnp.linalg.solve(A, -(J.T @ r) - smooth_w * (pose - pose0))
                norm = jnp.linalg.norm(delta)
                delta = jnp.where(norm > max_step, delta * (max_step / norm), delta)
                return jnp.clip(pose + delta, lower, upper), None
            pose_f, _ = jax.lax.scan(step, pose0, lm_lams)
            # Return vertices alongside pose so inverse() can update the FK
            # cache without an extra numpy hand_vertices call.
            return pose_f, _fwd_all(pose_f)

        self._jit_inverse = jax.jit(_inverse_jax)

        # Warmup — compile once at init time so first frame is fast.
        self._jit_inverse(
            jnp.zeros(n_pose, dtype=jnp.float64),
            jnp.zeros((n_markers, 3), dtype=jnp.float64))

    def inverse(self, position):
        """Estimate finger joint parameters from position.

        Parameters
        ----------
        position : array, shape (n_markers_per_finger, 3)
            Desired position of vertices.

        Returns
        -------
        current_pose : array, shape (n_finger_joints * 3,)
            Joint angles.
        """
        import jax.numpy as jnp
        pos = np.asarray(position, dtype=float)
        if np.any(np.isnan(pos)):
            # Missing marker: hold previous solution so NaN does not propagate.
            if self.last_forward_result is None:
                self.forward(self.current_pose)
            return self.current_pose
        # Sanitise warm-start: a previous NaN result must not poison the
        # temporal-smoothing term smooth_w*(pose-pose0).
        pose0 = np.where(np.isnan(self.current_pose), 0.0, self.current_pose)
        pose_f_jax, verts_f_jax = self._jit_inverse(
            jnp.array(pose0, dtype=jnp.float64),
            jnp.array(pos, dtype=jnp.float64))
        pose_f = np.array(pose_f_jax)
        if not np.any(np.isnan(pose_f)):
            self.current_pose[:] = pose_f
            # Use vertices already computed inside JIT — avoids an extra
            # numpy hand_vertices call that was the main per-frame bottleneck.
            self.last_forward_result = np.array(verts_f_jax)
        elif self.last_forward_result is None:
            self.forward(self.current_pose)
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
        """Compute error for numerical inverse kinematics.

        Parameters
        ----------
        finger_pose : array, shape (n_finger_joints * 3,)
            Joint angles.

        desired_finger_pos : array, shape (n_markers_per_finger, 3)
            Desired finger positions.
        """
        positions = self.forward_kinematics(finger_pose)
        desired_finger_pos = np.atleast_2d(desired_finger_pos)

        # TODO seems fragile, what if we only have a middle marker and no tip?
        # in case there are no middle markers available:
        positions = positions[:len(desired_finger_pos)]

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
