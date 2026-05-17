"""Embodiment mapping.

Maps MANO states to robotic hands.
"""
import warnings
import numpy as np

from .kinematics import Kinematics
from .record_markers import make_finger_kinematics
from .target_configurations import TARGET_CONFIG
from .timing import TimeableMixin
import pytransform3d.transformations as pt


class HandEmbodiment(TimeableMixin):
    """Solves embodiment mapping from MANO model to robotic hand.

    Parameters
    ----------
    hand_state : HandState
        State of the MANO mesh, defined internally by pose and shape
        parameters, and whether it is a left or right hand. It also
        stores the mesh.

    target_config : dict or str
        Configuration for the target system as a dictionary or string that
        identifies the target hand.

    use_fingers : tuple of str, optional (default: ('thumb', 'index', 'middle'))
        Fingers for which we compute the embodiment mapping.

    mano_finger_kinematics : list, optional (default: None)
        If finger kinematics are already available, e.g., from the record
        mapping, these can be passed here. Otherwise they will be created.

    initial_handbase2world : array-like, shape (4, 4), optional (default: None)
        Initial transform from hand base to world coordinates.

    only_tip : bool, optional (default: False)
        Use only tip frames in inverse kinematics.

    verbose : int, optional (default: 0)
        Verbosity level

    measure_time : bool
        Measure computation time for each frame.

    Attributes
    ----------
    finger_names_ : tuple of str
        Fingers for which we compute the embodiment mapping.

    hand_state_ : mocap.mano.HandState
        MANO hand state. This state should be updated by the record mapping
        so that we can perform a subsequent embodiment mapping based on the
        current state.

    transform_manager_ : pytransform3d.transform_manager.TransformManager
        Exposes transform manager that represents the target system.
        It holds information about all links, joints, visual objects, and
        collision objects of the target system (robotic hand). It is used
        to compute forward and inverse kinematics and can be used for
        visualization. This representation of the target system's state will
        be updated with each embodiment mapping.
    """
    def __init__(
            self, hand_state, target_config,
            use_fingers=("thumb", "index", "middle"),
            mano_finger_kinematics=None, initial_handbase2world=None,
            only_tip=False, verbose=0, measure_time=False,
            objective='cartesian'):
        super(HandEmbodiment, self).__init__(verbose or measure_time)

        if isinstance(target_config, str):
            target_config = TARGET_CONFIG[target_config]
        self.finger_names_ = use_fingers
        self.hand_state_ = hand_state
        if mano_finger_kinematics is None:
            self.mano_finger_kinematics = {}
            for finger_name in use_fingers:
                self.mano_finger_kinematics[finger_name] = \
                    make_finger_kinematics(self.hand_state_, finger_name)
        else:
            self.mano_finger_kinematics = mano_finger_kinematics
        self.handbase2robotbase = target_config["handbase2robotbase"]
        for finger_name in use_fingers:
            assert finger_name in self.mano_finger_kinematics

        self.target_kin, self.vis_kin = load_kinematic_model(target_config)
        self.target_finger_chains = {}
        self.vis_finger_chains = {}
        self.joint_angles = {}
        self.base_frame = target_config["base_frame"]
        for finger_name in target_config["ee_frames"].keys():
            assert finger_name in target_config["joint_names"]
            assert finger_name in target_config["ee_frames"]
            ee_frames = [target_config["ee_frames"][finger_name]]
            if not only_tip and "intermediate_frames" in target_config:
                ee_frames.append(
                    target_config["intermediate_frames"][finger_name])
            self.target_finger_chains[finger_name] = \
                self.target_kin.create_multi_chain(
                    target_config["joint_names"][finger_name],
                    self.base_frame, ee_frames)
            self.vis_finger_chains[finger_name] = \
                self.vis_kin.create_multi_chain(
                    target_config["joint_names"][finger_name],
                    self.base_frame, ee_frames)
            self.joint_angles[finger_name] = \
                np.zeros(len(target_config["joint_names"][finger_name]))

        self._update_hand_base_pose(initial_handbase2world)

        self.coupled_joints = target_config.get("coupled_joints", None)
        self.post_embodiment_hook = target_config.get(
            "post_embodiment_hook", None)

        self.verbose = verbose
        self.objective = objective

        if objective == 'cartesian':
            self._build_vmap_ik(target_config)
        else:
            self._build_vmap_mesh_ik(target_config, objective)

    def _build_vmap_ik(self, target_config):
        """Build and warm-up a vmapped JIT that parallelises IK over all fingers.

        Uses jax.vmap to vectorise the LM IK over all fingers, allowing XLA
        to fuse their FK + Jacobian + solve into a single CUDA kernel.
        """
        import jax
        import jax.numpy as jnp
        from .kinematics import _rot4_jax_dynamic

        chains_list = [self.target_finger_chains[f] for f in self.finger_names_]
        n_f = len(chains_list)
        n_ee_list = [len(c.ee_frames) for c in chains_list]
        assert len(set(n_ee_list)) == 1, "All fingers must have the same number of EE frames for vmap"
        n_ee = n_ee_list[0]
        max_j = max(c.n_joints for c in chains_list)
        max_steps = max(
            len(entries)
            for c in chains_list
            for entries in c._jax_params.all_path_entries)

        # Padded arrays: (n_f, n_ee, max_steps, ...)
        T0s_np = np.tile(np.eye(4), (n_f, n_ee, max_steps, 1, 1))
        is_j_np = np.zeros((n_f, n_ee, max_steps), dtype=bool)
        axes_np = np.zeros((n_f, n_ee, max_steps, 3))
        jidx_np = np.zeros((n_f, n_ee, max_steps), dtype=np.int32)
        # angle_transform: (scale, offset, lo, hi) per step
        at_np = np.zeros((n_f, n_ee, max_steps, 4))
        at_np[:, :, :, 0] = 1.0        # scale=1
        at_np[:, :, :, 2] = -np.inf    # lo=-inf
        at_np[:, :, :, 3] = np.inf     # hi=+inf
        lower_np = np.zeros((n_f, max_j))
        upper_np = np.zeros((n_f, max_j))

        for fi, chain in enumerate(chains_list):
            for ji in range(chain.n_joints):
                lower_np[fi, ji] = chain.joint_limits[ji, 0]
                upper_np[fi, ji] = chain.joint_limits[ji, 1]
            for ei, entries in enumerate(chain._jax_params.all_path_entries):
                for si, entry in enumerate(entries):
                    if entry[0] == 'joint':
                        _, c2p, axis, cidx, _, sc, off, lo, hi = entry
                        T0s_np[fi, ei, si] = np.array(c2p)
                        is_j_np[fi, ei, si] = True
                        axes_np[fi, ei, si] = axis
                        jidx_np[fi, ei, si] = cidx
                        at_np[fi, ei, si] = [sc, off, lo, hi]
                    else:
                        T0s_np[fi, ei, si] = np.array(entry[1])
                        # is_j stays False; padding identity already set

        # Per-finger EE weights: (n_f, n_ee*3). ee_weights in target_config is a
        # dict mapping finger_name -> [w_ee0, w_ee1, ...].  Weighting tip > middle
        # (e.g. [1.0, 0.1]) makes the optimizer prioritise Cartesian tip accuracy
        # over matching the intermediate marker, preventing over-flexion of distal
        # joints when the finger proportions of the robot differ from MANO.
        ee_w_config = target_config.get("ee_weights", {})
        weights_np = np.ones((n_f, n_ee * 3))
        for fi, fname in enumerate(self.finger_names_):
            if fname in ee_w_config:
                w_ee = np.asarray(ee_w_config[fname], dtype=float)  # (n_ee,)
                weights_np[fi] = np.repeat(w_ee, 3)                 # (n_ee*3,)

        jreg_cfg = target_config.get("joint_weights", {})
        jreg_np = np.zeros((n_f, max_j))
        for fi, fname in enumerate(self.finger_names_):
            if fname in jreg_cfg:
                w_j = np.asarray(jreg_cfg[fname], dtype=float)
                jreg_np[fi, :len(w_j)] = w_j

        T0s_j = jnp.array(T0s_np)
        is_j_j = jnp.array(is_j_np)
        axes_j = jnp.array(axes_np)
        jidx_j = jnp.array(jidx_np)
        at_j = jnp.array(at_np)
        lower_j = jnp.array(lower_np)
        upper_j = jnp.array(upper_np)
        weights_j = jnp.array(weights_np)
        jreg_j = jnp.array(jreg_np)
        # Shadow FK Jacobian: dp/dq ~ link_length ~ 0.01-0.04 m/rad,
        # so J^T J diagonal ~ 1e-4 to 1.6e-3.  λ_min = 1e-4 keeps us in the
        # gradient-descent regime and prevents Newton-like overshoots (~0.4 rad)
        # that cause oscillation.  Step clamped to 0.15 rad for stability.
        # Temporal smoothing (smooth_w) penalises deviation from the previous
        # frame's solution: adds smooth_w to the Hessian diagonal and a gradient
        # term smooth_w*(q-q_prev), preventing jumps between local minima.
        lm_lams = jnp.exp(jnp.linspace(jnp.log(0.01), jnp.log(1e-4), 15))
        max_step_rob = 0.15
        smooth_w = jnp.float64(1e-3)

        def _fk_one_ee(q, T0s_e, is_j_e, axes_e, jidx_e, at_e):
            """FK for one EE path — Python loop unrolled by JAX for fusion."""
            T_acc = jnp.eye(4, dtype=jnp.float64)
            for si in range(max_steps):  # static loop: unrolled at trace time
                t0 = T0s_e[si]
                is_j = is_j_e[si]
                ax = axes_e[si]
                jidx = jidx_e[si]
                at = at_e[si]
                raw = q[jidx]
                theta = jnp.clip(at[0] * raw + at[1], at[2], at[3])
                R4 = _rot4_jax_dynamic(ax, theta)
                T_joint = t0 @ R4
                step_T = jnp.where(is_j, T_joint, t0)
                T_acc = step_T @ T_acc
            return T_acc[:3, 3]

        def _fk_finger(q, T0s_f, is_j_f, axes_f, jidx_f, at_f):
            return jnp.stack([
                _fk_one_ee(q, T0s_f[ei], is_j_f[ei], axes_f[ei], jidx_f[ei], at_f[ei])
                for ei in range(n_ee)])

        def _ik_finger(q0, des, T0s_f, is_j_f, axes_f, jidx_f, at_f,
                       lower_f, upper_f, w_f, jreg_f):
            def fk(q):
                return _fk_finger(q, T0s_f, is_j_f, axes_f, jidx_f, at_f)

            def lm_step(q, lam):
                r = (des - fk(q)).ravel()
                J = jax.jacfwd(fk)(q).reshape(-1, max_j)
                # Apply per-EE weights: scale residual and Jacobian rows so that
                # J_w^T J_w = J^T W^2 J with W = diag(w_f).  This prioritises
                # tip accuracy (high weight) over the intermediate marker (low
                # weight) without changing the solver structure.
                r_w = r * w_f
                J_w = J * w_f[:, None]
                # Per-joint regularisation: adds diag(jreg_f) to the Hessian
                # and jreg_f*q to the gradient.  Discourages abduction joints
                # (e.g. LFJ4) from being exploited to reduce along-finger
                # Cartesian error at the cost of orthogonal displacement.
                # Temporal smoothing: augments Hessian by smooth_w*I and adds
                # smooth_w*(q-q0) to the gradient, keeping q near the previous
                # frame's solution without stacking extra Jacobian rows.
                A = (J_w.T @ J_w
                     + jnp.diag(jreg_f)
                     + (lam + smooth_w) * jnp.eye(max_j, dtype=jnp.float64))
                rhs = J_w.T @ r_w - jreg_f * q - smooth_w * (q - q0)
                delta = jnp.linalg.solve(A, rhs)
                norm = jnp.linalg.norm(delta)
                delta = jnp.where(norm > max_step_rob, delta * (max_step_rob / norm), delta)
                return jnp.clip(q + delta, lower_f, upper_f), None

            q_f, _ = jax.lax.scan(lm_step, q0, lm_lams)
            return q_f

        def _ik_all_vmap(q0s, dess):
            return jax.vmap(_ik_finger)(
                q0s, dess, T0s_j, is_j_j, axes_j, jidx_j, at_j,
                lower_j, upper_j, weights_j, jreg_j)

        self._jit_ik_vmap = jax.jit(_ik_all_vmap)
        self._vmap_n_f = n_f
        self._vmap_max_j = max_j
        self._chains_n_joints = [c.n_joints for c in chains_list]

        # Warmup
        q0s0 = jnp.zeros((n_f, max_j), dtype=jnp.float64)
        dess0 = jnp.zeros((n_f, n_ee, 3), dtype=jnp.float64)
        self._jit_ik_vmap(q0s0, dess0)

    def _build_vmap_mesh_ik(self, target_config, objective):
        """Build and warm-up a vmapped JIT for mesh-distance-based IK.

        Computes contact surface points for each robot link in local frame
        once at init time (Poisson-sampled), then uses jax.grad on the chosen
        point-cloud metric to drive gradient-descent LM steps per frame.
        """
        import jax
        import jax.numpy as jnp
        import open3d as o3d
        from .kinematics import _rot4_jax_dynamic, _build_jax_fk_params
        from .metrics import MANO_CONTACT_SURFACE_VERTICES, CONTACT_SURFACE_VERTICES
        from .mesh_metrics_jax import METRIC_FNS

        metric_fn = METRIC_FNS[objective]

        contact_surface_verts = target_config.get("contact_surface_vertices")
        if contact_surface_verts is None:
            raise ValueError(
                f"target_config must include 'contact_surface_vertices' for "
                f"objective='{objective}'. Add it to the hand config or use "
                f"objective='cartesian'.")

        chains_list = [self.target_finger_chains[f] for f in self.finger_names_]
        n_f = len(chains_list)
        max_j = max(c.n_joints for c in chains_list)

        n_vis_list = [len(contact_surface_verts[f]) for f in self.finger_names_]
        assert len(set(n_vis_list)) == 1, \
            "All fingers must have the same number of contact visuals for vmap"
        n_vis = n_vis_list[0]

        max_mano_pts = max(
            len(MANO_CONTACT_SURFACE_VERTICES[f]) for f in self.finger_names_)

        # --- Extract local-frame robot contact points (one-time numpy op) ---
        tm = self.vis_kin.tm
        n_robot_pts = 100  # Poisson samples per visual
        local_pts_np = np.zeros((n_f, n_vis, n_robot_pts, 3))

        for fi, fname in enumerate(self.finger_names_):
            vis_dict = contact_surface_verts[fname]
            for vi, (vis_name, vertex_indices) in enumerate(vis_dict.items()):
                link_name = vis_name.split(":")[1].split("/")[0]
                mesh_obj = next(m for m in tm.visuals if m.frame == vis_name)

                o3d_mesh = o3d.io.read_triangle_mesh(mesh_obj.filename)
                verts = np.array(o3d_mesh.vertices) * mesh_obj.scale

                T_vis2link = tm.get_transform(vis_name, link_name)
                R, t = T_vis2link[:3, :3], T_vis2link[:3, 3]
                verts_link = (R @ verts.T).T + t

                tri_mesh = o3d.geometry.TriangleMesh()
                tri_mesh.vertices = o3d.utility.Vector3dVector(verts_link)
                triangles = np.array(o3d_mesh.triangles)
                vertex_set = set(vertex_indices)
                triangles_filt = np.array([
                    tri for tri in triangles
                    if any(v in vertex_set for v in tri)])
                tri_mesh.triangles = o3d.utility.Vector3iVector(triangles_filt)

                pc = tri_mesh.sample_points_poisson_disk(n_robot_pts)
                local_pts_np[fi, vi] = np.array(pc.points)

        # --- Build FK path arrays for each contact link ---
        contact_link_names = []
        for fname in self.finger_names_:
            vis_names = list(contact_surface_verts[fname].keys())
            contact_link_names.append([
                vn.split(":")[1].split("/")[0] for vn in vis_names])

        fk_entries_all = []
        for fi, fname in enumerate(self.finger_names_):
            chain = chains_list[fi]
            per_vis = []
            for link_name in contact_link_names[fi]:
                params = _build_jax_fk_params(
                    tm, chain.joint_names, self.base_frame, [link_name])
                per_vis.append(params.all_path_entries[0])
            fk_entries_all.append(per_vis)

        link_max_steps = max(
            len(entries)
            for fi_entries in fk_entries_all
            for entries in fi_entries)

        T0s_link_np = np.tile(np.eye(4), (n_f, n_vis, link_max_steps, 1, 1))
        is_j_link_np = np.zeros((n_f, n_vis, link_max_steps), dtype=bool)
        axes_link_np = np.zeros((n_f, n_vis, link_max_steps, 3))
        jidx_link_np = np.zeros((n_f, n_vis, link_max_steps), dtype=np.int32)
        at_link_np = np.zeros((n_f, n_vis, link_max_steps, 4))
        at_link_np[:, :, :, 0] = 1.0
        at_link_np[:, :, :, 2] = -np.inf
        at_link_np[:, :, :, 3] = np.inf

        for fi, per_vis in enumerate(fk_entries_all):
            for vi, entries in enumerate(per_vis):
                for si, entry in enumerate(entries):
                    if entry[0] == 'joint':
                        _, c2p, axis, cidx, _, sc, off, lo, hi = entry
                        T0s_link_np[fi, vi, si] = np.array(c2p)
                        is_j_link_np[fi, vi, si] = True
                        axes_link_np[fi, vi, si] = axis
                        jidx_link_np[fi, vi, si] = cidx
                        at_link_np[fi, vi, si] = [sc, off, lo, hi]
                    else:
                        T0s_link_np[fi, vi, si] = np.array(entry[1])

        lower_np = np.zeros((n_f, max_j))
        upper_np = np.zeros((n_f, max_j))
        for fi, chain in enumerate(chains_list):
            for ji in range(chain.n_joints):
                lower_np[fi, ji] = chain.joint_limits[ji, 0]
                upper_np[fi, ji] = chain.joint_limits[ji, 1]

        jreg_cfg = target_config.get("joint_weights", {})
        jreg_np = np.zeros((n_f, max_j))
        for fi, fname in enumerate(self.finger_names_):
            if fname in jreg_cfg:
                w_j = np.asarray(jreg_cfg[fname], dtype=float)
                jreg_np[fi, :len(w_j)] = w_j

        n_mano_per_finger = np.array([
            len(MANO_CONTACT_SURFACE_VERTICES[f]) for f in self.finger_names_],
            dtype=np.int32)

        local_pts_j = jnp.array(local_pts_np)
        T0s_link_j = jnp.array(T0s_link_np)
        is_j_link_j = jnp.array(is_j_link_np)
        axes_link_j = jnp.array(axes_link_np)
        jidx_link_j = jnp.array(jidx_link_np)
        at_link_j = jnp.array(at_link_np)
        lower_j = jnp.array(lower_np)
        upper_j = jnp.array(upper_np)
        jreg_j = jnp.array(jreg_np)

        lm_lams = jnp.exp(jnp.linspace(jnp.log(0.01), jnp.log(1e-4), 15))
        max_step_rob = 0.15
        smooth_w = jnp.float64(1e-3)

        def _fk_link_transform(q, T0s_l, is_j_l, axes_l, jidx_l, at_l):
            T_acc = jnp.eye(4, dtype=jnp.float64)
            for si in range(link_max_steps):
                t0 = T0s_l[si]
                is_j = is_j_l[si]
                ax = axes_l[si]
                jidx = jidx_l[si]
                at = at_l[si]
                raw = q[jidx]
                theta = jnp.clip(at[0] * raw + at[1], at[2], at[3])
                R4 = _rot4_jax_dynamic(ax, theta)
                T_joint = t0 @ R4
                step_T = jnp.where(is_j, T_joint, t0)
                T_acc = step_T @ T_acc
            return T_acc

        def _robot_contact_pts_finger(q, local_pts_f, T0s_f, is_j_f, axes_f,
                                      jidx_f, at_f):
            def transform_vis(local_l, T0s_l, is_j_l, axes_l, jidx_l, at_l):
                T = _fk_link_transform(q, T0s_l, is_j_l, axes_l, jidx_l, at_l)
                return (T[:3, :3] @ local_l.T).T + T[:3, 3]  # (n_robot_pts, 3)

            all_pts = jax.vmap(transform_vis)(
                local_pts_f, T0s_f, is_j_f, axes_f, jidx_f, at_f)
            return all_pts.reshape(-1, 3)

        def _ik_finger_mesh(q0, mano_pts_f, mask_f, T0s_f, is_j_f, axes_f,
                            jidx_f, at_f, local_pts_f, lower_f, upper_f,
                            jreg_f):
            def loss_fn(q):
                robot_pts = _robot_contact_pts_finger(
                    q, local_pts_f, T0s_f, is_j_f, axes_f, jidx_f, at_f)
                return metric_fn(mano_pts_f, robot_pts, mask_a=mask_f)

            def lm_step(q, lam):
                grad_metric = jax.grad(loss_fn)(q)
                grad_full = (grad_metric
                             + jreg_f * q
                             + smooth_w * (q - q0))
                A = ((lam + smooth_w) * jnp.eye(max_j, dtype=jnp.float64)
                     + jnp.diag(jreg_f))
                delta = jnp.linalg.solve(A, -grad_full)
                norm = jnp.linalg.norm(delta)
                delta = jnp.where(
                    norm > max_step_rob, delta * (max_step_rob / norm), delta)
                return jnp.clip(q + delta, lower_f, upper_f), None

            q_f, _ = jax.lax.scan(lm_step, q0, lm_lams)
            return q_f

        def _ik_all_mesh_vmap(q0s, mano_pts, masks):
            return jax.vmap(_ik_finger_mesh)(
                q0s, mano_pts, masks,
                T0s_link_j, is_j_link_j, axes_link_j, jidx_link_j, at_link_j,
                local_pts_j, lower_j, upper_j, jreg_j)

        self._jit_mesh_ik_vmap = jax.jit(_ik_all_mesh_vmap)
        self._vmap_n_f = n_f
        self._vmap_max_j = max_j
        self._chains_n_joints = [c.n_joints for c in chains_list]
        self._max_mano_pts = max_mano_pts
        self._n_mano_per_finger = n_mano_per_finger

        q0s0 = jnp.zeros((n_f, max_j), dtype=jnp.float64)
        mano_pts0 = jnp.zeros((n_f, max_mano_pts, 3), dtype=jnp.float64)
        masks0 = jnp.ones((n_f, max_mano_pts), dtype=jnp.float64)
        self._jit_mesh_ik_vmap(q0s0, mano_pts0, masks0)

    def _get_mano_contact_pts(self):
        """Extract current MANO contact surface points in robot base frame.

        hand_mesh.vertices are in world frame (recompute_mesh applies mano2world_).
        _update_hand_base_pose has already stored world2robotbase in the TM,
        so we use that to bring world-frame vertices into robot base frame.
        """
        from .metrics import MANO_CONTACT_SURFACE_VERTICES
        import jax.numpy as jnp

        world2robotbase = self.target_kin.tm.get_transform(
            "world", self.base_frame)

        verts = np.array(self.hand_state_.hand_mesh.vertices)
        n_f = self._vmap_n_f
        max_mano = self._max_mano_pts
        mano_pts_np = np.zeros((n_f, max_mano, 3))
        masks_np = np.zeros((n_f, max_mano))

        for fi, fname in enumerate(self.finger_names_):
            idx = MANO_CONTACT_SURFACE_VERTICES[fname]
            pts_robotbase = pt.transform(
                world2robotbase,
                pt.vectors_to_points(verts[idx]))[:, :3]
            n = len(idx)
            mano_pts_np[fi, :n] = pts_robotbase
            masks_np[fi, :n] = 1.0

        return jnp.array(mano_pts_np), jnp.array(masks_np)

    def solve(self, handbase2world=None, return_desired_positions=False,
              use_cached_forward_kinematics=False):
        """Solve embodiment.

        Internally we rely on the attribute hand_state_ to be updated
        before this function is called. The hand state will contain all
        necessary information to perform the embodiment of finger
        configurations.

        Parameters
        ----------
        handbase2world : array-like, shape (4, 4), optional (default: None)
            Transform from MANO base to world.

        return_desired_positions : bool, optional (default: False)
            Return desired position for each finger in robot's base frame.

        use_cached_forward_kinematics : bool, optional (defalt: False)
            If the underlying MANO model was previously fitted from data,
            you can use the cached results. If you want to use this option
            you should provide the constructor argument
            'mano_finger_kinematics'.

        Returns
        -------
        joint_angles : dict
            Maps finger names to corresponding joint angles in the order that
            is given in the target configuration.

        desired_positions : dict, optional
            Maps finger names to desired finger tip positions in robot base
            frame.
        """
        self.start_measurement()

        desired_positions = self._mano_forward_kinematics(use_cached_forward_kinematics)
        self._robotic_hand_inverse_kinematics(desired_positions)
        self._update_hand_base_pose(handbase2world)

        self.stop_measurement()
        if self.verbose:
            print(f"[{type(self).__name__}] Time for optimization: "
                  f"{self.last_timing():.4f} s")

        if self.vis_kin is not self.target_kin:
            for finger_name in self.finger_names_:
                self.vis_finger_chains[finger_name].forward(
                    self.joint_angles[finger_name])

        if return_desired_positions:
            return self.joint_angles, desired_positions
        else:
            return self.joint_angles

    def _mano_forward_kinematics(self, use_cached_forward_kinematics):
        """MANO forward kinematics.

        Parameters
        ----------
        use_cached_forward_kinematics : bool
            Reused cached result of forward kinematics from record mapping
            in MANO model.

        Returns
        -------
        desired_positions : dict
            Desired positions of expected marker positions in frame of the
            robotic target system.
        """
        desired_positions = {}
        for finger_name in self.finger_names_:
            if use_cached_forward_kinematics:
                # because it has been computed during record mapping
                markers_in_handbase = self.mano_finger_kinematics[finger_name].forward(
                    None, return_cached_result=True)
            else:
                markers_in_handbase = self.mano_finger_kinematics[finger_name].forward(
                    self.hand_state_.pose[
                        self.mano_finger_kinematics[
                            finger_name].finger_pose_param_indices])

            markers_in_robotbase = pt.transform(
                self.handbase2robotbase,
                pt.vectors_to_points(markers_in_handbase))[:, :3]
            n_ee_frames = len(self.target_finger_chains[finger_name].ee_frames)
            desired_positions[finger_name] = markers_in_robotbase[:n_ee_frames]
        return desired_positions

    def _robotic_hand_inverse_kinematics(self, desired_positions):
        """Robotic hand inverse kinematics.

        Parameters
        ----------
        desired_positions : dict
            Desired positions of expected marker positions in frame of the
            robotic target system. Only used when objective='cartesian'.
        """
        import jax.numpy as jnp
        n_f = self._vmap_n_f
        max_j = self._vmap_max_j
        q0s = np.zeros((n_f, max_j))
        for fi, f in enumerate(self.finger_names_):
            nj = self._chains_n_joints[fi]
            q0s[fi, :nj] = np.where(
                np.isnan(self.joint_angles[f]), 0.0, self.joint_angles[f])

        if self.objective == 'cartesian':
            n_ee = len(self.target_finger_chains[self.finger_names_[0]].ee_frames)
            dess = np.zeros((n_f, n_ee, 3))
            for fi, f in enumerate(self.finger_names_):
                dess[fi] = desired_positions[f]
            new_qs = np.array(self._jit_ik_vmap(
                jnp.array(q0s, dtype=jnp.float64),
                jnp.array(dess, dtype=jnp.float64)))
        else:
            mano_pts, masks = self._get_mano_contact_pts()
            new_qs = np.array(self._jit_mesh_ik_vmap(
                jnp.array(q0s, dtype=jnp.float64), mano_pts, masks))

        for fi, f in enumerate(self.finger_names_):
            nj = self._chains_n_joints[fi]
            result = new_qs[fi, :nj]
            if not np.any(np.isnan(result)):
                self.joint_angles[f] = result

        updated_fingers = set()
        if self.coupled_joints is not None:
            updated_fingers.update(self._average_coupled_joints())

        if self.post_embodiment_hook is not None:
            updated_fingers.update(self.post_embodiment_hook(self.joint_angles))

        for finger_name in updated_fingers:
            self.finger_forward_kinematics(
                finger_name, self.joint_angles[finger_name])

    def _average_coupled_joints(self):
        """Average joint angles of coupled joints that move together."""
        angle_sum = 0.0
        n_joints = 0
        for finger_name, joint_name in self.coupled_joints:
            if finger_name not in self.finger_names_:
                continue
            angle_sum += self.joint_angles[finger_name][
                self.target_finger_chains[
                    finger_name].joint_names.index(joint_name)]
            n_joints += 1

        if n_joints == 0:
            return

        average_angle = angle_sum / n_joints
        updated_fingers = set()
        for finger_name, joint_name in self.coupled_joints:
            self.joint_angles[finger_name][
                self.target_finger_chains[finger_name].joint_names.index(
                    joint_name)] = average_angle
            updated_fingers.add(finger_name)
        return updated_fingers

    def _update_hand_base_pose(self, handbase2world):
        """Compute pose of the base of the robotic target hand.

        Parameters
        ----------
        handbase2world : array, shape (4, 4)
            Transform from base of the robotic hand to world frame.
        """
        if handbase2world is None:
            world2robotbase = np.eye(4)
        else:
            world2robotbase = pt.concat(
                pt.invert_transform(handbase2world, check=False),
                self.handbase2robotbase)
        self.target_kin.tm.add_transform(
            "world", self.base_frame, world2robotbase)

        if self.vis_kin is not self.target_kin:
            self.vis_kin.tm.add_transform(
                "world", self.base_frame, world2robotbase)

    @property
    def transform_manager_(self):
        """Get transform manager."""
        return self.vis_kin.tm

    def finger_forward_kinematics(self, finger_name, joint_angles):
        """Forward kinematics for a finger of the target system.

        Parameters
        ----------
        finger_name : str
            Name of the finger.

        joint_angles : array-like, shape (n_joints,)
            Angles of the finger joints.

        Returns
        -------
        """
        return self.target_finger_chains[finger_name].forward(joint_angles)

    def solve_batch(self, pose_batch, handbase2world_batch=None):
        """Batch embodiment for T timesteps.

        Uses jax.lax.scan for O(1) memory overhead: one IK solve lives in JAX
        memory at a time. Warm-starts each step from the previous solution,
        which helps convergence on smooth trajectories.

        Parameters
        ----------
        pose_batch : array, shape (T, 48)
            MANO pose parameters per timestep.

        handbase2world_batch : array, shape (T, 4, 4), optional (default: None)
            Per-timestep transforms from MANO hand base to world. Not used
            in the current FK; provided for interface completeness.

        Returns
        -------
        joint_angles_batch : dict
            Maps finger name to array of shape (T, n_joints).
        """
        import jax
        import jax.numpy as jnp

        T = pose_batch.shape[0]
        desired = self._mano_fk_batch(pose_batch)  # finger → (T, n_ee, 3)

        result = {}
        for finger_name in self.finger_names_:
            chain = self.target_finger_chains[finger_name]
            des_jax = jnp.array(desired[finger_name], dtype=jnp.float64)
            init_q = jnp.zeros(chain.n_joints, dtype=jnp.float64)
            _, Q = jax.lax.scan(chain._ik_scan_step, init_q, des_jax)
            result[finger_name] = np.array(Q)

        return result

    def _mano_fk_batch(self, pose_batch):
        """Batch MANO FK for T timesteps → desired EE positions per finger.

        Parameters
        ----------
        pose_batch : array, shape (T, 48)

        Returns
        -------
        desired : dict
            finger_name → array, shape (T, n_ee, 3), in robot base frame.
        """
        import jax
        import jax.numpy as jnp
        from .mano_jax import build_hand_vertices_fn

        desired = {}
        for finger_name in self.finger_names_:
            mfk = self.mano_finger_kinematics[finger_name]
            fp = mfk.finger_pose_params
            idx = mfk.finger_pose_param_indices

            # Build (or reuse) a JAX FK function for this finger's sub-model
            if not hasattr(mfk, '_jax_fk_batch'):
                fn = build_hand_vertices_fn(fp['kintree_table'])
                mfk._jax_fk_batch = jax.jit(
                    jax.vmap(fn, in_axes=(None, None, None, None, 0)))

            # Construct optimizer pose batch: prepend 3 zeros (root joint)
            finger_poses = pose_batch[:, idx]               # (T, n_params)
            T_steps = finger_poses.shape[0]
            roots = np.zeros((T_steps, 3), dtype=np.float64)
            opt_poses = np.concatenate([roots, finger_poses], axis=1)  # (T, n_params+3)

            J = jnp.array(fp['J'], dtype=jnp.float64)
            weights = jnp.array(fp['weights'], dtype=jnp.float64)
            v_template = jnp.array(fp['v_template'], dtype=jnp.float64)
            posedirs = jnp.array(fp['posedirs'], dtype=jnp.float64)
            poses_jax = jnp.array(opt_poses, dtype=jnp.float64)

            # verts_batch: (T, n_finger_vertices, 3) in MANO hand base frame
            verts_batch = np.asarray(
                mfk._jax_fk_batch(J, weights, v_template, posedirs, poses_jax))

            # Take first n_ee vertices and transform to robot base frame
            n_ee = len(self.target_finger_chains[finger_name].ee_frames)
            markers = verts_batch[:, :n_ee, :]  # (T, n_ee, 3)

            R = self.handbase2robotbase[:3, :3]
            t = self.handbase2robotbase[:3, 3]
            desired[finger_name] = markers @ R.T + t[np.newaxis, np.newaxis, :]

        return desired


def load_kinematic_model(hand_config, unscaled_visual_model=True):
    """Load kinematic model of a robotic hand.

    Parameters
    ----------
    hand_config : dict
        Configuration of robotic target hand.

    unscaled_visual_model : bool, optional (default: True)
        Load an unscaled visual model of the hand in addition to the scaled
        version.

    Returns
    -------
    kin : Kinematics
        Forward and inverse kinematics.

    vis_kin : Kinematics
        Forward and inverse kinematics for visualization.
    """
    model = hand_config["model"]
    extra_args = {}
    if "package_dir" in model:
        extra_args["package_dir"] = model["package_dir"]
    if "mesh_path" in model:
        extra_args["mesh_path"] = model["mesh_path"]
    extra_args["scale"] = hand_config.get("scale", 1.0)
    with open(model["urdf"], "r") as f:
        kin = Kinematics(urdf=f.read(), **extra_args)
    if unscaled_visual_model:
        extra_args.pop("scale")
        with open(model["urdf"], "r") as f:
            vis_kin = Kinematics(urdf=f.read(), **extra_args)
    else:
        vis_kin = kin
    if "kinematic_model_hook" in model:
        kinematic_model_hook_args = hand_config.get("kinematic_model_hook_args", {})
        model["kinematic_model_hook"](kin, **kinematic_model_hook_args)
        if unscaled_visual_model:
            model["kinematic_model_hook"](vis_kin)
    if "virtual_joints_callbacks" in hand_config:
        for joint_name, callback in hand_config["virtual_joints_callbacks"].items():
            kin.tm.add_virtual_joint(joint_name, callback)
            if unscaled_visual_model:
                vis_kin.tm.add_virtual_joint(joint_name, callback)
    if "world" in kin.tm.nodes:
        warnings.warn(
            "'world' frame is already in URDF. Removing all connections to it.")
        invalid_connections = []
        for from_frame, to_frame in kin.tm.transforms:
            if "world" in (from_frame, to_frame):
                invalid_connections.append((from_frame, to_frame))
        for from_frame, to_frame in invalid_connections:
            kin.tm.remove_transform(from_frame, to_frame)
            if unscaled_visual_model:
                vis_kin.tm.remove_transform(from_frame, to_frame)
    return kin, vis_kin
