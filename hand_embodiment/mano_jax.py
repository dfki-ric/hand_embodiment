"""Pure JAX MANO forward kinematics — JIT and vmap compatible."""
import os
os.environ.setdefault("XLA_FLAGS", "--xla_gpu_enable_command_buffer=")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("JAX_COMPILATION_CACHE_DIR",
                      os.path.expanduser("~/.cache/hand_embodiment/jax_cache"))
import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from jaxtransform3d.rotations import matrix_from_compact_axis_angle
from jaxtransform3d.transformations import compose_transforms, create_transform


def build_hand_vertices_fn(kintree_table):
    """Create a jit/vmap-compatible hand_vertices function for a fixed topology.

    Parameters
    ----------
    kintree_table : array, shape (2, n_parts)
        Kinematic tree table. Must be a concrete (non-traced) numpy array since
        the tree topology is extracted once as Python constants.

    Returns
    -------
    fn : callable
        hand_vertices(J, weights, v_template, posedirs, pose) → (n_vertices, 3)
        Compatible with jax.jit and jax.vmap over the pose axis.
    """
    kintree_np = np.asarray(kintree_table)
    n_parts = kintree_np.shape[1]
    id_to_col = {int(kintree_np[1, i]): i for i in range(n_parts)}
    parent = [id_to_col[int(kintree_np[0, i])] for i in range(1, n_parts)]

    def _lrotmin(pose):
        return jnp.concatenate([
            (matrix_from_compact_axis_angle(pose[i]) - jnp.eye(3)).ravel()
            for i in range(1, n_parts)])

    def _global_rigid_transformation(pose, J):
        results = [None] * n_parts
        results[0] = create_transform(matrix_from_compact_axis_angle(pose[0]), J[0])
        for i in range(1, n_parts):
            T_local = create_transform(
                matrix_from_compact_axis_angle(pose[i]),
                J[i] - J[parent[i - 1]])
            results[i] = compose_transforms(results[parent[i - 1]], T_local)

        def _subtract_rest(i):
            j_h = jnp.append(J[i], 0.0)
            col = results[i] @ j_h
            return results[i].at[:, 3].add(-col)

        return jnp.stack([_subtract_rest(i) for i in range(n_parts)])

    def fn(J, weights, v_template, posedirs, pose):
        """Compute hand mesh vertices from pose parameters.

        Parameters
        ----------
        J : array, shape (n_parts, 3)
        weights : array, shape (n_vertices, n_parts)
        v_template : array, shape (n_vertices, 3)
        posedirs : array, shape (n_vertices, 3, 9 * (n_parts - 1))
        pose : array, shape (n_parts * 3,)

        Returns
        -------
        vertices : array, shape (n_vertices, 3)
        """
        pose_r = pose.reshape(n_parts, 3)
        v_posed = v_template + posedirs @ _lrotmin(pose_r)
        A = _global_rigid_transformation(pose_r, J)            # (n_parts, 4, 4)
        T_v = jnp.einsum('vk,kij->vij', weights, A)            # (n_vertices, 4, 4)
        rest_h = jnp.concatenate(
            [v_posed, jnp.ones((v_posed.shape[0], 1))], axis=1)  # (n_vertices, 4)
        return jnp.einsum('vij,vj->vi', T_v, rest_h)[:, :3]

    return fn
