"""Fast implementation of MANO kinematics."""
import numpy as np
cimport numpy as np
cimport cython
from libc.math cimport sqrt, cos, sin


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cpdef hand_vertices(
        np.ndarray[double, ndim=2] J,
        np.ndarray[double, ndim=2] weights,
        np.ndarray[long, ndim=2] kintree_table,
        np.ndarray[double, ndim=2] v_template,
        np.ndarray[double, ndim=3] posedirs,
        np.ndarray[double, ndim=1] pose):
    cdef np.ndarray[double, ndim=2] pose_reshaped = pose.reshape(-1, 3)
    cdef np.ndarray[double, ndim=2] v_posed = v_template + posedirs.dot(lrotmin(pose_reshaped))

    return forward_kinematic(pose_reshaped, v_posed, J, weights, kintree_table)

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef np.ndarray[double, ndim=1] lrotmin(np.ndarray[double, ndim=2] p):
    cdef np.ndarray[double, ndim=3] out = np.empty((len(p) - 1, 3, 3))
    cdef int i
    for i in range(0, len(out)):
        out[i] = _fast_matrix_from_axis_angle(p[i + 1])
        out[i, 0, 0] -= 1.0
        out[i, 1, 1] -= 1.0
        out[i, 2, 2] -= 1.0
    return out.reshape(-1)


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef np.ndarray[double, ndim=2] _fast_matrix_from_axis_angle(np.ndarray[double, ndim=1] a):
    cdef double angle = sqrt(a[0] * a[0] + a[1] * a[1] + a[2] * a[2])
    if angle == 0.0:
        return np.eye(3)

    cdef double ux = a[0] / angle
    cdef double uy = a[1] / angle
    cdef double uz = a[2] / angle

    cdef double c = cos(angle)
    cdef double s = sin(angle)
    cdef double ci = 1.0 - c
    cdef double ciux = ci * ux
    cdef double ciuy = ci * uy
    cdef double ciuz = ci * uz
    cdef np.ndarray[double, ndim=2] out = np.empty((3, 3))
    out[0, 0] = ciux * ux + c
    out[0, 1] = ciux * uy - uz * s
    out[0, 2] = ciux * uz + uy * s
    out[1, 0] = ciuy * ux + uz * s
    out[1, 1] = ciuy * uy + c
    out[1, 2] = ciuy * uz - ux * s
    out[2, 0] = ciuz * ux - uy * s
    out[2, 1] = ciuz * uy + ux * s
    out[2, 2] = ciuz * uz + c
    return out


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef np.ndarray[double, ndim=2] forward_kinematic(
        np.ndarray[double, ndim=2] pose,
        np.ndarray[double, ndim=2] v,
        np.ndarray[double, ndim=2] J,
        np.ndarray[double, ndim=2] weights,
        np.ndarray[long, ndim=2] kintree_table):
    cdef np.ndarray[double, ndim=3] A = global_rigid_transformation(pose, J, kintree_table)
    cdef np.ndarray[double, ndim=3] T = np.einsum("kij,lk->lji", A, weights)

    cdef np.ndarray[double, ndim=2] v_transformed = (
        T[:, 0, :] * v[:, 0, np.newaxis] +
        T[:, 1, :] * v[:, 1, np.newaxis] +
        T[:, 2, :] * v[:, 2, np.newaxis] +
        T[:, 3, :])[:, :3]

    return v_transformed


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
cdef np.ndarray[double, ndim=3] global_rigid_transformation(np.ndarray[double, ndim=2] pose, np.ndarray[double, ndim=2] J, np.ndarray[long, ndim=2] kintree_table):
    cdef int n_parts = kintree_table.shape[1]
    cdef int i

    cdef dict id_to_col = {kintree_table[1, i]: i for i in range(kintree_table.shape[1])}
    cdef dict parent = {i: id_to_col[kintree_table[0, i]] for i in range(1, kintree_table.shape[1])}

    cdef np.ndarray[double, ndim=3] results = np.empty((n_parts, 4, 4))
    _fast_transform_from(pose[0, :], J[0, :], results[0])
    cdef np.ndarray[double, ndim=2] T = np.empty((4, 4))
    for i in range(1, n_parts):
        _fast_transform_from(pose[i, :], J[i] - J[parent[i]], T)
        results[i] = results[parent[i]].dot(T)

    for i in range(n_parts):
        results[i, :3, 3] -= results[i, :3, :3].dot(J[i])
    return results


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
cdef _fast_transform_from(np.ndarray[double, ndim=1] a, np.ndarray[double, ndim=1] p, np.ndarray[double, ndim=2] out):
    out[:3, :3] = _fast_matrix_from_axis_angle(a)
    out[0, 3] = p[0]
    out[1, 3] = p[1]
    out[2, 3] = p[2]
    out[3, 0] = 0.0
    out[3, 1] = 0.0
    out[3, 2] = 0.0
    out[3, 3] = 1.0
