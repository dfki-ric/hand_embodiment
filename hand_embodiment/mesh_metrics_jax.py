"""Pure-JAX point-cloud distance metrics for the mesh-based embodiment IK.

All functions operate on float64 point clouds and are differentiable
w.r.t. pts_b (robot contact surface, which depends on joint angles).
Functions accept an optional float mask for pts_a to handle padded arrays
used when vmapping over fingers with different MANO contact-vertex counts.
"""
import os
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)


def _sq_dists(pts_a, pts_b):
    """Pairwise squared Euclidean distances.

    Parameters
    ----------
    pts_a : array, shape (M, 3)
    pts_b : array, shape (N, 3)

    Returns
    -------
    sq : array, shape (M, N)
    """
    diff = pts_a[:, None, :] - pts_b[None, :, :]   # (M, N, 3)
    return jnp.sum(diff ** 2, axis=-1)              # (M, N)


def chamfer_distance(pts_a, pts_b, mask_a=None):
    """Symmetric Chamfer distance between two point clouds.

    Uses logsumexp soft-min instead of hard min so that the gradient is
    Lipschitz-continuous w.r.t. both pts_b and the pose that generates
    pts_b.  Hard min produces piecewise-constant gradients that flip
    direction when the nearest-neighbour assignment changes between IK
    iterations or consecutive recording frames, causing oscillation.
    Temperature T = 1e-4 m² ≈ (1 cm)² matches the Hausdorff smooth-max
    temperature and keeps the approximation error < 1 mm.

    Parameters
    ----------
    pts_a : array, shape (M, 3)
        MANO contact surface (fixed per frame).
    pts_b : array, shape (N, 3)
        Robot contact surface (function of joint angles).
    mask_a : array, shape (M,), float, optional
        1.0 for valid MANO points, 0.0 for padding rows.

    Returns
    -------
    dist : scalar
        May be slightly negative when T is comparable to the squared distances
        (soft-min is always ≤ true min). This does not affect optimisation: the
        minimiser is the same as for hard chamfer; only the landscape is smoother.
    """
    sq = _sq_dists(pts_a, pts_b)    # (M, N)
    T = jnp.float64(1e-4)           # soft-min temperature in m²; ≈(1 cm)²

    # Soft-min per MANO point (nearest robot neighbour)
    softmin_a_b = -T * jax.nn.logsumexp(-sq / T, axis=1)   # (M,)

    if mask_a is not None:
        mean_a_b = jnp.sum(softmin_a_b * mask_a) / (jnp.sum(mask_a) + 1e-8)
        # Exclude padded MANO rows from the B→A soft-min by replacing their
        # distances with a large sentinel before taking the minimum over axis=0.
        sq_masked = jnp.where(
            mask_a[:, None] > 0.5, sq, jnp.full_like(sq, 1e10))
        softmin_b_a = -T * jax.nn.logsumexp(-sq_masked / T, axis=0)  # (N,)
    else:
        mean_a_b = jnp.mean(softmin_a_b)
        softmin_b_a = -T * jax.nn.logsumexp(-sq / T, axis=0)          # (N,)

    mean_b_a = jnp.mean(softmin_b_a)
    return (mean_a_b + mean_b_a) * 0.5


def hausdorff_distance(pts_a, pts_b, mask_a=None):
    """Smooth Hausdorff distance.

    The outer max is replaced by log-sum-exp (smooth max) to avoid the
    discontinuous gradient flips that occur when the maximising nearest-
    neighbour pair changes between optimiser iterations.  The approximation
    error is O(T * log N) ≈ 0.5 mm for T=1e-4 and N=100 points, which is
    negligible for IK purposes.

    Parameters
    ----------
    pts_a : array, shape (M, 3)
    pts_b : array, shape (N, 3)
    mask_a : array, shape (M,), float, optional

    Returns
    -------
    dist : scalar
    """
    sq = _sq_dists(pts_a, pts_b)   # (M, N)
    # Temperature for smooth max in sq-distance units (1e-4 m^2 ≈ (1 cm)^2).
    T = jnp.float64(1e-4)

    if mask_a is not None:
        sq_inf = jnp.where(mask_a[:, None] > 0.5, sq, jnp.full_like(sq, 1e10))
        min_a_b = jnp.min(sq_inf, axis=1)     # (M,)
        # Exclude padded rows from the soft max with a large negative sentinel.
        min_a_b_valid = jnp.where(
            mask_a > 0.5, min_a_b, jnp.full_like(min_a_b, -1e10))
        h_a_b = T * jax.nn.logsumexp(min_a_b_valid / T)
        min_b_a = jnp.min(sq_inf, axis=0)     # (N,)
    else:
        min_a_b = jnp.min(sq, axis=1)
        h_a_b = T * jax.nn.logsumexp(min_a_b / T)
        min_b_a = jnp.min(sq, axis=0)

    h_b_a = T * jax.nn.logsumexp(min_b_a / T)
    return jnp.sqrt(jnp.maximum(h_a_b, h_b_a) + 1e-12)


def sinkhorn_distance(pts_a, pts_b, eps=0.01, n_iter=50, mask_a=None):
    """Entropic-regularised Earth-Mover's (Sinkhorn) distance.

    Log-domain Sinkhorn iterations with uniform marginals avoid overflow.
    When mask_a is provided, the MANO marginal is renormalised to the valid
    points only; padding rows are assigned log-weight -inf (excluded).

    Parameters
    ----------
    pts_a : array, shape (M, 3)
    pts_b : array, shape (N, 3)
    eps : float
        Regularisation strength; smaller values → closer to true EMD.
    n_iter : int
        Number of Sinkhorn iterations (static for lax.scan unrolling).
    mask_a : array, shape (M,), float, optional

    Returns
    -------
    dist : scalar
    """
    M_cost = jnp.sqrt(_sq_dists(pts_a, pts_b) + 1e-12)   # (M, N)
    M_pts = pts_a.shape[0]
    N_pts = pts_b.shape[0]

    if mask_a is not None:
        n_valid = jnp.sum(mask_a)
        log_a = jnp.where(
            mask_a > 0.5,
            -jnp.log(n_valid + 1e-8),
            jnp.full(M_pts, -1e10, dtype=jnp.float64))
    else:
        log_a = jnp.full(M_pts, -jnp.log(jnp.float64(M_pts)), dtype=jnp.float64)

    log_b = jnp.full(N_pts, -jnp.log(jnp.float64(N_pts)), dtype=jnp.float64)
    K = -M_cost / eps   # (M, N) log-kernel

    def step(f, _):
        g = eps * (log_b - jax.nn.logsumexp(K + f[:, None], axis=0))
        f_new = eps * (log_a - jax.nn.logsumexp(K + g[None, :], axis=1))
        return f_new, None

    f0 = jnp.zeros(M_pts, dtype=jnp.float64)
    f, _ = jax.lax.scan(step, f0, jnp.arange(n_iter))
    g = eps * (log_b - jax.nn.logsumexp(K + f[:, None], axis=0))

    return jnp.sum(jnp.exp(log_a) * f) + jnp.sum(jnp.exp(log_b) * g)


def fscore(pts_a, pts_b, threshold=0.01, tau=1e-4, mask_a=None):
    """Soft (differentiable) F-score for use as an IK objective.

    Uses a sigmoid to approximate the hard threshold, yielding smooth
    gradients. Returns the *negated* F-score so that minimisation
    corresponds to maximising the true contact overlap.

    Parameters
    ----------
    pts_a : array, shape (M, 3)
    pts_b : array, shape (N, 3)
    threshold : float
        Distance threshold for a "match" (metres).
    tau : float
        Sigmoid temperature; smaller → harder (less smooth) threshold.
    mask_a : array, shape (M,), float, optional

    Returns
    -------
    neg_fscore : scalar
        Negated F-score in [−1, 0]; minimising gives higher F-score.
    """
    sq = _sq_dists(pts_a, pts_b)   # (M, N)
    thr2 = jnp.float64(threshold ** 2)
    tau_v = jnp.float64(tau)
    # Soft-min temperature for nearest-neighbour lookup (same as chamfer/hausdorff).
    T = jnp.float64(1e-4)

    if mask_a is not None:
        sq_masked = jnp.where(
            mask_a[:, None] > 0.5, sq, jnp.full_like(sq, 1e10))
        softmin_a_b = -T * jax.nn.logsumexp(-sq_masked / T, axis=1)  # (M,)
        softmin_b_a = -T * jax.nn.logsumexp(-sq_masked / T, axis=0)  # (N,)
    else:
        softmin_a_b = -T * jax.nn.logsumexp(-sq / T, axis=1)   # (M,)
        softmin_b_a = -T * jax.nn.logsumexp(-sq / T, axis=0)   # (N,)

    prec_vals = jax.nn.sigmoid((thr2 - softmin_a_b) / tau_v)   # (M,)
    rec_vals = jax.nn.sigmoid((thr2 - softmin_b_a) / tau_v)    # (N,)

    if mask_a is not None:
        precision = jnp.sum(prec_vals * mask_a) / (jnp.sum(mask_a) + 1e-8)
    else:
        precision = jnp.mean(prec_vals)
    recall = jnp.mean(rec_vals)

    f = 2.0 * precision * recall / (precision + recall + 1e-8)
    return -f   # negate: minimising → higher F-score


METRIC_FNS = {
    "chamfer": chamfer_distance,
    "hausdorff": hausdorff_distance,
    "sinkhorn": sinkhorn_distance,
    "fscore": fscore,
}
