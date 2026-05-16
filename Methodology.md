# Methodology: JAX-Based Optimization for Real-Time Hand Embodiment

We use $\hat{}$ for estimated or measured quantities and $\tilde{}$ for
configuration parameters, following the notation of Fabisch et al. [1].

---

## Overview

In the original framework [1], both the record mapping (MANO IK) and the
embodiment mapping (robotic IK) were solved with Sequential Least Squares
Programming (SLSQP), using numerically estimated gradients.  This approach
suffered from two practical deficiencies: (i) finite-difference Jacobians
introduce approximation error and require $O(n)$ forward-kinematics
evaluations per Newton step; (ii) SLSQP solves each finger sequentially in
Python, leaving the other four fingers idle on the CPU.

We replace both solvers with a **Levenberg–Marquardt (LM) optimizer**
whose Jacobian is computed by **forward-mode automatic differentiation** and
whose per-finger problems are solved **simultaneously** via functional
vectorisation.  The implementation is based on JAX [2], a Python library
for composable function transformations (JIT compilation, automatic
differentiation, and vectorisation over arrays) with an XLA backend that
targets both CPU and GPU.

---

## 1. Record Mapping: MANO Inverse Kinematics

### 1.1 Residual Formulation

As in [1], the pose of the MANO model in world frame is

$$T_{\texttt{world},\texttt{MANO}} = \hat{T}_{\texttt{world},\texttt{hand}}\, \tilde{T}_{\texttt{hand},\texttt{MANO}},$$

and estimating each finger's joint angles $\boldsymbol{q}_i \in \mathbb{R}^9$
reduces to five independent least-squares problems.  We reformulate these as
nonlinear least-squares problems by stacking a **residual vector**

$$\boldsymbol{r}(\boldsymbol{q}_i)
= \begin{bmatrix}
    f_{\tilde{\boldsymbol{\beta}},i,1}(\boldsymbol{q}_i) - \hat{\boldsymbol{p}}_{i,1} \\
    f_{\tilde{\boldsymbol{\beta}},i,2}(\boldsymbol{q}_i) - \hat{\boldsymbol{p}}_{i,2} \\
    \tilde{\boldsymbol{w}}_{i,+} \circ \max(\boldsymbol{q}_i,\, \boldsymbol{0}) \\
    \tilde{\boldsymbol{w}}_{i,-} \circ \max(-\boldsymbol{q}_i,\, \boldsymbol{0})
  \end{bmatrix}
\in \mathbb{R}^{6 + 2 \times 9},$$

where $f_{\tilde{\boldsymbol{\beta}},i,j}$ is MANO forward kinematics for
vertex $j$ of finger $i$, $\hat{\boldsymbol{p}}_{i,j}$ are the measured
marker positions, and $\tilde{\boldsymbol{w}}_{i,+},
\tilde{\boldsymbol{w}}_{i,-} \in \mathbb{R}^9$ are per-DOF regularisation
weights ($\circ$ denotes the Hadamard product).  Minimising
$\frac{1}{2}\|\boldsymbol{r}(\boldsymbol{q}_i)\|^2$ recovers the objective
in [1].

> **Why component-wise, not dot-product?**  The earlier formulation used
> $\|\tilde{\boldsymbol{w}}_{i,+}^T \boldsymbol{q}_i^+\|^2$ (a scalar
> squared), which couples all DOFs in the gradient: the gradient of DOF $k$
> receives contributions from every other DOF through the product.  For the
> thumb, whose weights $\tilde{\boldsymbol{w}}$ span two orders of magnitude,
> this cross-coupling amplifies the effective regularisation gradient
> approximately 19-fold relative to the temporal-smoothing gradient, causing
> the optimizer to oscillate between local minima.  The component-wise form
> above yields independent gradients $\partial r / \partial q_k =
> \tilde{w}_k\, \text{sign}(q_k)$, removing this coupling.

### 1.2 Levenberg–Marquardt Iterations

Let $\mathbf{J}_k = \partial \boldsymbol{r}/\partial \boldsymbol{q}\big|_{\boldsymbol{q}^{(k)}}$
be the Jacobian at iteration $k$.  Each LM step solves the augmented
normal equations

$$\left(\mathbf{J}_k^T \mathbf{J}_k + (\lambda_k + \omega_s)\mathbf{I}\right)
  \Delta\boldsymbol{q}^{(k)}
= -\mathbf{J}_k^T \boldsymbol{r}^{(k)} - \omega_s \!\left(\boldsymbol{q}^{(k)} - \boldsymbol{q}^{(0)}\right),$$

and updates

$$\boldsymbol{q}^{(k+1)} = \operatorname{clip}\!\left(
  \boldsymbol{q}^{(k)} + \operatorname{clip}_{\delta}\bigl(\Delta\boldsymbol{q}^{(k)}\bigr),\;
  \tilde{\boldsymbol{q}}^{\min}_i,\; \tilde{\boldsymbol{q}}^{\max}_i\right),$$

where $\operatorname{clip}_{\delta}(\Delta) = \Delta \cdot
\min\!\bigl(1,\, \delta_{\max}/\|\Delta\|\bigr)$ limits the step norm to
$\delta_{\max} = 0.2\,\text{rad}$ to prevent divergence when marker
targets are unreachable.

The damping $\lambda_k$ follows an exponential schedule

$$\lambda_k = \exp\!\left(\log\lambda_0 + \frac{k}{K-1}(\log\lambda_K - \log\lambda_0)\right),$$

with $\lambda_0 = 0.05$, $\lambda_K = 10^{-3}$, $K = 15$ iterations.
The schedule begins in the gradient-descent regime (large $\lambda$),
which prevents divergence for large initial residuals, and transitions to
the quasi-Newton regime (small $\lambda$) for fine convergence.

### 1.3 Temporal Smoothing

The term $\omega_s (\boldsymbol{q}^{(k)} - \boldsymbol{q}^{(0)})$ in
the right-hand side, together with the $\omega_s \mathbf{I}$ augmentation
of the Hessian approximation, implements **temporal smoothing**.  It is
equivalent to adding an extra residual block
$\sqrt{\omega_s}(\boldsymbol{q} - \boldsymbol{q}^{(0)})$ without
enlarging the Jacobian, where $\boldsymbol{q}^{(0)}$ is the solution
accepted at the previous time step.  With $\omega_s = 2 \times 10^{-3}$,
the optimizer is penalised for large departures from the previous frame,
which suppresses jumps between local minima caused by partial marker
occlusion or noise.  To prevent NaN contamination across frames, we
sanitise the warm start as

$$\boldsymbol{q}^{(0)} \leftarrow
  \boldsymbol{q}^{(0)} \odot \lnot \operatorname{isnan}(\boldsymbol{q}^{(0)}),$$

and write back the result only if it contains no NaN values.

### 1.4 Automatic Differentiation

The Jacobian $\mathbf{J}_k$ is computed by **forward-mode automatic
differentiation** (`jax.jacfwd`).  Unlike finite differences, which require
$n = 9$ additional MANO forward passes per iteration to approximate
$\mathbf{J}_k$, `jacfwd` propagates $n$ dual-number tangent vectors
through the computation graph in a single forward pass, yielding the exact
Jacobian at the cost of $O(n)$ floating-point operations — with no
approximation error and no step-size hyperparameter.  For the MANO model,
this halves the effective per-iteration work compared to the two-sided
finite-difference stencil used by scipy's SLSQP.

### 1.5 Parallel Solution over Fingers via `jax.vmap`

All five MANO finger kinematics share the same kinematic topology (a
4-joint linear chain) and identical array shapes:
$\mathbf{J}_f \in \mathbb{R}^{4 \times 3}$,
$\mathbf{W}_f \in \mathbb{R}^{2 \times 4}$,
$\boldsymbol{v}^0_f \in \mathbb{R}^{2 \times 3}$,
$\mathbf{P}_f \in \mathbb{R}^{2 \times 3 \times 27}$.
This enables expressing the per-finger LM solve as a pure function

$$\phi\!\left(\boldsymbol{q}^{(0)},\, \hat{\mathbf{P}},\, \boldsymbol{m},\,
  \mathbf{J}_f,\, \mathbf{W}_f,\, \boldsymbol{v}^0_f,\, \mathbf{P}_f,\,
  \tilde{\boldsymbol{w}}_f\right)
\mapsto
\left(\boldsymbol{q}^*_f,\, \boldsymbol{v}^*_f\right),$$

where $\boldsymbol{m} \in \{0,1\}^2$ is a per-finger marker mask
(entries are 1 for available markers, 0 for missing ones), and applying
`jax.vmap` over the leading finger axis to obtain

$$(\boldsymbol{q}^*_1, \ldots, \boldsymbol{q}^*_5),\;
  (\boldsymbol{v}^*_1, \ldots, \boldsymbol{v}^*_5)
= \texttt{vmap}(\phi)\!\left(
    [\boldsymbol{q}^{(0)}_1,\ldots],\,
    [\hat{\mathbf{P}}_1,\ldots],\,
    [\boldsymbol{m}_1,\ldots],\,
    [\mathbf{J}_1,\ldots],\,\ldots
  \right).$$

`jax.vmap` transforms $\phi$ into a batched function that XLA executes as a
single fused kernel, letting the GPU solve all five fingers in parallel
rather than sequentially.  The entire vmapped solve is further wrapped in
`jax.jit`, which traces the computation once at initialisation and caches
the compiled XLA program for subsequent frames.

The 15-iteration loop over $\lambda_k$ is expressed as `jax.lax.scan`,
which unrolls the fixed-length loop into a single XLA `while`-free program
and enables kernel fusion across all iterations.

---

## 2. Embodiment Mapping: Robotic Inverse Kinematics

The embodiment mapping solves the same class of problem: given desired
end-effector positions $f_{\tilde{\boldsymbol{\beta}},i,j}(\boldsymbol{q}_i^*)$
from the record mapping, find joint angles $\boldsymbol{r}_i$ of the
robotic hand that minimise

$$\boldsymbol{r}_i^* = \operatorname*{arg\,min}_{\boldsymbol{r}_i}
  \sum_{j=1}^{2}
  \left\|f_{\tilde{\boldsymbol{\beta}},i,j}(\boldsymbol{q}_i^*) - g_{i,j}(\boldsymbol{r}_i)\right\|^2$$

subject to $\tilde{\boldsymbol{r}}^{\min}_i \leq \boldsymbol{r}_i \leq \tilde{\boldsymbol{r}}^{\max}_i$,
where $g_{i,j}$ is the forward kinematics of the robotic finger.

We apply the same LM solver structure:

$$\left(\mathbf{G}_k^T \mathbf{G}_k + (\mu_k + \omega_s)\mathbf{I}\right)
  \Delta\boldsymbol{r}^{(k)}
= \mathbf{G}_k^T \boldsymbol{e}^{(k)} - \omega_s\!\left(\boldsymbol{r}^{(k)} - \boldsymbol{r}^{(0)}\right),$$

where $\boldsymbol{e}^{(k)} = (g_{i,j}(\boldsymbol{r}^{(k)}) - \text{desired}).\text{ravel}()$
and $\mathbf{G}_k = \partial\, \boldsymbol{e}/\partial\,\boldsymbol{r}$ via `jax.jacfwd`.
The robotic FK $g_{i,j}$ is implemented as a pure JAX function by unrolling
the kinematic chain (sequence of fixed transforms and axis-angle rotations)
into a static Python loop at trace time, so the entire chain evaluates as a
single fused XLA computation.

For the Shadow Dexterous Hand, the damping schedule uses $\mu_0 = 10^{-2}$,
$\mu_K = 10^{-4}$, $K = 15$ iterations, with step clamp
$\delta_{\max} = 0.15\,\text{rad}$ and temporal smoothing $\omega_s = 10^{-3}$.
The schedule keeps $\mu_k$ above $\approx \|\mathbf{G}^T\mathbf{G}\|$ throughout,
preventing Newton-like overshoots caused by the shorter link lengths of the
robotic hand (Jacobian scale $\approx 0.01$–$0.04$ m/rad, giving
$\mathbf{G}^T\mathbf{G} \approx 10^{-4}$–$1.6\times10^{-3}$).

All fingers are solved in parallel via `jax.vmap` over a batched FK
function parameterised by the per-finger kinematic data (joint axes,
transforms, limits), compiled once at initialisation with `jax.jit`.

---

## 3. Summary of Transformations

The following JAX functional transforms are composed to build the final
solver:

| Transform | Role |
|-----------|------|
| `jax.jacfwd` | Exact Jacobian of the residual via forward-mode AD |
| `jax.lax.scan` | Fixed 15-step LM loop, fused by XLA |
| `jax.vmap` | Parallel execution over all 5 fingers in one kernel |
| `jax.jit` | Ahead-of-time XLA compilation; amortised over all frames |

Because all four transforms operate on pure functions without side effects,
they can be freely composed: `jax.jit(jax.vmap(lax.scan(jacfwd(...))))`.
The resulting program contains no Python overhead in the inner loop.

---

## 4. Computational Performance

Measurements on an NVIDIA GPU with the Shadow Dexterous Hand as target:

| Stage | This work | Fabisch et al. [1] (mean) |
|---|---|---|
| MANO record mapping (5 fingers) | ~11 ms | ~10–15 ms |
| Robotic embodiment (Shadow, 5 fingers) | ~14 ms | ~5–8 ms |
| **Total pipeline** | **~25 ms (~40 Hz)** | **~15–23 ms** |

The mean total throughput is comparable to the original CPU-based SLSQP
implementation, but the **minimum throughput is qualitatively different**:
the original solver exhibited per-frame times exceeding one second (Table VII
in [1], minimum Shadow frequency 0.7 Hz), caused by SLSQP convergence
failures.  The JAX LM solver has bounded, deterministic per-frame cost
(fixed $K = 15$ iterations), eliminating these tail latencies.  This
property is essential for teleoperation, where occasional slow frames
disrupt the control loop.

---

## References

[1] A. Fabisch, M. Uliano, D. Marschner, M. Laux, J. Brust, M. Controzzi,
    "A Modular Approach to the Embodiment of Hand Motions from Human
    Demonstrations," *IEEE-RAS International Conference on Humanoid Robots*,
    2022.

[2] J. Bradbury, R. Frostig, P. Hawkins, M. J. Johnson, C. Leary,
    D. Maclaurin, G. Necula, A. Paszke, J. VanderPlas, S. Wanderman-Milne,
    Q. Zhang, "JAX: composable transformations of Python+NumPy programs,"
    2018, http://github.com/google/jax.
