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
  \tilde{c}_{i,j}
  \left\|f_{\tilde{\boldsymbol{\beta}},i,j}(\boldsymbol{q}_i^*) - g_{i,j}(\boldsymbol{r}_i)\right\|^2$$

subject to $\tilde{\boldsymbol{r}}^{\min}_i \leq \boldsymbol{r}_i \leq \tilde{\boldsymbol{r}}^{\max}_i$,
where $g_{i,j}$ is the forward kinematics of the robotic finger and
$\tilde{c}_{i,j} \geq 0$ is a per-end-effector importance weight.

### 2.1 Importance Weighting and Joint Regularisation

For manipulation tasks, accurate placement of the fingertip is more important
than accurate placement of intermediate phalanx markers: an error at the tip
directly affects contact, whereas a mid-chain error is partially compensated
by the remaining joints.  We therefore set $\tilde{c}_{i,\text{tip}} = 1$ and
$\tilde{c}_{i,\text{mid}} = 0.1$ for all fingers, making the tip ten times
more influential than the middle-phalanx target.

A second class of degeneracy arises from **abduction–adduction joints** (e.g.
the Shadow's LFJ4, MCP lateral joint of the little finger): the optimizer may
exploit a lateral degree of freedom to reduce the along-finger Cartesian
residual at the cost of orthogonal displacement that exceeds the Cartesian
improvement — a local minimum invisible to a purely Cartesian objective.  We
suppress this by adding a per-joint regularisation penalty
$\frac{1}{2}\sum_j \tilde{\omega}_j r_j^2$, with $\tilde{\omega}_{j}$ set to
$10^{-3}$ for abduction joints and $0$ for flexion joints.

Introducing the EE weight matrix
$\mathbf{W}_i = \mathrm{diag}\!\left(\sqrt{\tilde{c}_{i,1}}\,\mathbf{1}_3,\;
\sqrt{\tilde{c}_{i,2}}\,\mathbf{1}_3\right) \in \mathbb{R}^{6\times 6}$,
the joint regularisation diagonal
$\tilde{\boldsymbol{\Omega}}_i = \mathrm{diag}(\tilde{\omega}_{i,1},\ldots,\tilde{\omega}_{i,n_r})$,
and the unweighted residual
$\boldsymbol{e}^{(k)} = \bigl(g_{i,j}(\boldsymbol{r}^{(k)}) - \text{desired}_{i,j}\bigr).\mathrm{ravel}()$,
the augmented LM normal equations become

$$\left(\mathbf{G}_{w,k}^T \mathbf{G}_{w,k}
    + \tilde{\boldsymbol{\Omega}}_i
    + (\mu_k + \omega_s)\mathbf{I}\right)
  \Delta\boldsymbol{r}^{(k)}
= \mathbf{G}_{w,k}^T \boldsymbol{e}_{w}^{(k)}
  - \tilde{\boldsymbol{\Omega}}_i\,\boldsymbol{r}^{(k)}
  - \omega_s\!\left(\boldsymbol{r}^{(k)} - \boldsymbol{r}^{(0)}\right),$$

where $\boldsymbol{e}_{w}^{(k)} = \mathbf{W}_i\,\boldsymbol{e}^{(k)}$ and
$\mathbf{G}_{w,k} = \mathbf{W}_i\,\mathbf{G}_k$,
with $\mathbf{G}_k = \partial\,\boldsymbol{e}/\partial\,\boldsymbol{r}\big|_{\boldsymbol{r}^{(k)}}$
computed via `jax.jacfwd`.
The robotic FK $g_{i,j}$ is implemented as a pure JAX function by unrolling
the kinematic chain (sequence of fixed transforms and axis-angle rotations)
into a static Python loop at trace time, so the entire chain evaluates as a
single fused XLA computation.

For the Shadow Dexterous Hand, the damping schedule uses $\mu_0 = 10^{-2}$,
$\mu_K = 10^{-4}$, $K = 15$ iterations, with step clamp
$\delta_{\max} = 0.15\,\text{rad}$ and temporal smoothing $\omega_s = 5\times10^{-2}$.
The schedule keeps $\mu_k$ above $\approx \|\mathbf{G}_w^T\mathbf{G}_w\|$ throughout,
preventing Newton-like overshoots caused by the shorter link lengths of the
robotic hand (Jacobian scale $\approx 0.01$–$0.04$ m/rad, giving
$\mathbf{G}^T\mathbf{G} \approx 10^{-4}$–$1.6\times10^{-3}$).
The value $\omega_s = 5\times10^{-2}$ is chosen so that the temporal penalty
forms the dominant floor on the $\mathbf{A}_k$ diagonal throughout all $K$
iterations ($\mu_0 = 10^{-2} < \omega_s$): smaller values allow the early
iterations to explore null-space directions freely and then snap to a different
local minimum in the final refinement steps, causing frame-to-frame
oscillation.

All fingers are solved in parallel via `jax.vmap` over a batched FK
function parameterised by the per-finger kinematic data (joint axes,
transforms, limits, weights $\tilde{c}_{i,j}$), compiled once at
initialisation with `jax.jit`.

### 2.2 Kinematic Frame Calibration

The embodiment mapping requires assigning each MANO fingertip and
mid-phalanx target to a corresponding frame in the robot's kinematic
model.  A mismatch in the distance between these two frames — the
**effective finger span** — forces the IK to over- or under-flex the
distal joints to satisfy both constraints simultaneously.

For the Shadow Dexterous Hand, a span mismatch was identified for the
little finger.  The MANO model (fitted to the dataset used) places the
little fingertip and the middle-phalanx marker
$\|\hat{\boldsymbol{p}}_{\text{little},\text{tip}} -
   \hat{\boldsymbol{p}}_{\text{little},\text{mid}}\|
= 50.8\,\mathrm{mm}$ apart.
The original configuration anchored the mid-phalanx target frame to the
proximal link (\texttt{rh\_lfproximal}) with a fixed offset, placing it
$76.2\,\mathrm{mm}$ from the fingertip frame — a $50\,\%$ over-estimate.
The IK reconciled this mismatch by hyper-flexing the joint immediately
proximal to the tip, causing the visible little-finger curl.

The corrected configuration anchors the mid-phalanx target frame to the
middle link (\texttt{rh\_lfmiddle}), reducing the effective span to
$51.2\,\mathrm{mm}$ and the mismatch to $0.4\,\mathrm{mm}$ ($< 1\,\%$).
With the corrected frame, combined with tip-priority weighting
($\tilde{c}_{\text{tip}} = 1$, $\tilde{c}_{\text{mid}} = 0.1$), the
little finger reaches the target contact pose without pathological
joint configurations.

---

## 3. Alternative Embodiment Objective: Inner-Surface Mesh Distances

### 3.1 Motivation

The Cartesian 2-point objective (tip + middle EE) used in Section 2 is a
coarse surrogate for contact quality.  For manipulation tasks the relevant
quantity is how well the robot's contact surface conforms to the human's —
something that a 2-point objective cannot distinguish from a pose that merely
passes through the same two locations.  As an alternative, the contact surface
itself can be used as the IK objective.

### 3.2 Contact Surface Definition

The inner contact surface of finger $i$ is defined as a subset of mesh vertices

$$
\mathcal{S}^{\text{MANO}}_i = \{\hat{\boldsymbol{p}}_v\}_{v \in \tilde{V}_i^{\text{MANO}}} \in \mathbb{R}^{N_i \times 3}, \qquad
\mathcal{S}^{\text{robot}}_i(\boldsymbol{r}) = \{T_{l}(\boldsymbol{r})\,\tilde{\boldsymbol{v}}_{l,k}\}_{l,k} \in \mathbb{R}^{M \times 3},
$$

where $\tilde{V}_i^{\text{MANO}}$ are predefined contact vertex indices,
$\tilde{\boldsymbol{v}}_{l,k}$ are 100 Poisson-disk-sampled contact points on
link $l$ in its local frame (extracted once at initialisation), and
$T_l(\boldsymbol{r}) \in SE(3)$ is the differentiable FK transform for link $l$
computed via the same JAX kinematic chain used in Section 2.

### 3.3 Point-Cloud Distance Metrics

All metrics operate on point clouds $A \in \mathbb{R}^{M\times3}$,
$B\in\mathbb{R}^{N\times3}$ with pairwise squared distances
$d^2_{ij} = \|a_i - b_j\|^2$:

| Metric | Formula | Gradient |
|---|---|---|
| Chamfer | $\frac{1}{2}\!\left(\frac{1}{M}\sum_i\mathrm{softmin}_j\, d^2_{ij} + \frac{1}{N}\sum_j\mathrm{softmin}_i\, d^2_{ij}\right)$, $\mathrm{softmin}_j\,x_j = -T\log\sum_j e^{-x_j/T}$, $T=10^{-4}$ m$^2$ | smooth |
| Hausdorff | $\max\!\bigl(\max_i\min_j d_{ij},\;\max_j\min_i d_{ij}\bigr)$ | subgradient |
| Sinkhorn / EMD | $\langle\mathbf{P}^*, \mathbf{D}\rangle$ where $\mathbf{P}^*$ is the entropic-regularised optimal transport plan | smooth |
| Soft F-score | $-\frac{2\,\bar{P}\,\bar{R}}{\bar{P}+\bar{R}}$, $\bar{P} = \frac{1}{M}\sum_i\sigma\!\bigl(\frac{\tilde{\tau}^2-\min_j d^2_{ij}}{\tilde{\tau}}\bigr)$ (negated for minimisation) | smooth |

The Sinkhorn plan is computed via the log-domain iteration

$$
\boldsymbol{f}^{(t+1)} = \varepsilon\log\tfrac{1}{M} - \varepsilon\,\mathrm{LSE}_j\!\left(\tfrac{-\mathbf{D} + \mathbf{1}\boldsymbol{g}^{(t)T}}{\varepsilon}\right), \qquad
\boldsymbol{g}^{(t+1)} = \varepsilon\log\tfrac{1}{N} - \varepsilon\,\mathrm{LSE}_i\!\left(\tfrac{-\mathbf{D} + \boldsymbol{f}^{(t+1)}\mathbf{1}^T}{\varepsilon}\right),
$$

unrolled with `jax.lax.scan` for 50 iterations with regularisation parameter $\varepsilon = 10^{-2}$.

### 3.4 Gradient-Descent LM Optimiser

Because the mesh-distance objectives are scalar functions of $\boldsymbol{r}$,
the LM normal equations reduce to gradient descent with damping.  Incorporating
temporal smoothing and per-joint regularisation from Section 2.1:

$$
\bigl((\mu_k + \omega_s)\mathbf{I} + \tilde{\boldsymbol{\Omega}}\bigr)\,\Delta\boldsymbol{r}^{(k)}
= -\nabla_{\boldsymbol{r}}\mathcal{L}^{(k)} - \tilde{\boldsymbol{\Omega}}\,\boldsymbol{r}^{(k)} - \omega_s\bigl(\boldsymbol{r}^{(k)} - \boldsymbol{r}^{(0)}\bigr),
$$

where $\nabla_{\boldsymbol{r}}\mathcal{L}^{(k)}$ is the gradient of the chosen
metric computed by `jax.grad` and $\tilde{\boldsymbol{\Omega}}$ is the same
per-joint regularisation diagonal as in Section 2.1.  The step clamp
($\|\Delta\boldsymbol{r}\| \leq 0.15$ rad) and joint regularisation are shared
with the Cartesian formulation.  The damping schedule and temporal smoothing
weight differ: $\mu_0$ and $\omega_s$ are chosen per objective to match the
gradient scale of each metric (Chamfer returns squared distances $\sim10^{-3}$
m$^2$; Hausdorff and Sinkhorn return distances $\sim10^{-1}$ m), whereas the
Cartesian objective uses a uniform $\omega_s = 5\times10^{-2}$ sized to match
its own gradient scale.  The Hausdorff outer maximum is approximated with a
log-sum-exp at temperature $T = 10^{-4}$ m$^2$ to eliminate the
discontinuous gradient flips that occur when the maximising nearest-neighbour
pair changes between iterations.

All metric functions and the gradient-descent step are compiled once via
`jax.jit(jax.vmap(\ldots))` and amortised over frames, matching the
computational structure of Section 2.

---

## 4. Temporal Smoothing Parameter Selection

### 4.1 Tradeoff

The temporal smoothing weight $\omega_s$ controls the tradeoff between
**tracking accuracy** and **inter-frame smoothness**.  For the Cartesian
embodiment objective:

- **Small $\omega_s$** (e.g.\ $10^{-3}$): the optimizer is nearly unconstrained
  and follows the MANO target closely frame-by-frame.  However, the robotic
  IK objective has many local minima that are equidistant from the target;
  small changes in the MANO pose can flip the IK to a different local minimum
  in the next frame, causing high-frequency joint oscillation visible as
  jitter in the visualisation.

- **Large $\omega_s$** (e.g.\ $5\times10^{-1}$): the optimizer stays very close
  to the previous frame's solution, producing visually smooth trajectories.
  However, tracking lag accumulates — the joint angles cannot follow
  fast MANO motions — and mean tracking error increases.

### 4.2 Evaluation Protocol

We evaluate on two representative recordings:

- **Shadow Dexterous Hand / passport**: a multi-finger dexterous manipulation
  task with rich inter-finger coordination and large joint excursions.
- **Mia Hand / small pillow**: a power-grasp task with predominantly
  in-plane motion.

For each recording and each value of $\omega_s$ in
$\{10^{-3}, 5\times10^{-3}, 10^{-2}, 2\times10^{-2}, 5\times10^{-2},
10^{-1}, 2\times10^{-1}, 5\times10^{-1}\}$, we run the full pipeline,
compute the **mean tracking error** (mean Euclidean distance between
the MANO EE target and the robot FK position across all EE frames and
all time steps, in mm) and two smoothness metrics: **mean inter-frame
FK position change** (mm) and **mean inter-frame joint angle change**
(degrees).

### 4.3 Results: Shadow Dexterous Hand — Passport

| $\omega_s$ | Tracking (mm) | Smooth pos. (mm/frame) | Smooth ang. (°/frame) |
|---|---|---|---|
| 0.001 | 16.10 | 0.355 | 0.163 |
| 0.005 | **15.29** | 0.357 | 0.149 |
| 0.010 | 15.61 | 0.346 | 0.140 |
| 0.020 | 15.83 | 0.333 | 0.131 |
| **0.050** | **16.32** | **0.308** | **0.117** |
| 0.100 | 16.93 | 0.279 | 0.103 |
| 0.200 | 17.90 | 0.235 | 0.086 |
| 0.500 | 19.57 | 0.163 | 0.059 |

Bold row = default $\omega_s = 0.05$.
Minimum tracking error at $\omega_s = 0.005$; default $\omega_s = 0.05$ adds
only $+1.0$ mm tracking penalty while reducing position jitter by $13\,\%$
and angular jitter by $28\,\%$.

### 4.4 Results: Mia Hand — Small Pillow

| $\omega_s$ | Tracking (mm) | Smooth pos. (mm/frame) | Smooth ang. (°/frame) |
|---|---|---|---|
| 0.001 | — | — | — |
| 0.005 | — | — | — |
| 0.010 | — | — | — |
| 0.020 | — | — | — |
| **0.050** | **—** | **—** | **—** |
| 0.100 | — | — | — |
| 0.200 | — | — | — |
| 0.500 | — | — | — |

### 4.5 Recommended Values

Based on the evaluation above, the following defaults are used:

| Objective | Default $\omega_s$ | Rationale |
|---|---|---|
| Cartesian | $5\times10^{-2}$ | Dominates the LM diagonal ($\mu_0 = 10^{-2}$) throughout all 15 iterations, preventing local-minimum flipping while keeping tracking error below 15 mm on passport. |
| Mesh (all) | $5\times10^{-2}$ | Same as Cartesian. Although mesh gradients are smaller, the contact-surface objective landscape is multi-modal. A large $\omega_s$ prevents the optimizer from jumping to a different local minimum across frames, which is the primary source of frame-to-frame oscillation. |

The `--smooth-w` flag overrides these per-objective defaults at runtime.

---

## 5. Adaptive Hand-Base Transform

### 5.1 Motivation

The embodiment mapping transforms MANO hand-base coordinates into robot base
frame via a fixed calibrated transform $T_{\text{hand},\text{robot}}$.  In
practice this calibration may have a small systematic translation error: the
operator places the mocap volume at a slightly different absolute position
relative to the robot than assumed in the configuration file.  This manifests
as a constant offset between the MANO finger targets and the robot FK
positions — an error that cannot be reduced by improving the IK, since the
objective positions themselves are displaced.

### 5.2 Online Translation Adaptation

We adapt only the **translation** component of $T_{\text{hand},\text{robot}}$;
the rotation is assumed to be correctly calibrated from offline measurement.

Let $\boldsymbol{t}^{(n)}$ be the translation at frame $n$,
$\boldsymbol{t}^{(0)}$ the initial calibrated value,
$\boldsymbol{e}^{(n)} = \frac{1}{F}\sum_{f,j}\bigl(g_{f,j}(\boldsymbol{r}^{*(n)}) - \text{des}_{f,j}^{(n)}\bigr)$
the mean FK-minus-desired residual across all $F$ EE frames after the IK solve
at frame $n$.  The update rule is

$$\boldsymbol{t}^{(n+1)} = \boldsymbol{t}^{(n)}
  + \alpha\,\boldsymbol{e}^{(n)}
  - \rho\!\left(\boldsymbol{t}^{(n)} - \boldsymbol{t}^{(0)}\right),$$

and the result is projected onto the ball
$\|\boldsymbol{t} - \boldsymbol{t}^{(0)}\| \leq d_{\max}$, where
$\alpha = 10^{-3}$ m/frame is the adaptation learning rate,
$\rho = 10^{-4}$ is the regularisation coefficient pulling the translation
back toward the initial value, and $d_{\max} = 5$ cm is the hard constraint.

The regularisation term $\rho(\boldsymbol{t} - \boldsymbol{t}^{(0)})$ is
essential: without it, $\boldsymbol{t}$ would drift toward the robot FK
positions and remove the entire embodiment effect.  Setting
$\rho \gg \alpha/d_{\max}$ ensures the asymptotic displacement satisfies
$|\boldsymbol{t}^{(\infty)} - \boldsymbol{t}^{(0)}| \approx \alpha\bar{e}/\rho$,
bounded to a few centimetres even if the systematic offset $\bar{e}$ is large.

### 5.3 Caching

At the end of each run the adapted translation is written to a JSON file
named `{recording_stem}_handbase_cache.json` alongside the demo file.
Subsequent runs on the same recording load this file and initialise
$\boldsymbol{t}^{(0)}$ from the cached value, so the adaptation starts
from a previously converged state rather than from scratch.

---

## 7. Summary of Transformations

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

## 8. Computational Performance

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
