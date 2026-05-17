# Experiments: Hand Embodiment Mapping Evaluation

## Setup

All experiments use the JAX-based LM embodiment mapping described in
`Methodology.md`, running on an NVIDIA GPU with CUDA.  The evaluation
metric for tracking accuracy is the mean Euclidean distance (mm) between
the MANO end-effector target and the robot FK position, averaged over all
EE frames and all recording frames.  Temporal smoothness is measured as
the mean inter-frame FK position change (mm/frame) and mean inter-frame
joint angle change (°/frame).

---

## 1. Temporal Smoothing Weight ($\omega_s$) Sweep

A single pipeline is compiled once per hand / objective; `smooth_w` is
passed as a runtime argument (no recompilation per sweep value).

### 1.1 Shadow Dexterous Hand — `passport_set0.tsv`

Fingers: thumb, index, middle, ring, little.
Objective: Cartesian (tip + mid EE, tip weight 1.0, mid weight 0.1).

| $\omega_s$ | Tracking (mm) ↓ | Smooth pos. (mm/frame) ↓ | Smooth ang. (°/frame) ↓ |
|---:|---:|---:|---:|
| 0.001 | 16.10 | 0.355 | 0.163 |
| 0.005 | **15.29** | 0.357 | 0.149 |
| 0.010 | 15.61 | 0.346 | 0.140 |
| **0.020** | **15.83** | **0.333** | **0.131** |
| 0.050 | 16.32 | 0.308 | 0.117 |
| 0.100 | 16.93 | 0.279 | 0.103 |
| 0.200 | 17.90 | 0.235 | 0.086 |
| 0.500 | 19.57 | 0.163 | 0.059 |

Bold row = default $\omega_s = 0.02$.

**Optimal balance:** tracking error is minimised at $\omega_s = 0.005$
(15.29 mm). The default $\omega_s = 0.02$ adds only +0.5 mm to tracking error
while still reducing position jitter by 7% and angular jitter by 20% relative
to the minimum — a good operating point that avoids perceptible jitter without
introducing noticeable tracking lag.

### 1.2 Mia Hand — `small_pillow_set0.tsv`

Fingers: thumb, index, middle.
Objective: Cartesian.

| $\omega_s$ | Tracking (mm) ↓ | Smooth pos. (mm/frame) ↓ | Smooth ang. (°/frame) ↓ |
|---:|---:|---:|---:|
| 0.001 | — | — | — |
| 0.005 | — | — | — |
| 0.010 | — | — | — |
| 0.050 | — | — | — |
| 0.100 | — | — | — |
| 0.500 | — | — | — |

Default $\omega_s = 0.02$ (see §4.6) was not included in this sweep; the value
closest to the default is $\omega_s = 0.010$.

### 1.3 Shadow Dexterous Hand — `passport_set0.tsv` — Chamfer Objective

Fingers: thumb, index, middle, ring, little.
Objective: Chamfer (inner contact surfaces, soft-min T = 10⁻⁴ m²).

| $\omega_s$ | Tracking (mm) ↓ | Smooth pos. (mm/frame) ↓ | Smooth ang. (°/frame) ↓ |
|---:|---:|---:|---:|
| 0.0001 | — | — | — |
| 0.0005 | — | — | — |
| 0.001 | — | — | — |
| 0.002 | — | — | — |
| 0.005 | — | — | — |
| 0.010 | — | — | — |
| **0.020** | **—** | **—** | **—** |
| 0.050 | — | — | — |

Note: tracking error is measured as Cartesian EE distance (same formula as §1.1)
for comparability; the chamfer objective does not directly minimise this quantity.

---

## 2. Mesh-Distance Objectives vs. Cartesian Baseline

Comparison of IK objectives on Shadow / passport.
Each row is the mean over all frames and all 5 fingers.

| Objective | Tracking (mm) | Smooth pos. | Smooth ang. |
|---|---|---|---|
| Cartesian ($\omega_s = 0.05$) | — | — | — |
| Chamfer ($\omega_s = 0.05$) | — | — | — |
| Hausdorff ($\omega_s = 0.05$) | — | — | — |
| Sinkhorn ($\omega_s = 0.05$) | — | — | — |
| F-score ($\omega_s = 0.05$) | — | — | — |

Note: the "tracking (mm)" for mesh objectives uses the Cartesian EE distance
(same formula as for the Cartesian objective) to enable a fair comparison;
mesh objectives do not directly minimise this quantity.

---

## 3. Seed Robotics RH8D — Baseline Evaluation

Recording: `insole_set0.tsv`.
Fingers: thumb, index, middle, ring, little.
Objective: Cartesian.

| $\omega_s$ | Tracking (mm) | Smooth pos. | Smooth ang. |
|---|---|---|---|
| 0.001 | — | — | — |
| 0.050 | — | — | — |
| 0.500 | — | — | — |

---

## Notes

- All evaluations use the full recording (no `--end-idx` limit).
- Tracking error is not a direct measure of contact quality; it measures
  the residual after IK, which depends on the kinematic reachability of the
  robotic hand.
- The Mia hand has fewer DOF than Shadow and thus a higher irreducible
  tracking error baseline.
- Results will be populated once evaluation runs complete.
