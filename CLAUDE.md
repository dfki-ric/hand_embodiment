# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This library implements **embodiment mapping** for robotic hands from human hand demonstrations (Fabisch et al., IEEE Humanoids 2022). The pipeline transfers human hand motions — recorded via marker-based motion capture — to robotic hands, using the [MANO](https://mano.is.tue.mpg.de/) parametric hand model as an intermediate representation. This decoupling lets you swap the MoCap approach or the target robot without changing the other side.

The three-stage pipeline:
1. **Record mapping** — estimates MANO pose/shape from Qualisys MoCap marker positions
2. **MANO state** — intermediate representation (shape β ∈ ℝ¹⁰, pose ∈ ℝ⁴⁸)
3. **Embodiment mapping** — solves robot IK from MANO end-effector positions

## Commands

```bash
# Install for development
pip install -e .
pip install -e ".[test]"   # with test dependencies

# Run all tests
pytest test/

# Run a single test file
pytest test/test_kinematics.py

# Run a single test
pytest test/test_markers_to_robot.py::test_shadow_pipeline

# Build docs
cd doc && make html
```

### Quick end-to-end example (uses bundled test data)

```bash
python bin/vis_markers_to_robot.py shadow \
  --demo-file test/data/recording.tsv \
  --mocap-config examples/config/markers/20210826_april.yaml \
  --mano-config examples/config/mano/20210610_april.yaml \
  --record-mapping-config examples/config/record_mapping/20211105_april.yaml \
  --visual-objects pillow-small --show-mano
```

## Architecture

### Core modules

| Module | Class / role |
|---|---|
| `mano.py` | `HandState` — wraps MANO mesh (Open3D); updated by record mapping, consumed by embodiment mapping |
| `mano_jax.py` | Pure-JAX MANO forward kinematics; JIT-compiled for the LM solver |
| `record_markers.py` | `MarkerBasedRecordMapping` — MANO IK from marker positions |
| `embodiment.py` | `HandEmbodiment` — robot IK from MANO EE positions |
| `pipelines.py` | `MoCapToRobot` — composes the two steps above |
| `kinematics.py` | `FastUrdfTransformManager` (numba-accelerated URDF FK) + JAX-based LM solver |
| `target_configurations.py` | `TARGET_CONFIG` dict — built-in robot hand definitions |
| `metrics.py` | Contact-surface vertex indices per hand/finger |
| `mesh_metrics_jax.py` | JAX mesh-distance objectives (Chamfer, Hausdorff, Sinkhorn, F-score) |

### Optimization — why JAX

Both the record mapping and embodiment mapping are solved with a **Levenberg–Marquardt** optimizer instead of SLSQP (original paper). The JAX implementation provides:
- `jax.jacfwd` — exact Jacobians via forward-mode AD (no finite differences)
- `jax.vmap` — all 5 fingers solved in parallel in one XLA kernel
- `jax.lax.scan` — fixed 15-step LM loop, fully fused by XLA
- `jax.jit` — compiled once at init, amortised over all frames (~40 Hz on GPU)

JAX is configured globally in `kinematics.py`: `jax_enable_x64 = True`. XLA GPU preallocate is disabled; cache lives in `~/.cache/hand_embodiment/jax_cache`.

### IK objective modes

The `objective` parameter of `HandEmbodiment` / `MoCapToRobot`:

- `'cartesian'` (default) — 2-point Cartesian distance (tip + mid phalanx) per finger; tip weight 1.0, mid weight 0.1
- `'chamfer'`, `'hausdorff'`, `'sinkhorn'`, `'fscore'` — mesh contact-surface distances (`mesh_metrics_jax.py`), using **soft-min** (log-sum-exp, T = 10⁻⁴ m²) to avoid the gradient oscillation of hard nearest-neighbour assignment

Temporal smoothing weight `smooth_w` defaults to **0.02** for all objectives (knee of the tracking/smoothness tradeoff on Shadow/passport). Override at runtime with `--smooth-w`.

### Configuration files

Three YAML config types, all under `examples/config/`:

| Type | Path | Contents |
|---|---|---|
| MoCap markers | `markers/*.yaml` | Marker names, hand/finger definitions, scale |
| MANO | `mano/*.yaml` | Shape params `betas` (ℝ¹⁰), `mano2hand_markers` transform |
| Record mapping | `record_mapping/*.yaml` | Per-finger weights, action weights, smoothing params |

Robot configurations live in `target_configurations.py` as Python dicts (see `TARGET_CONFIG`). Non-standard hands can be defined in a custom YAML and passed via `--robot-config`.

### Adding a new robotic hand

Add an entry to `TARGET_CONFIG` in `target_configurations.py`. Required keys: `joint_names`, `base_frame`, `ee_frames`, `intermediate_frames`, `handbase2robotbase` (4×4, MANO base → robot base), and `model` (URDF path + `package_dir`). Optional: `virtual_joints_callbacks` for coupled/binary joints (see `MiaVirtualThumbJoint` as an example), and `contact_surface_vertices` for mesh objectives.

### Adaptive hand-base

`HandEmbodiment(adapt_handbase=True)` online-adapts the **translation** of `handbase2robotbase` to correct systematic MoCap calibration offsets. The adapted state is cached in `{recording_stem}_handbase_cache.json` beside the demo file and loaded on subsequent runs.

## Version control

This repository uses two remotes:
- `origin` — primary remote (GitLab)
- `internal` — internal DFKI remote with additional branches (e.g. `internal/feat/rh8d`)

The current development branch is `feat/jax`. The `main` branch tracks the published version.

## Prerequisites

The **MANO model** is not bundled and must be downloaded from [mano.is.tue.mpg.de](https://mano.is.tue.mpg.de/). After downloading, run:

```bash
python hand_embodiment/model/mano/prepare_mano.py
```

The public **dataset** (763 demonstrations, 8 tasks) is on Zenodo: `10.5281/zenodo.20180859`.
