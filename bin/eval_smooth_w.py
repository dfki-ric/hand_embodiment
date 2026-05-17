"""Evaluate the tradeoff between tracking accuracy and temporal smoothness.

Sweeps --smooth-w across a logarithmic grid for one or more recordings and
reports per-objective metrics:
  * tracking_err_mm : mean distance between MANO EE target and robot FK
  * smooth_mm       : mean |Δ position| across consecutive frames (robot FK)
  * smooth_deg      : mean |Δ joint angle| across consecutive frames

Example calls:
python bin/eval_smooth_w.py shadow \
  --demo-file raw_data/passport_set0.tsv \
  --mocap-config examples/config/markers/20211112_april.yaml \
  --mano-config examples/config/mano/20211105_april.yaml \
  --record-mapping-config examples/config/record_mapping/20211105_april.yaml

python bin/eval_smooth_w.py mia \
  --demo-file raw_data/small_pillow_set0.tsv \
  --mocap-config examples/config/markers/20210826_april.yaml \
  --mano-config examples/config/mano/20210610_april.yaml \
  --record-mapping-config examples/config/record_mapping/20211105_april.yaml \
  --mia-thumb-adducted
"""
import argparse
import numpy as np

from hand_embodiment.mocap_dataset import HandMotionCaptureDataset
from hand_embodiment.pipelines import MoCapToRobot
from hand_embodiment.target_configurations import TARGET_CONFIG
from hand_embodiment.command_line import (
    add_hand_argument, add_configuration_arguments,
    add_playback_control_arguments, add_objective_argument)


def parse_args():
    parser = argparse.ArgumentParser()
    add_hand_argument(parser)
    parser.add_argument("--demo-file", type=str, required=True)
    add_configuration_arguments(parser)
    add_playback_control_arguments(parser)
    parser.add_argument("--mia-thumb-adducted", action="store_true")
    parser.add_argument(
        "--objective", default="cartesian",
        choices=["cartesian", "chamfer", "hausdorff", "sinkhorn", "fscore"])
    parser.add_argument(
        "--smooth-w-values", type=float, nargs="+",
        default=[0.001, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5],
        help="Sweep of smooth_w values to evaluate.")
    return parser.parse_args()


def run_sweep(args, pipeline, dataset, finger_names, smooth_w_values):
    """Run evaluation sweep over smooth_w values using a single compiled pipeline."""
    results = []
    for smooth_w in smooth_w_values:
        pipeline.record_mapping_.reset()
        # Reset joint angles so warm-start for first frame is always zero
        for f in finger_names:
            pipeline.embodiment_mapping_.joint_angles[f] = np.zeros_like(
                pipeline.embodiment_mapping_.joint_angles[f])
        tracking_errs = []
        smooth_pos = []
        smooth_ang = []
        prev_q = None

        for i in range(dataset.n_steps):
            pipeline.record_mapping_.estimate(
                dataset.get_hand_markers(i),
                dataset.get_finger_markers(i))
            joint_angles, desired = pipeline.embodiment_mapping_.solve(
                handbase2world=pipeline.record_mapping_.mano2world_,
                return_desired_positions=True,
                smooth_w=smooth_w)

            # Tracking error: distance MANO EE → robot FK
            for f in finger_names:
                chain = pipeline.embodiment_mapping_.target_finger_chains[f]
                fk_T = chain.forward(joint_angles[f])
                fk_pos = np.array([T[:3, 3] for T in fk_T])
                des_pos = desired[f]
                n = min(len(fk_pos), len(des_pos))
                tracking_errs.append(
                    np.mean(np.linalg.norm(fk_pos[:n] - des_pos[:n], axis=1)) * 1000)

            # Temporal smoothness
            cur_q = {f: joint_angles[f].copy() for f in finger_names}
            if prev_q is not None:
                for f in finger_names:
                    delta_q = np.abs(cur_q[f] - prev_q[f])
                    smooth_ang.append(np.mean(delta_q) * (180.0 / np.pi))
                    chain = pipeline.embodiment_mapping_.target_finger_chains[f]
                    T_cur  = chain.forward(cur_q[f])
                    T_prev = chain.forward(prev_q[f])
                    pos_cur  = np.array([T[:3, 3] for T in T_cur])
                    pos_prev = np.array([T[:3, 3] for T in T_prev])
                    smooth_pos.append(
                        np.mean(np.linalg.norm(pos_cur - pos_prev, axis=1)) * 1000)
            prev_q = cur_q

        results.append({
            "smooth_w":      smooth_w,
            "tracking_mm":   float(np.mean(tracking_errs)),
            "smooth_pos_mm": float(np.mean(smooth_pos)) if smooth_pos else 0.0,
            "smooth_ang_deg": float(np.mean(smooth_ang)) if smooth_ang else 0.0,
            "n_frames":      dataset.n_steps,
        })
    return results


def main():
    args = parse_args()

    dataset = HandMotionCaptureDataset(
        args.demo_file,
        mocap_config=args.mocap_config,
        skip_frames=getattr(args, 'skip_frames', 1),
        start_idx=getattr(args, 'start_idx', None),
        end_idx=getattr(args, 'end_idx', None))

    finger_names = list(set(dataset.finger_names).intersection(
        set(TARGET_CONFIG[args.hand]["ee_frames"].keys())))

    # Build pipeline once — smooth_w is now a runtime arg, no recompilation needed
    pipeline = MoCapToRobot(
        args.hand, args.mano_config, finger_names,
        record_mapping_config=args.record_mapping_config,
        objective=args.objective)

    if args.hand == "mia":
        angle = 1.0 if args.mia_thumb_adducted else -1.0
        pipeline.set_constant_joint("j_thumb_opp_binary", angle)

    print(f"Objective: {args.objective}")
    print(f"Demo:      {args.demo_file}")
    print(f"Frames:    {dataset.n_steps}")
    print()
    header = f"{'smooth_w':>10s}  {'track_mm':>10s}  {'smooth_pos_mm':>14s}  {'smooth_ang_deg':>15s}"
    print(header)
    print("-" * len(header))

    for r in run_sweep(args, pipeline, dataset, finger_names, args.smooth_w_values):
        print(f"{r['smooth_w']:>10.4f}  "
              f"{r['tracking_mm']:>10.3f}  "
              f"{r['smooth_pos_mm']:>14.3f}  "
              f"{r['smooth_ang_deg']:>15.3f}")


if __name__ == "__main__":
    main()
