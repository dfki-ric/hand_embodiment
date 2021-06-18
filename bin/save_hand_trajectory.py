"""
Example call:

python bin/save_hand_trajectory.py mia --mia-thumb-adducted --demo-file data/QualisysAprilTest/april_test_009.tsv --output trajectory_009.csv
"""
import argparse
import time
import numpy as np
import tqdm
from hand_embodiment.mocap_dataset import HandMotionCaptureDataset
from hand_embodiment.target_dataset import RoboticHandDataset
from hand_embodiment.pipelines import MoCapToRobot


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "hand", type=str,
        help="Name of the hand. Possible options: mia, shadow_hand")
    parser.add_argument(
        "--demo-file", type=str,
        default="data/QualisysAprilTest/april_test_010.tsv",
        help="Demonstration that should be used.")
    parser.add_argument(
        "--mocap-config", type=str,
        default="examples/config/markers/20210520_april.yaml",
        help="MoCap configuration file.")
    parser.add_argument(
        "--mano-config", type=str,
        default="examples/config/mano/20210520_april.yaml",
        help="MANO configuration file.")
    parser.add_argument(
        "--output", type=str, default="trajectory.csv",
        help="Output file (.csv).")
    parser.add_argument(
        "--start-idx", type=int, default=None, help="Start index.")
    parser.add_argument(
        "--end-idx", type=int, default=None, help="Start index.")
    parser.add_argument(
        "--show-mano", action="store_true", help="Show MANO mesh")
    parser.add_argument(
        "--skip-frames", type=int, default=1,
        help="Skip this number of frames between animated frames.")
    parser.add_argument(
        "--mia-thumb-adducted", action="store_true",
        help="Adduct thumb of Mia hand.")

    return parser.parse_args()


def main():
    args = parse_args()

    dataset = HandMotionCaptureDataset(
        args.demo_file, mocap_config=args.mocap_config,
        skip_frames=args.skip_frames, start_idx=args.start_idx,
        end_idx=args.end_idx)

    pipeline = MoCapToRobot(args.hand, args.mano_config, dataset.finger_names)

    if args.hand == "mia":
        if args.mia_thumb_adducted:
            pipeline.transform_manager_.set_joint("j_thumb_opp_binary", 1.0)
        else:
            pipeline.transform_manager_.set_joint("j_thumb_opp_binary", -1.0)

    output_dataset = RoboticHandDataset(finger_names=dataset.finger_names)
    start_time = time.time()
    for t in tqdm.tqdm(range(dataset.n_steps)):
        ee_pose, joint_angles = pipeline.estimate(
            dataset.get_hand_markers(t), dataset.get_finger_markers(t))

        joint_angles_t = joint_angles.copy()
        if args.hand == "mia":  # TODO refactor
            j_min, j_max = pipeline.transform_manager_.get_joint_limits("j_thumb_opp")
            if args.mia_thumb_adducted:
                thumb_opp = j_max
            else:
                thumb_opp = j_min
            joint_angles_t["thumb"] = np.hstack((joint_angles_t["thumb"], [thumb_opp]))
        output_dataset.append(ee_pose, joint_angles_t)

    duration = time.time() - start_time
    time_per_frame = duration / dataset.n_steps
    frequency = dataset.n_steps / duration
    print(f"Embodiment mapping done after {duration:.2f} s, "
          f"{time_per_frame:.4f} s per frame, {frequency:.1f} Hz")

    output_dataset.export(args.output, args.hand, pipeline.hand_config_)
    # TODO convert frequency
    print(f"Saved demonstration to '{args.output}'")


if __name__ == "__main__":
    main()
