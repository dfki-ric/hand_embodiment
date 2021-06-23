"""
Example call:

python bin/save_hand_trajectory.py mia --mia-thumb-adducted --demo-file data/QualisysAprilTest/april_test_009.tsv --output trajectory_009.csv
"""
import argparse
from hand_embodiment.mocap_dataset import HandMotionCaptureDataset
from hand_embodiment.pipelines import MoCapToRobot
from hand_embodiment.target_dataset import convert_mocap_to_robot


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
        angle = 1.0 if args.mia_thumb_adducted else -1.0
        pipeline.set_constant_joint("j_thumb_opp_binary", angle)

    output_dataset = convert_mocap_to_robot(dataset, pipeline, verbose=1)

    if args.hand == "mia":
        j_min, j_max = pipeline.transform_manager_.get_joint_limits("j_thumb_opp")
        thumb_opp = j_max if args.mia_thumb_adducted else j_min
        output_dataset.add_constant_finger_joint("j_thumb_opp", thumb_opp)

    output_dataset.export(args.output, pipeline.hand_config_)
    # TODO convert frequency
    print(f"Saved demonstration to '{args.output}'")


if __name__ == "__main__":
    main()
