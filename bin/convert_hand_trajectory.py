"""Convert a MoCap trajectory to a robotic hand: record and embodiment mapping.

Example call:

python bin/convert_hand_trajectory.py mia --mia-thumb-adducted --demo-file data/QualisysAprilTest/april_test_009.tsv --output trajectory_009.csv
"""
import argparse
from hand_embodiment.mocap_dataset import HandMotionCaptureDataset
from hand_embodiment.pipelines import MoCapToRobot
from hand_embodiment.target_dataset import convert_mocap_to_robot
from hand_embodiment.command_line import (
    add_hand_argument, add_configuration_arguments,
    add_playback_control_arguments)


def parse_args():
    parser = argparse.ArgumentParser()
    add_hand_argument(parser)
    parser.add_argument(
        "--demo-files", type=str, nargs="*",
        default=["data/QualisysAprilTest/april_test_010.tsv"],
        help="Demonstrations that should be used.")
    add_configuration_arguments(parser)
    parser.add_argument(
        "--output", type=str, default="trajectory.csv",
        help="Output file (.csv).")
    add_playback_control_arguments(parser)
    parser.add_argument(
        "--mia-thumb-adducted", action="store_true",
        help="Adduct thumb of Mia hand.")

    return parser.parse_args()


def main():
    args = parse_args()

    for demo_file in args.demo_files:
        dataset = HandMotionCaptureDataset(
            demo_file, mocap_config=args.mocap_config,
            skip_frames=args.skip_frames, start_idx=args.start_idx,
            end_idx=args.end_idx)

        pipeline = MoCapToRobot(
            args.hand, args.mano_config, dataset.finger_names,
            record_mapping_config=args.record_mapping_config)

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
