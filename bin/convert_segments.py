"""Convert MoCap segments to a robotic hand: record and embodiment mapping."""
import argparse

from hand_embodiment.mocap_dataset import SegmentedHandMotionCaptureDataset
from hand_embodiment.pipelines import MoCapToRobot
from hand_embodiment.target_dataset import convert_mocap_to_robot
from hand_embodiment.timing import timing_report
from hand_embodiment.command_line import (
    add_hand_argument, add_configuration_arguments,
    add_frame_transform_arguments)
from hand_embodiment.mocap_objects import extract_mocap_origin2object_generic


def parse_args():
    parser = argparse.ArgumentParser()
    add_hand_argument(parser)
    parser.add_argument(
        "segment_label", type=str,
        help="Label of the segment that should be used.")
    parser.add_argument(
        "--demo-files", type=str, nargs="*",
        default=["data/20210616_april/metadata/Measurement24.json"],
        help="Demonstrations that should be used.")
    add_configuration_arguments(parser)
    parser.add_argument(
        "--label-field", type=str, default="l1",
        help="Name of the label field in metadata file.")
    parser.add_argument(
        "--output", type=str, default="segment_%02d.csv",
        help="Output file pattern (.csv).")
    parser.add_argument(
        "--skip-frames", type=int, default=1,
        help="Skip this number of frames between animated frames.")
    parser.add_argument(
        "--interpolate-missing-markers", action="store_true",
        help="Interpolate NaNs.")
    parser.add_argument(
        "--mia-thumb-adducted", action="store_true",
        help="Adduct thumb of Mia hand.")
    parser.add_argument(
        "--measure-time", action="store_true",
        help="Measure time of record and embodiment mapping.")
    add_frame_transform_arguments(parser)

    return parser.parse_args()


def main():
    args = parse_args()

    dataset = SegmentedHandMotionCaptureDataset(
        args.demo_files[0], args.segment_label, mocap_config=args.mocap_config,
        label_field=args.label_field)
    pipeline = MoCapToRobot(args.hand, args.mano_config, dataset.finger_names,
                            record_mapping_config=args.record_mapping_config,
                            robot_config=args.robot_config,
                            measure_time=args.measure_time)

    total_segment_idx = 0
    for demo_file in args.demo_files:
        dataset = SegmentedHandMotionCaptureDataset(
            demo_file, args.segment_label, mocap_config=args.mocap_config,
            interpolate_missing_markers=args.interpolate_missing_markers,
            label_field=args.label_field)
        if dataset.n_segments == 0:
            continue

        if args.hand == "mia":
            angle = 1.0 if args.mia_thumb_adducted else -1.0
            pipeline.set_constant_joint("j_thumb_opp_binary", angle)

        for i in range(dataset.n_segments):
            dataset.select_segment(i)

            mocap_origin2origin = extract_mocap_origin2object_generic(args, dataset)

            output_dataset = convert_mocap_to_robot(
                dataset, pipeline, mocap_origin2origin=mocap_origin2origin,
                verbose=1)

            if args.hand == "mia":
                j_min, j_max = pipeline.transform_manager_.get_joint_limits("j_thumb_opp")
                thumb_opp = j_max if args.mia_thumb_adducted else j_min
                output_dataset.add_constant_finger_joint("j_thumb_opp", thumb_opp)

            output_filename = args.output % total_segment_idx
            output_dataset.export(output_filename, pipeline.hand_config_)
            # TODO convert frequency
            print(f"Saved demonstration to '{output_filename}'")
            total_segment_idx += 1

    if args.measure_time:
        timing_report(pipeline.record_mapping_, title="record mapping")
        timing_report(pipeline.embodiment_mapping_, title="embodiment mapping")


if __name__ == "__main__":
    main()
