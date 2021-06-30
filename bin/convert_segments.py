"""
Example call:

python bin/convert_segments.py mia close --mia-thumb-adducted --mocap-config examples/config/markers/20210616_april.yaml --demo-file data/20210616_april/metadata/Measurement16.json --output dataset_16_segment_%d.csv
python bin/convert_segments.py mia close --mia-thumb-adducted --mocap-config examples/config/markers/20210616_april.yaml --demo-file data/20210616_april/metadata/Measurement23.json --output dataset_23_segment_%d.csv
python bin/convert_segments.py mia close --mia-thumb-adducted --mocap-config examples/config/markers/20210616_april.yaml --demo-file data/20210616_april/metadata/Measurement24.json --output dataset_24_segment_%d.csv
"""
import argparse
from hand_embodiment.mocap_dataset import SegmentedHandMotionCaptureDataset
from hand_embodiment.pipelines import MoCapToRobot
from hand_embodiment.target_dataset import convert_mocap_to_robot
from hand_embodiment.vis_utils import insole_pose


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "hand", type=str,
        help="Name of the hand. Possible options: mia, shadow_hand")
    parser.add_argument(
        "segment_label", type=str,
        help="Label of the segment that should be used.")
    parser.add_argument(
        "--demo-file", type=str,
        default="data/20210616_april/metadata/Measurement24.json",
        help="Demonstration that should be used.")
    parser.add_argument(
        "--mocap-config", type=str,
        default="examples/config/markers/20210616_april.yaml",
        help="MoCap configuration file.")
    parser.add_argument(
        "--mano-config", type=str,
        default="examples/config/mano/20210520_april.yaml",
        help="MANO configuration file.")
    parser.add_argument(
        "--output", type=str, default="segment_%02d.csv",
        help="Output file pattern (.csv).")
    parser.add_argument(
        "--show-mano", action="store_true", help="Show MANO mesh")
    parser.add_argument(
        "--skip-frames", type=int, default=1,
        help="Skip this number of frames between animated frames.")
    parser.add_argument(
        "--mia-thumb-adducted", action="store_true",
        help="Adduct thumb of Mia hand.")
    parser.add_argument(
        "--insole-hack", action="store_true",
        help="Save insole pose at the beginning of the segment.")

    return parser.parse_args()


def main():
    args = parse_args()

    dataset = SegmentedHandMotionCaptureDataset(
        args.demo_file, args.segment_label, mocap_config=args.mocap_config)

    pipeline = MoCapToRobot(args.hand, args.mano_config, dataset.finger_names)

    if args.hand == "mia":
        angle = 1.0 if args.mia_thumb_adducted else -1.0
        pipeline.set_constant_joint("j_thumb_opp_binary", angle)

    for i in range(dataset.n_segments):
        dataset.select_segment(i)

        if args.insole_hack:
            ####################################################################
            ####################################################################
            # python bin/convert_segments.py mia close --mia-thumb-adducted --mocap-config examples/config/markers/20210616_april.yaml --demo-file data/20210616_april/metadata/Measurement16.json --output dataset_16_segment_%d.csv --insole-hack
            import numpy as np
            import pytransform3d.transformations as pt
            ee2origin = np.empty((dataset.n_steps, 4, 4))
            insole_back = np.zeros(3)
            insole_front = np.array([1, 0, 0])
            for t in range(dataset.n_steps):
                additional_markers = dataset.get_additional_markers(t)
                marker_names = dataset.config.get("additional_markers", ())
                if not any(np.isnan(additional_markers[marker_names.index("insole_front")])):
                    insole_front = additional_markers[marker_names.index("insole_front")]
                if not any(np.isnan(additional_markers[marker_names.index("insole_back")])):
                    insole_back = additional_markers[marker_names.index("insole_back")]
                origin_pose = insole_pose(insole_back, insole_front)
                ee2origin[t] = pt.invert_transform(origin_pose)
            ####################################################################
            ####################################################################
        else:
            ee2origin = None

        output_dataset = convert_mocap_to_robot(
            dataset, pipeline, ee2origin=ee2origin, verbose=1)

        if args.hand == "mia":
            j_min, j_max = pipeline.transform_manager_.get_joint_limits("j_thumb_opp")
            thumb_opp = j_max if args.mia_thumb_adducted else j_min
            output_dataset.add_constant_finger_joint("j_thumb_opp", thumb_opp)

        output_filename = args.output % i
        output_dataset.export(output_filename, pipeline.hand_config_)
        # TODO convert frequency
        print(f"Saved demonstration to '{output_filename}'")


if __name__ == "__main__":
    main()
