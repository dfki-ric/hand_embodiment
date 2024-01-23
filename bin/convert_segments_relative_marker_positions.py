"""Make fingers markers in MoCap segments relative to MANO mesh.

Example call:

    python bin/convert_segments_relative_marker_positions.py close --demo-files examples/eigengrasps/*.json --mocap-config examples/config/markers/20220328_april.yaml --mano-config examples/config/mano/20211105_april.yaml
"""
import argparse
import warnings

import numpy as np
import pytransform3d.transformations as pt
import tqdm

from hand_embodiment.mocap_dataset import SegmentedHandMotionCaptureDataset
from hand_embodiment.record_markers import MarkerBasedRecordMapping, estimate_hand_pose
from hand_embodiment.config import load_mano_config, load_record_mapping_config
from hand_embodiment.command_line import add_configuration_arguments


def parse_args():
    parser = argparse.ArgumentParser()
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
        "--output", type=str, default="relative_markers.csv",
        help="Output file (.csv).")
    parser.add_argument(
        "--skip-frames", type=int, default=1,
        help="Skip this number of frames between animated frames.")
    parser.add_argument(
        "--interpolate-missing-markers", action="store_true",
        help="Interpolate NaNs.")

    return parser.parse_args()


def main():
    args = parse_args()

    finger_names = ["thumb", "index", "middle", "ring", "little"]

    warnings.simplefilter("ignore")

    mano2hand_markers, betas = load_mano_config(args.mano_config)
    record_mapping_config = args.record_mapping_config
    if record_mapping_config is not None:
        record_mapping_config = load_record_mapping_config(
            args.record_mapping_config)
    mbrm = MarkerBasedRecordMapping(
        left=False, mano2hand_markers=mano2hand_markers,
        shape_parameters=betas, record_mapping_config=record_mapping_config,
        use_fingers=finger_names, verbose=0, measure_time=False)

    output_dataset = []
    for demo_file in tqdm.tqdm(args.demo_files):
        dataset = SegmentedHandMotionCaptureDataset(
            demo_file, args.segment_label, mocap_config=args.mocap_config,
            interpolate_missing_markers=args.interpolate_missing_markers,
            label_field=args.label_field)
        if dataset.n_segments == 0:
            continue

        for i in tqdm.tqdm(range(dataset.n_segments), leave=False):
            dataset.select_segment(i)

            for t in range(dataset.n_steps):
                hand_markers = dataset.get_hand_markers(t)
                finger_markers = dataset.get_finger_markers(t)

                current_hand_markers2world = estimate_hand_pose(*hand_markers)
                if np.any(np.isnan(current_hand_markers2world)):
                    continue

                mano2world_ = pt.concat(
                    mbrm.mano2hand_markers_, current_hand_markers2world)
                world2mano = pt.invert_transform(mano2world_, check=False)

                sample = []
                markers_in_mano = {}
                for finger_name in finger_names:
                    markers_in_world = np.atleast_2d(finger_markers[finger_name])
                    markers_in_mano[finger_name] = np.dot(
                        pt.vectors_to_points(markers_in_world), world2mano.T)[:, :3]
                    sample.extend(markers_in_mano[finger_name].ravel())

                """
                import pytransform3d.visualizer as pv
                fig = pv.figure()
                mbrm.estimate(hand_markers, finger_markers)
                fig.add_geometry(mbrm.hand_state_.hand_mesh)
                for finger_name in finger_markers:
                    fig.scatter(finger_markers[finger_name], s=0.006)
                fig.scatter(hand_markers, s=0.006)
                fig.show()
                """

                output_dataset.append(sample)

    output_dataset = np.array(output_dataset)
    np.savetxt(args.output, output_dataset)


if __name__ == "__main__":
    main()
