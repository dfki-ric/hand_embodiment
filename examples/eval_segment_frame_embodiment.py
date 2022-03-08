"""Example calls:
python examples/eval_segment_frame_embodiment.py mia close 0 100 --mocap-config examples/config/markers/20211119_april.yaml --mano-config examples/config/mano/20211105_april.yaml --mia-thumb-adducted --show-mano --demo-file data/20211119_april/20211119_r_WK37_insole_set0.json --insole
python examples/eval_segment_frame_embodiment.py shadow grasp 0 80 --mocap-config examples/config/markers/20211126_april_pillow.yaml --mano-config examples/config/mano/20211105_april.yaml --mia-thumb-adducted --show-mano --demo-file data/20211126_april_pillow/20211126_r_WK37_big_pillow_set0.json
python examples/eval_segment_frame_embodiment.py shadow insert 0 120 --mocap-config examples/config/markers/20211217_april.yaml --mano-config examples/config/mano/20211105_april.yaml --mia-thumb-adducted --show-mano --demo-file data/20211217_april/20211217_r_WK37_passport_box_set0.json --passport-closed
"""

import argparse
import numpy as np
import json
from pytransform3d import visualizer as pv
from mocap.visualization import scatter
from hand_embodiment.mocap_dataset import SegmentedHandMotionCaptureDataset
from hand_embodiment.pipelines import MoCapToRobot
from hand_embodiment.vis_utils import AnimationCallback
from hand_embodiment.command_line import (
    add_animation_arguments, add_configuration_arguments)
from hand_embodiment.metrics import (
    CONTACT_SURFACE_VERTICES, distances_robot_to_mano)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "hand", type=str,
        help="Name of the hand. Possible options: mia, shadow_hand")
    parser.add_argument(
        "segment_label", type=str,
        help="Label of the segment that should be used.")
    parser.add_argument(
        "segment", type=int,
        help="Segments of demonstration that should be used.")
    parser.add_argument(
        "frame", type=int,
        help="Frame of segment that should be used.")
    parser.add_argument(
        "--demo-file", type=str,
        default="data/QualisysAprilTest/april_test_010.tsv",
        help="Demonstration that should be used.")
    add_configuration_arguments(parser)
    parser.add_argument(
        "--interpolate-missing-markers", action="store_true",
        help="Interpolate NaNs.")
    parser.add_argument(
        "--show-mano", action="store_true", help="Show MANO mesh")
    parser.add_argument(
        "--mia-thumb-adducted", action="store_true",
        help="Adduct thumb of Mia hand.")
    parser.add_argument(
        "--no-metric", action="store_true",
        help="Don't compute the metric, only show the configuration.")
    parser.add_argument(
        "--output-file", default=None,
        help="File to which the result should be written.")
    parser.add_argument(
        "--output-image", default=None,
        help="Image to which we render the configuration.")
    add_animation_arguments(parser)
    return parser.parse_args()


def main():
    args = parse_args()

    dataset = SegmentedHandMotionCaptureDataset(
        args.demo_file, args.segment_label, mocap_config=args.mocap_config,
        interpolate_missing_markers=args.interpolate_missing_markers)

    dataset.select_segment(args.segment)

    pipeline = MoCapToRobot(
        args.hand, args.mano_config, dataset.finger_names,
        record_mapping_config=args.record_mapping_config, verbose=1)

    if args.hand == "mia":
        angle = 1.0 if args.mia_thumb_adducted else -1.0
        pipeline.set_constant_joint("j_thumb_opp_binary", angle)

    fig = pv.figure()
    fig.plot_transform(np.eye(4), s=1)
    markers = fig.scatter(dataset.get_markers(0), s=0.006)

    animation_callback = AnimationCallback(fig, pipeline, args, show_robot=True)
    fig.view_init(azim=45)
    fig.set_zoom(0.3)

    drawn_artists = animation_callback(args.frame, markers, dataset, pipeline)
    for a in drawn_artists:
        for geometry in a.geometries:
            fig.update_geometry(geometry)

    if not args.no_metric:
        print("Starting computation of distances...")
        ROBOT_CONTACT_SURFACE_VERTICES = CONTACT_SURFACE_VERTICES[args.hand]
        fingers = ["thumb", "index", "middle", "ring", "little"]
        dists = distances_robot_to_mano(
            pipeline.embodiment_mapping_.hand_state_, animation_callback.robot,
            ROBOT_CONTACT_SURFACE_VERTICES, fingers)
        print("DONE")
        print(dists)
        if args.output_file is not None:
            with open(args.output_file, "w") as f:
                json.dump(dists, f)

    if args.output_image is None:
        fig.show()
    else:
        fig.save_image(args.output_image)


if __name__ == "__main__":
    main()
