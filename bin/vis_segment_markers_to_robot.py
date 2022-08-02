"""Transfer MoCap data to robotic hand: record and embodiment mapping."""
import argparse
import numpy as np
from pytransform3d import visualizer as pv
from hand_embodiment.mocap_dataset import SegmentedHandMotionCaptureDataset
from hand_embodiment.pipelines import MoCapToRobot
from hand_embodiment.vis_utils import AnimationCallback
from hand_embodiment.command_line import (
    add_hand_argument, add_animation_arguments, add_configuration_arguments)


def parse_args():
    parser = argparse.ArgumentParser()
    add_hand_argument(parser)
    parser.add_argument(
        "segment_label", type=str,
        help="Label of the segment that should be used.")
    parser.add_argument(
        "--demo-file", type=str,
        default="data/QualisysAprilTest/april_test_010.tsv",
        help="Demonstration that should be used.")
    parser.add_argument(
        "--segments", type=int, default=None, nargs="+",
        help="Segments of demonstration that should be used.")
    add_configuration_arguments(parser)
    parser.add_argument(
        "--interpolate-missing-markers", action="store_true",
        help="Interpolate NaNs.")
    parser.add_argument(
        "--show-mano", action="store_true", help="Show MANO mesh")
    parser.add_argument(
        "--mia-thumb-adducted", action="store_true",
        help="Adduct thumb of Mia hand.")
    add_animation_arguments(parser)
    return parser.parse_args()


def main():
    args = parse_args()

    dataset = SegmentedHandMotionCaptureDataset(
        args.demo_file, args.segment_label, mocap_config=args.mocap_config,
        interpolate_missing_markers=args.interpolate_missing_markers)

    segments = args.segments
    if segments is None:
        segments = list(range(dataset.n_segments))
    dataset.select_segment(segments[0])

    pipeline = MoCapToRobot(
        args.hand, args.mano_config, dataset.finger_names,
        record_mapping_config=args.record_mapping_config, verbose=1,
        robot_config=args.robot_config)

    if args.hand == "mia":
        angle = 1.0 if args.mia_thumb_adducted else -1.0
        pipeline.set_constant_joint("j_thumb_opp_binary", angle)

    fig = pv.figure()
    fig.plot_transform(np.eye(4), s=1)
    markers = fig.scatter(dataset.get_markers(0), s=0.006)

    animation_callback = AnimationCallback(fig, pipeline, args, show_robot=True)
    fig.view_init(azim=45)
    fig.set_zoom(0.3)
    while fig.visualizer.poll_events():
        for segment_idx in segments:
            dataset.select_segment(segment_idx)
            fig.animate(
                animation_callback, dataset.n_steps, loop=False,
                fargs=(markers, dataset, pipeline))

    fig.show()


if __name__ == "__main__":
    main()
