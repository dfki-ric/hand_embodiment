"""Transfer MoCap markers to MANO: record mapping."""
import argparse
import numpy as np
import pytransform3d.visualizer as pv

from hand_embodiment.mocap_dataset import HandMotionCaptureDataset
from hand_embodiment.pipelines import MoCapToRobot
from hand_embodiment.vis_utils import AnimationCallback
from hand_embodiment.command_line import (
    add_animation_arguments, add_configuration_arguments,
    add_playback_control_arguments)


MARKER_COLORS = [
    (0.5, 0.5, 0.5), (0.5, 0.5, 0.5), (0.5, 0.5, 0.5),
    (1, 0, 0), (0.5, 0, 0),
    (0, 1, 0), (0, 0.5, 0),
    (0, 0, 1), (0, 0, 0.5),
    (1, 1, 0), (0.5, 0.5, 0),
    (0, 1, 1), (0, 0.5, 0.5),
    (1, 0, 1), (0.5, 0, 0.5),
    (0.5, 0.5, 0.5), (0.5, 0.5, 0.5), (0.5, 0.5, 0.5),
    (0.5, 0.5, 0.5), (0.5, 0.5, 0.5), (0.5, 0.5, 0.5),
    (0.5, 0.5, 0.5), (0.5, 0.5, 0.5), (0.5, 0.5, 0.5),
]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--demo-file", type=str,
        default="data/QualisysAprilTest/april_test_010.tsv",
        help="Demonstration that should be used.")
    add_configuration_arguments(parser)
    add_playback_control_arguments(parser)
    parser.add_argument(
        "--interpolate-missing-markers", action="store_true",
        help="Interpolate NaNs.")
    parser.add_argument(
        "--hide-mano", action="store_true", help="Hide MANO mesh")
    add_animation_arguments(parser)
    return parser.parse_args()


def main():
    args = parse_args()

    dataset = HandMotionCaptureDataset(
        args.demo_file, mocap_config=args.mocap_config,
        skip_frames=args.skip_frames, start_idx=args.start_idx,
        end_idx=args.end_idx,
        interpolate_missing_markers=args.interpolate_missing_markers)

    pipeline = MoCapToRobot(
        "mia", args.mano_config, dataset.finger_names,
        record_mapping_config=args.record_mapping_config, verbose=1)

    fig = pv.figure()
    fig.plot_transform(np.eye(4), s=0.5)
    markers = dataset.get_markers(0)
    markers = fig.scatter(markers, s=0.006, c=MARKER_COLORS[:len(markers)])

    animation_callback = AnimationCallback(fig, pipeline, args)

    fig.view_init(azim=45)
    fig.animate(
        animation_callback, dataset.n_steps, loop=True,
        fargs=(markers, dataset, pipeline))

    fig.show()


if __name__ == "__main__":
    main()
