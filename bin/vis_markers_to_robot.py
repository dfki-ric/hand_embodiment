"""Transfer MoCap data to robotic hand: record and embodiment mapping."""
import argparse
import time

import numpy as np
from pytransform3d import visualizer as pv
from hand_embodiment.mocap_dataset import HandMotionCaptureDataset
from hand_embodiment.pipelines import MoCapToRobot
from hand_embodiment.vis_utils import AnimationCallback
from hand_embodiment.command_line import (
    add_hand_argument, add_animation_arguments, add_configuration_arguments,
    add_playback_control_arguments)
from hand_embodiment.target_configurations import TARGET_CONFIG


def parse_args():
    parser = argparse.ArgumentParser()
    add_hand_argument(parser)
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
        "--show-mano", action="store_true", help="Show MANO mesh")
    parser.add_argument(
        "--mia-thumb-adducted", action="store_true",
        help="Adduct thumb of Mia hand.")
    parser.add_argument(
        "--measure-time", action="store_true",
        help="Measure time of record and embodiment mapping.")
    add_animation_arguments(parser)
    return parser.parse_args()


def main():
    args = parse_args()

    dataset = HandMotionCaptureDataset(
        args.demo_file, mocap_config=args.mocap_config,
        skip_frames=args.skip_frames, start_idx=args.start_idx,
        end_idx=args.end_idx,
        interpolate_missing_markers=args.interpolate_missing_markers)

    finger_names = list(set(dataset.finger_names).intersection(
        set(TARGET_CONFIG[args.hand]["ee_frames"].keys())))
    print(finger_names)
    pipeline = MoCapToRobot(args.hand, args.mano_config, finger_names,
                            record_mapping_config=args.record_mapping_config,
                            verbose=1, measure_time=args.measure_time,
                            robot_config=args.robot_config)

    if args.hand == "mia":
        angle = 1.0 if args.mia_thumb_adducted else -1.0
        pipeline.set_constant_joint("j_thumb_opp_binary", angle)

    fig = pv.figure()
    fig.plot_transform(np.eye(4), s=0.5)
    markers = fig.scatter(dataset.get_markers(0), s=0.006)

    animation_callback = AnimationCallback(fig, pipeline, args, show_robot=True)
    fig.view_init(azim=45)
    while fig.visualizer.poll_events():
        fig.animate(
            animation_callback, dataset.n_steps, loop=False,
            fargs=(markers, dataset, pipeline))
        if args.measure_time:
            print(f"Average frequency of record mapping: "
                  f"{1.0 / np.mean(pipeline.record_mapping_.timings_)} Hz")
            print(f"Average frequency of embodiment mapping: "
                  f"{1.0 / np.mean(pipeline.embodiment_mapping_.timings_)} Hz")
            pipeline.clear_timings()
            if fig.visualizer.poll_events():
                time.sleep(5)

    fig.show()


if __name__ == "__main__":
    main()
