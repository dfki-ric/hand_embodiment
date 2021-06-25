"""Example calls:
python examples/vis_segment_to_robot.py mia close --mocap-config examples/config/markers/20210616_april.yaml --mano-config examples/config/mano/20210616_april.yaml --demo-file data/20210616_april/metadata/Measurement24.json --segment 0
"""

import argparse
import time
import numpy as np
from pytransform3d import visualizer as pv
from mocap.visualization import scatter
from hand_embodiment.mocap_dataset import SegmentedHandMotionCaptureDataset
from hand_embodiment.pipelines import MoCapToRobot


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
        default="data/QualisysAprilTest/april_test_010.tsv",
        help="Demonstration that should be used.")
    parser.add_argument(
        "--segment", type=int, default=0,
        help="Segment of demonstration that should be used.")
    parser.add_argument(
        "--mocap-config", type=str,
        default="examples/config/markers/20210520_april.yaml",
        help="MoCap configuration file.")
    parser.add_argument(
        "--mano-config", type=str,
        default="examples/config/mano/20210520_april.yaml",
        help="MANO configuration file.")
    parser.add_argument(
        "--show-mano", action="store_true", help="Show MANO mesh")
    parser.add_argument(
        "--mia-thumb-adducted", action="store_true",
        help="Adduct thumb of Mia hand.")
    parser.add_argument(
        "--delay", type=float, default=0,
        help="Delay in seconds before starting the animation")

    return parser.parse_args()


def animation_callback(t, markers, hand, robot, dataset, pipeline, delay):
    if t == 1:
        pipeline.reset()
        time.sleep(delay)

    pipeline.estimate(dataset.get_hand_markers(t),
                      dataset.get_finger_markers(t))

    markers.set_data(dataset.get_markers(t))
    robot.set_data()
    if hand is not None:
        hand.set_data()
        return markers, hand, robot
    else:
        return markers, robot


def main():
    args = parse_args()

    dataset = SegmentedHandMotionCaptureDataset(
        args.demo_file, args.segment_label, mocap_config=args.mocap_config)
    dataset.select_segment(args.segment)

    pipeline = MoCapToRobot(args.hand, args.mano_config, dataset.finger_names,
                            verbose=1)

    if args.hand == "mia":
        angle = 1.0 if args.mia_thumb_adducted else -1.0
        pipeline.set_constant_joint("j_thumb_opp_binary", angle)

    fig = pv.figure()
    fig.plot_transform(np.eye(4), s=1)
    markers = scatter(fig, dataset.get_markers(0), s=0.006)
    robot = pipeline.make_robot_artist()
    robot.add_artist(fig)
    if args.show_mano:
        hand = pipeline.make_hand_artist()
        hand.add_artist(fig)
    else:
        hand = None

    fig.view_init(azim=45)
    fig.set_zoom(0.3)
    fig.animate(
        animation_callback, dataset.n_steps, loop=True,
        fargs=(markers, hand, robot, dataset, pipeline, args.delay))

    fig.show()


if __name__ == "__main__":
    main()
