"""Visualize hand trajectory after embodiment mapping of a segment.

Example call:

python examples/vis_embodied_segment.py mia ../april_prototype_rl/data/dataset_grasp_insole/20210819_r_WK37_insole_set1_2.csv
"""
import argparse
import time
import numpy as np
from pytransform3d import visualizer as pv
from hand_embodiment.command_line import add_hand_argument
from hand_embodiment.target_dataset import RoboticHandDataset
from hand_embodiment.target_configurations import TARGET_CONFIG
from hand_embodiment.embodiment import load_kinematic_model


def parse_args():
    parser = argparse.ArgumentParser()
    add_hand_argument(parser)
    parser.add_argument(
        "dataset", type=str, help="Dataset that should be used.")
    parser.add_argument(
        "--delay", type=float, default=0,
        help="Delay in seconds before starting the animation")
    return parser.parse_args()


def main():
    args = parse_args()

    hand_config = TARGET_CONFIG[args.hand]
    dataset = RoboticHandDataset.import_from_file(args.dataset, hand_config)

    tm = load_kinematic_model(hand_config).tm
    tm.add_transform("world", hand_config["base_frame"], np.eye(4))

    fig = pv.figure()
    fig.plot_transform(np.eye(4), s=0.3)
    graph = pv.Graph(
        tm, "world", show_frames=True,
        whitelist=[hand_config["base_frame"]], show_connections=False,
        show_visuals=True, show_collision_objects=False, show_name=False,
        s=0.02)
    graph.add_artist(fig)
    fig.view_init(azim=45)
    fig.animate(
        animation_callback, dataset.n_steps, loop=True,
        fargs=(dataset, hand_config, tm, graph, args.delay))
    fig.show()


def animation_callback(step, dataset, hand_config, tm, graph, delay):
    if step == 1:
        time.sleep(delay)

    tm.add_transform(hand_config["base_frame"], "world", dataset.get_ee_pose(step))
    finger_joint_angles = dataset.get_finger_joint_angles(step)
    for finger in hand_config["joint_names"].keys():
        for i, joint in enumerate(hand_config["joint_names"][finger]):
            tm.set_joint(joint, finger_joint_angles[finger][i])
    graph.set_data()
    return graph


if __name__ == "__main__":
    main()
