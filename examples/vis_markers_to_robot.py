"""Example calls:
python examples/vis_markers_to_robot.py mia --demo-file data/Qualisys_pnp/20151005_r_AV82_PickAndPlace_BesMan_labeled_02.tsv --mocap-config examples/config/markers/20151005_besman.yaml --mano-config examples/config/mano/20151005_besman.yaml
python examples/vis_markers_to_robot.py mia --demo-file data/QualisysAprilTest/april_test_005.tsv
python examples/vis_markers_to_robot.py mia --demo-file data/20210610_april/Measurement2.tsv --mocap-config examples/config/markers/20210610_april.yaml --mano-config examples/config/mano/20210610_april.yaml --mia-thumb-adducted
python examples/vis_markers_to_robot.py mia --demo-file data/20210616_april/Measurement16.tsv --mocap-config examples/config/markers/20210616_april.yaml --mano-config examples/config/mano/20210610_april.yaml --skip-frames 1 --show-mano
"""

import argparse
import time
import numpy as np
from pytransform3d import visualizer as pv
from mocap.visualization import scatter
from hand_embodiment.record_markers import MarkerBasedRecordMapping
from hand_embodiment.vis_utils import ManoHand
from hand_embodiment.embodiment import HandEmbodiment
from hand_embodiment.target_configurations import TARGET_CONFIG
from hand_embodiment.mocap_dataset import HandMotionCaptureDataset
from hand_embodiment.config import load_mano_config


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "hand", type=str,
        help="Name of the hand. Possible options: mia, shadow_hand")
    parser.add_argument(
        "--demo-file", type=str,
        default="data/QualisysAprilTest/april_test_010.tsv",
        help="Demonstration that should be used.")
    parser.add_argument(
        "--mocap-config", type=str,
        default="examples/config/markers/20210520_april.yaml",
        help="MoCap configuration file.")
    parser.add_argument(
        "--mano-config", type=str,
        default="examples/config/mano/20210520_april.yaml",
        help="MANO configuration file.")
    parser.add_argument(
        "--start-idx", type=int, default=0, help="Start index.")
    parser.add_argument(
        "--end-idx", type=int, default=None, help="Start index.")
    parser.add_argument(
        "--show-mano", action="store_true", help="Show MANO mesh")
    parser.add_argument(
        "--skip-frames", type=int, default=15,
        help="Skip this number of frames between animated frames.")
    parser.add_argument(
        "--demo-idx", type=int, default=2,
        help="Index of demonstration that should be used.")
    parser.add_argument(
        "--mia-thumb-adducted", action="store_true",
        help="Adduct thumb of Mia hand.")
    parser.add_argument(
        "--delay", type=float, default=0,
        help="Delay in seconds before starting the animation")

    return parser.parse_args()


def animation_callback(t, markers, hand, robot, hse, dataset, emb, delay):
    if t == 1:
        hse.reset()
        time.sleep(delay)
    markers.set_data(dataset.get_markers(t))
    hse.estimate(dataset.get_hand_markers(t), dataset.get_finger_markers(t))
    emb.solve(hse.mano2world_, use_cached_forward_kinematics=True)
    robot.set_data()
    if hand is not None:
        hand.set_data()
        return markers, hand, robot
    else:
        return markers, robot


def main():
    args = parse_args()

    dataset = HandMotionCaptureDataset(
        args.demo_file, mocap_config=args.mocap_config,
        skip_frames=args.skip_frames, start_idx=args.start_idx,
        end_idx=args.end_idx)

    hand_config = TARGET_CONFIG[args.hand]

    mano2hand_markers, betas = load_mano_config(args.mano_config)
    mbrm = MarkerBasedRecordMapping(
        left=False, mano2hand_markers=mano2hand_markers,
        shape_parameters=betas, verbose=1)
    emb = HandEmbodiment(
        mbrm.hand_state_, hand_config,
        use_fingers=dataset.finger_names,
        mano_finger_kinematics=mbrm.mano_finger_kinematics_,
        initial_handbase2world=mbrm.mano2world_, verbose=1)
    if args.hand == "mia":
        if args.mia_thumb_adducted:
            emb.target_kin.tm.set_joint("j_thumb_opp_binary", 1.0)
        else:
            emb.target_kin.tm.set_joint("j_thumb_opp_binary", -1.0)

    fig = pv.figure()
    fig.plot_transform(np.eye(4), s=1)
    markers = scatter(fig, dataset.get_markers(0), s=0.006)
    robot = pv.Graph(
        emb.target_kin.tm, "world", show_frames=True,
        whitelist=[hand_config["base_frame"]], show_connections=False,
        show_visuals=True, show_collision_objects=False, show_name=False,
        s=0.02)
    robot.add_artist(fig)
    if args.show_mano:
        hand = ManoHand(mbrm, show_mesh=True, show_vertices=False)
        hand.add_artist(fig)
    else:
        hand = None

    fig.view_init(azim=45)
    fig.set_zoom(0.3)
    fig.animate(
        animation_callback, dataset.n_steps, loop=True,
        fargs=(markers, hand, robot, mbrm, dataset, emb, args.delay))

    fig.show()


if __name__ == "__main__":
    main()
