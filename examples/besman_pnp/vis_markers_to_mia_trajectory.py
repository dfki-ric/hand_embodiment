"""
Example call:

python examples/vis_markers_to_mia_trajectory.py mia --start-idx 8000 --end-idx 8700
"""
import argparse
import numpy as np
from pytransform3d import visualizer as pv
from pytransform3d import transformations as pt
from mocap.visualization import scatter
from hand_embodiment.record_markers import MarkerBasedRecordMapping
from hand_embodiment.vis_utils import ManoHand
from hand_embodiment.embodiment import HandEmbodiment
from hand_embodiment.target_configurations import TARGET_CONFIG
from hand_embodiment.mocap_dataset import HandMotionCaptureDataset


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "hand", type=str,
        help="Name of the hand. Possible options: mia, shadow_hand")
    parser.add_argument(
        "--start-idx", type=int, default=0, help="Start index.")
    parser.add_argument(
        "--end-idx", type=int, default=None, help="Start index.")
    parser.add_argument(
        "--show-mano", action="store_true", help="Show MANO mesh")
    parser.add_argument(
        "--skip-frames", type=int, default=15,
        help="Skip this number of frames between animated frames.")

    return parser.parse_args()


args = parse_args()


finger_names = ["thumb", "index", "middle"]
hand_marker_names = ["Hand top", "Hand left", "Hand right"]
finger_marker_names = {"thumb": "Thumb", "index": "Index", "middle": "Middle"}
additional_marker_names = []
dataset = HandMotionCaptureDataset(
    "data/Qualisys_pnp/20151005_r_AV82_PickAndPlace_BesMan_labeled_02.tsv",
    finger_names, hand_marker_names, finger_marker_names,
    additional_marker_names, skip_frames=15, start_idx=args.start_idx,
    end_idx=args.end_idx)


def animation_callback(t, markers, hand, robot, hse, emb, dataset):
    markers.set_data(dataset.get_markers(t))
    hse.estimate(dataset.get_hand_markers(t), dataset.get_finger_markers(t))
    emb.solve(hse.mano2world_, use_cached_forward_kinematics=True)
    robot.set_data()
    if args.show_mano:
        hand.set_data()
        return markers, hand, robot
    else:
        return markers, robot


hand_config = TARGET_CONFIG[args.hand]


fig = pv.figure()

fig.plot_transform(np.eye(4), s=0.5)

markers = scatter(fig, dataset.get_markers(0), s=0.005)

mano2hand_markers = pt.transform_from_exponential_coordinates([0.048, 1.534, -0.092, -0.052, -0.031, 0.045])
betas = np.array([-2.424, -1.212, -1.869, -1.616, -4.091, -1.768, -0.808, 2.323, 1.111, 1.313])

action_weight = 0.02
hse = MarkerBasedRecordMapping(
    left=False, mano2hand_markers=mano2hand_markers, shape_parameters=betas,
    verbose=1)
emb = HandEmbodiment(
    hse.hand_state_, hand_config, mano_finger_kinematics=hse.mano_finger_kinematics_,
    initial_handbase2world=hse.mano2world_, verbose=1)
robot = pv.Graph(
    emb.target_kin.tm, "world", show_frames=True, whitelist=[hand_config["base_frame"]],
    show_connections=False, show_visuals=True, show_collision_objects=False,
    show_name=False, s=0.02)
robot.add_artist(fig)
hand = ManoHand(hse)
if args.show_mano:
    hand.add_artist(fig)

fig.view_init()
fig.animate(animation_callback, dataset.n_steps, loop=True,
            fargs=(markers, hand, robot, hse, emb, dataset))

fig.show()
