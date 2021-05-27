"""
Example call:

python examples/vis_markers_to_mia_trajectory.py mia --start-idx 8000 --end-idx 8700
"""
import argparse
import numpy as np
from pytransform3d import visualizer as pv
from pytransform3d import transformations as pt
from mocap.visualization import scatter
from mocap import qualisys
from mocap import pandas_utils
from mocap.cleaning import interpolate_nan, median_filter
from mocap import conversion
from hand_embodiment.record_markers import MarkerBasedRecordMapping
from hand_embodiment.vis_utils import ManoHand
from hand_embodiment.embodiment import HandEmbodiment
from hand_embodiment.target_configurations import MIA_CONFIG, SHADOW_HAND_CONFIG
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
    parser.add_argument(
        "--demo-idx", type=int, default=2,
        help="Index of demonstration that should be used.")

    return parser.parse_args()


args = parse_args()


filename = "data/QualisysAprilTest/april_test_004.tsv"
skip_frames = 1
finger_names = ["thumb", "index", "middle", "ring"]
hand_marker_names = ["hand_top", "hand_left", "hand_right"]
finger_marker_names = {"thumb": "thumb_tip", "index": "index_tip",
                       "middle": "middle_tip", "ring": "ring_tip"}
additional_marker_names = ["index_middle", "middle_middle", "ring_middle"]
dataset = HandMotionCaptureDataset(
    filename, finger_names, hand_marker_names, finger_marker_names, additional_marker_names,
    skip_frames=5)


def animation_callback(t, markers, hand, robot, hse, dataset, emb):
    if t == 0:
        hse.reset()
        import time
        time.sleep(1)
    markers.set_data(dataset.get_markers(t))
    hse.estimate(dataset.get_hand_markers(t), dataset.get_finger_markers(t))
    emb.solve(hse.mano2world_, use_cached_forward_kinematics=True)
    robot.set_data()
    if args.show_mano:
        hand.set_data()
        return markers, hand, robot
    else:
        return markers, robot


if args.hand == "shadow_hand":
    hand_config = SHADOW_HAND_CONFIG
elif args.hand == "mia":
    hand_config = MIA_CONFIG
else:
    raise Exception(f"Unknown hand: '{args.hand}'")


fig = pv.figure()

fig.plot_transform(np.eye(4), s=0.5)

marker_pos = dataset.get_markers(0)
markers = scatter(fig, marker_pos, s=0.005)

mano2hand_markers = pt.transform_from_exponential_coordinates(np.array([-0.103, 1.97, -0.123, -0.066, -0.034, 0.083]))
betas = np.array([-3.5, 0, 0, 0, 0, 0, 0, 0, 0, 0])

mbrm = MarkerBasedRecordMapping(
    left=False, mano2hand_markers=mano2hand_markers, shape_parameters=betas,
    verbose=1)
emb = HandEmbodiment(
    mbrm.hand_state_, hand_config,
    use_fingers=("thumb", "index", "middle", "ring"),
    mano_finger_kinematics=mbrm.mano_finger_kinematics_,
    initial_handbase2world=mbrm.mano2world_, verbose=1)
robot = pv.Graph(
    emb.target_kin.tm, "world", show_frames=True, whitelist=[hand_config["base_frame"]],
    show_connections=False, show_visuals=True, show_collision_objects=False,
    show_name=False, s=0.02)
robot.add_artist(fig)
hand = ManoHand(mbrm, show_mesh=True, show_vertices=False)
if args.show_mano:
    hand.add_artist(fig)

fig.view_init(azim=45)
fig.set_zoom(0.7)
fig.animate(
    animation_callback, dataset.n_steps, loop=True,
    fargs=(markers, hand, robot, mbrm, dataset, emb))

fig.show()
