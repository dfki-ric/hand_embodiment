"""
Example call:

python examples/vis_markers_to_mia_trajectory.py mia --start-idx 8000 --end-idx 8700
"""
import argparse
import numpy as np
from pytransform3d import visualizer as pv
from pytransform3d import transformations as pt
from mocap.visualization import scatter
import glob
from mocap import qualisys
from mocap import pandas_utils
from mocap.cleaning import interpolate_nan, median_filter
from mocap import conversion
from hand_embodiment.record_markers import ManoHand, MarkerBasedRecordMapping
from hand_embodiment.embodiment import HandEmbodiment
from hand_embodiment.target_configurations import MIA_CONFIG, SHADOW_HAND_CONFIG


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


pattern = "data/Qualisys_pnp/*.tsv"
filename = list(sorted(glob.glob(pattern)))[args.demo_idx]
trajectory = qualisys.read_qualisys_tsv(filename=filename)

hand_trajectory = pandas_utils.extract_markers(trajectory, ["Hand left", "Hand right", "Hand top", "Middle", "Index", "Thumb"])
hand_trajectory = hand_trajectory.iloc[
                  args.start_idx:args.end_idx:args.skip_frames]

hand_trajectory = median_filter(interpolate_nan(hand_trajectory), 3).iloc[2:]

hand_left = conversion.array_from_dataframe(hand_trajectory, ["Hand left X", "Hand left Y", "Hand left Z"])
hand_right = conversion.array_from_dataframe(hand_trajectory, ["Hand right X", "Hand right Y", "Hand right Z"])
hand_top = conversion.array_from_dataframe(hand_trajectory, ["Hand top X", "Hand top Y", "Hand top Z"])
middle = conversion.array_from_dataframe(hand_trajectory, ["Middle X", "Middle Y", "Middle Z"])
index = conversion.array_from_dataframe(hand_trajectory, ["Index X", "Index Y", "Index Z"])
thumb = conversion.array_from_dataframe(hand_trajectory, ["Thumb X", "Thumb Y", "Thumb Z"])


def animation_callback(t, markers, hand, robot, hse, hand_top, hand_left, hand_right, thumb, index, middle, emb):
    markers.set_data([hand_top[t], hand_left[t], hand_right[t], middle[t], index[t], thumb[t]])
    hse.estimate(
        [hand_top[t], hand_left[t], hand_right[t]],
        {"thumb": thumb[t], "index": index[t], "middle": middle[t]})
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

t = 0
marker_pos = [hand_top[t], hand_left[t], hand_right[t], thumb[t], index[t], middle[t]]
markers = scatter(fig, np.vstack([v for v in marker_pos]), s=0.005)

mano2hand_markers = pt.transform_from_exponential_coordinates([0.048, 1.534, -0.092, -0.052, -0.031, 0.045])
betas = np.array([-2.424, -1.212, -1.869, -1.616, -4.091, -1.768, -0.808, 2.323, 1.111, 1.313])

action_weight = 0.02
hse = MarkerBasedRecordMapping(
    left=False, action_weight=action_weight,
    mano2hand_markers=mano2hand_markers, shape_parameters=betas, verbose=1)
emb = HandEmbodiment(
    hse.hand_state_, hand_config, mano_finger_kinematics=hse.mano_finger_kinematics_,
    initial_handbase2world=hse.mano2world_, verbose=1)
robot = pv.Graph(
    emb.target_kin.tm, "world", show_frames=False,
    show_connections=False, show_visuals=True, show_collision_objects=False,
    show_name=False, s=0.02)
robot.add_artist(fig)
hand = ManoHand(hse)
if args.show_mano:
    hand.add_artist(fig)

fig.view_init()
fig.animate(animation_callback, len(hand_top), loop=True, fargs=(markers, hand, robot, hse, hand_top, hand_left, hand_right, thumb, index, middle, emb))

fig.show()
