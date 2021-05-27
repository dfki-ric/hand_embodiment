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


skip_frames = 5
filename = "data/QualisysAprilTest/april_test_013.tsv"
trajectory = qualisys.read_qualisys_tsv(filename=filename)

marker_names = ["hand_left", "hand_right", "hand_top", "ring_middle", "middle_middle", "index_middle", "ring_tip", "middle_tip", "index_tip", "thumb_tip"]
hand_trajectory = pandas_utils.extract_markers(trajectory, marker_names).copy()
column_names = pandas_utils.match_columns(hand_trajectory, marker_names, keep_time=False)
for column_name in column_names:
    hand_trajectory[column_name].replace(0.0, np.nan, inplace=True)
hand_trajectory = hand_trajectory.iloc[::skip_frames]

hand_trajectory = median_filter(interpolate_nan(hand_trajectory), 3).iloc[2:]

hand_left = conversion.array_from_dataframe(hand_trajectory, ["hand_left X", "hand_left Y", "hand_left Z"])
hand_right = conversion.array_from_dataframe(hand_trajectory, ["hand_right X", "hand_right Y", "hand_right Z"])
hand_top = conversion.array_from_dataframe(hand_trajectory, ["hand_top X", "hand_top Y", "hand_top Z"])
ring_middle = conversion.array_from_dataframe(hand_trajectory, ["ring_middle X", "ring_middle Y", "ring_middle Z"])
middle_middle = conversion.array_from_dataframe(hand_trajectory, ["middle_middle X", "middle_middle Y", "middle_middle Z"])
index_middle = conversion.array_from_dataframe(hand_trajectory, ["index_middle X", "index_middle Y", "index_middle Z"])
ring = conversion.array_from_dataframe(hand_trajectory, ["ring_tip X", "ring_tip Y", "ring_tip Z"])
middle = conversion.array_from_dataframe(hand_trajectory, ["middle_tip X", "middle_tip Y", "middle_tip Z"])
index = conversion.array_from_dataframe(hand_trajectory, ["index_tip X", "index_tip Y", "index_tip Z"])
thumb = conversion.array_from_dataframe(hand_trajectory, ["thumb_tip X", "thumb_tip Y", "thumb_tip Z"])


def animation_callback(t, markers, hand, robot, hse, hand_top, hand_left, hand_right, thumb, index, middle, ring, emb):
    if t == 0:
        hse.reset()
        import time
        time.sleep(1)
    markers.set_data([hand_top[t], hand_left[t], hand_right[t], middle[t], index[t], thumb[t], ring[t]])
    hse.estimate(
        [hand_top[t], hand_left[t], hand_right[t]],
        {"thumb": thumb[t], "index": index[t], "middle": middle[t], "ring": ring[t]})
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
marker_pos = [hand_top[t], hand_left[t], hand_right[t], thumb[t], index[t], middle[t], ring[t]]
markers = scatter(fig, np.vstack([v for v in marker_pos]), s=0.005)

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
    animation_callback, len(hand_top), loop=True,
    fargs=(markers, hand, robot, mbrm, hand_top, hand_left, hand_right, thumb,
           index, middle, ring, emb))

fig.show()
