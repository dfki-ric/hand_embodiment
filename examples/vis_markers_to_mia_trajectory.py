import numpy as np
from pytransform3d import visualizer as pv
from mocap.visualization import scatter
import glob
from mocap import qualisys
from mocap import pandas_utils
from mocap.cleaning import interpolate_nan, median_filter
from mocap import conversion
from hand_embodiment.record_markers import ManoHand, ManoStateEstimator
from hand_embodiment.embodiment import HandEmbodiment
from hand_embodiment.target_configurations import MIA_CONFIG


pattern = "data/Qualisys_pnp/*.tsv"
demo_idx = 2
skip_frames = 15
show_mano = False
filename = list(sorted(glob.glob(pattern)))[demo_idx]
trajectory = qualisys.read_qualisys_tsv(filename=filename)

hand_trajectory = pandas_utils.extract_markers(trajectory, ["Hand left", "Hand right", "Hand top", "Middle", "Index", "Thumb"])
hand_trajectory = hand_trajectory.iloc[::skip_frames]

hand_trajectory = median_filter(interpolate_nan(hand_trajectory), 3).iloc[2:]

hand_left = conversion.array_from_dataframe(hand_trajectory, ["Hand left X", "Hand left Y", "Hand left Z"])
hand_right = conversion.array_from_dataframe(hand_trajectory, ["Hand right X", "Hand right Y", "Hand right Z"])
hand_top = conversion.array_from_dataframe(hand_trajectory, ["Hand top X", "Hand top Y", "Hand top Z"])
middle = conversion.array_from_dataframe(hand_trajectory, ["Middle X", "Middle Y", "Middle Z"])
index = conversion.array_from_dataframe(hand_trajectory, ["Index X", "Index Y", "Index Z"])
thumb = conversion.array_from_dataframe(hand_trajectory, ["Thumb X", "Thumb Y", "Thumb Z"])


def animation_callback(t, markers, hand, mia, hse, hand_top, hand_left, hand_right, thumb, index, middle, emb):
    markers.set_data([hand_top[t], hand_left[t], hand_right[t], middle[t], index[t], thumb[t]])
    hse.estimate(
        [hand_top[t], hand_left[t], hand_right[t]],
        {"thumb": thumb[t], "index": index[t], "middle": middle[t]})
    emb.solve(hse.mano2world)
    mia.set_data()
    if show_mano:
        hand.set_data()
        return markers, hand, mia
    else:
        return markers, mia


fig = pv.figure()

fig.plot_transform(np.eye(4), s=0.5)

t = 0
marker_pos = [hand_top[t], hand_left[t], hand_right[t], thumb[t], index[t], middle[t]]
markers = scatter(fig, np.vstack([v for v in marker_pos]), s=0.005)

action_weight = 0.02
hse = ManoStateEstimator(left=False, action_weight=action_weight, verbose=1)
emb = HandEmbodiment(
    hse.hand_state, MIA_CONFIG, mano_finger_kinematics=hse.finger_estimators,
    initial_handbase2world=hse.mano2world, verbose=1)
mia = pv.Graph(
    emb.target_kin.tm, "world", show_frames=False,
    show_connections=False, show_visuals=True, show_collision_objects=False,
    show_name=False, s=0.02)
mia.add_artist(fig)
hand = ManoHand(hse)
if show_mano:
    hand.add_artist(fig)

fig.view_init()
fig.animate(animation_callback, len(hand_top), loop=True, fargs=(markers, hand, mia, hse, hand_top, hand_left, hand_right, thumb, index, middle, emb))

fig.show()
