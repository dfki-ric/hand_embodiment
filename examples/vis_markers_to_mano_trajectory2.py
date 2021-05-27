import numpy as np
import pytransform3d.visualizer as pv
import pytransform3d.transformations as pt
from mocap.visualization import scatter
from mocap import qualisys
from mocap import pandas_utils
from mocap.cleaning import interpolate_nan, median_filter
from mocap import conversion
from hand_embodiment.record_markers import ManoHand, MarkerBasedRecordMapping


skip_frames = 1
filename = "data/QualisysAprilTest/april_test_007.tsv"
trajectory = qualisys.read_qualisys_tsv(filename=filename)

hand_trajectory = pandas_utils.extract_markers(trajectory, ["hand_left", "hand_right", "hand_top", "ring_middle", "middle_middle", "index_middle", "ring_tip", "middle_tip", "index_tip", "thumb_tip"])
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


def animation_callback(t, markers, hand, hse, hand_top, hand_left, hand_right, thumb, index, middle, ring):
    markers.set_data([hand_top[t], hand_left[t], hand_right[t], middle[t], index[t], thumb[t], ring[t]])
    hse.estimate(
        [hand_top[t], hand_left[t], hand_right[t]],
        {"thumb": thumb[t], "index": index[t], "middle": middle[t], "ring": ring[t]})
    hand.set_data()
    return markers, hand


fig = pv.figure()

fig.plot_transform(np.eye(4), s=0.5)

t = 0
marker_pos = [hand_top[t], hand_left[t], hand_right[t], thumb[t], index[t], middle[t], ring[t]]
markers = scatter(fig, np.vstack([v for v in marker_pos]), s=0.005)

mano2hand_markers = pt.transform_from_exponential_coordinates(np.array([-0.103, 1.97, -0.123, -0.066, -0.034, 0.083]))
betas = np.array([-3.5, 0, 0, 0, 0, 0, 0, 0, 0, 0])

hse = MarkerBasedRecordMapping(
    left=False, mano2hand_markers=mano2hand_markers, shape_parameters=betas,
    verbose=1)
hand = ManoHand(hse)
hand.add_artist(fig)

fig.view_init()
fig.animate(
    animation_callback, len(hand_top), loop=True,
    fargs=(markers, hand, hse, hand_top, hand_left, hand_right, thumb, index,
           middle, ring))

fig.show()
