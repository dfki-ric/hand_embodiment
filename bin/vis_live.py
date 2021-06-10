import json
import numpy as np
import pytransform3d.visualizer as pv
from mocap.visualization import scatter
from hand_embodiment.record_markers import MarkerBasedRecordMapping
from hand_embodiment.config import load_mano_config
from hand_embodiment.vis_utils import ManoHand


def animation_callback(step, markers, mbrm, hand):
    try:
        with open("comm.json", "r") as f:
            result = json.load(f)
    except:
        return []

    marker_pos = np.empty((len(result), 3))
    for i, marker_name in enumerate(result):
        marker_pos[i] = result[marker_name]

    # Filter positions
    nan = np.isnan(marker_pos)
    valid = not np.any(nan)
    marker_pos[nan] = 0.0

    markers.set_data(marker_pos)

    if not valid:
        return markers

    # TODO Markers on hand in order 'hand_top', 'hand_left', 'hand_right'.
    hand_markers = [np.array(result["hand_top"]),
                    np.array(result["hand_left"]),
                    np.array(result["hand_right"])]
    # TODO Positions of markers on fingers.
    finger_markers = {
        "thumb": np.array(result["thumb_tip"]),
        "index": np.array(result["index_tip"]),
        "middle": np.array(result["middle_tip"]),
        "ring": np.array(result["ring_tip"]),
        "little": np.array(result["little_tip"])
    }
    mbrm.estimate(hand_markers, finger_markers)

    hand.set_data()

    return markers, hand


mano2hand_markers, betas = load_mano_config(
    "examples/config/april_test_mano.yaml")
mbrm = MarkerBasedRecordMapping(
    left=False, mano2hand_markers=mano2hand_markers,
    shape_parameters=betas, verbose=1)
hand = ManoHand(mbrm, show_mesh=True, show_vertices=False)

fig = pv.figure()
fig.plot_transform(s=0.5)
markers = scatter(fig, np.zeros((13, 3)), s=0.006)
hand.add_artist(fig)
fig.view_init(azim=-70)
fig.animate(
    animation_callback, 1, loop=True,
    fargs=(markers, mbrm, hand))
fig.show()
