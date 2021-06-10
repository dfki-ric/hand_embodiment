import json
import numpy as np
import pytransform3d.visualizer as pv
from hand_embodiment.record_markers import MarkerBasedRecordMapping
from hand_embodiment.config import load_mano_config
from mocap.visualization import scatter


def animation_callback(step, markers, mbrm):
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

    return markers


mano2hand_markers, betas = load_mano_config(
    "examples/config/april_test_mano.yaml")
mbrm = MarkerBasedRecordMapping(
    left=False, mano2hand_markers=mano2hand_markers,
    shape_parameters=betas, verbose=1)

fig = pv.figure()
fig.plot_transform(s=0.5)
markers = scatter(fig, np.zeros((13, 3)), s=0.005)
fig.view_init(azim=-70)
fig.animate(
    animation_callback, 1, loop=True,
    fargs=(markers, mbrm))
fig.show()
