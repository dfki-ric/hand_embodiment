import numpy as np
import pytransform3d.visualizer as pv
import pytransform3d.transformations as pt
from mocap.visualization import scatter
from hand_embodiment.record_markers import MarkerBasedRecordMapping
from hand_embodiment.vis_utils import ManoHand
from hand_embodiment.mocap_dataset import HandMotionCaptureDataset


filename = "data/QualisysAprilTest/april_test_010.tsv"
skip_frames = 1
finger_names = ["thumb", "index", "middle", "ring"]
hand_marker_names = ["hand_top", "hand_left", "hand_right"]
finger_marker_names = {"thumb": "thumb_tip", "index": "index_tip",
                       "middle": "middle_tip", "ring": "ring_tip"}
additional_marker_names = ["index_middle", "middle_middle", "ring_middle"]
dataset = HandMotionCaptureDataset(
    filename, finger_names, hand_marker_names, finger_marker_names, additional_marker_names,
    skip_frames=2)


def animation_callback(t, markers, hand, hse, dataset):
    if t == 0:
        hse.reset()
        import time
        time.sleep(5)
    markers.set_data(dataset.get_markers(t))
    hse.estimate(dataset.get_hand_markers(t), dataset.get_finger_markers(t))
    hand.set_data()
    return markers, hand


fig = pv.figure()

fig.plot_transform(np.eye(4), s=0.5)

marker_pos = dataset.get_markers(0)
markers = scatter(fig, marker_pos, s=0.005)

mano2hand_markers = pt.transform_from_exponential_coordinates(np.array([-0.103, 1.97, -0.123, -0.066, -0.034, 0.083]))
betas = np.array([-3.5, 0, 0, 0, 0, 0, 0, 0, 0, 0])

mbrm = MarkerBasedRecordMapping(
    left=False, mano2hand_markers=mano2hand_markers, shape_parameters=betas,
    verbose=1)
hand = ManoHand(mbrm, show_mesh=True, show_vertices=False)
hand.add_artist(fig)

fig.view_init(azim=45)
fig.set_zoom(0.7)
fig.animate(
    animation_callback, dataset.n_steps, loop=True,
    fargs=(markers, hand, mbrm, dataset))

fig.show()
