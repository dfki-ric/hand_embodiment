import numpy as np
import pytransform3d.visualizer as pv
import pytransform3d.transformations as pt
from mocap.visualization import scatter
from hand_embodiment.record_markers import MarkerBasedRecordMapping
from hand_embodiment.vis_utils import ManoHand
from hand_embodiment.mocap_dataset import HandMotionCaptureDataset


finger_names = ["thumb", "index", "middle"]
hand_marker_names = ["Hand top", "Hand left", "Hand right"]
finger_marker_names = {"thumb": "Thumb", "index": "Index", "middle": "Middle"}
additional_marker_names = []
dataset = HandMotionCaptureDataset(
    "data/Qualisys_pnp/20151005_r_AV82_PickAndPlace_BesMan_labeled_02.tsv",
    finger_names, hand_marker_names, finger_marker_names,
    additional_marker_names, skip_frames=15)


def animation_callback(t, markers, hand, hse, dataset):
    markers.set_data(dataset.get_markers(t))
    hse.estimate(dataset.get_hand_markers(t), dataset.get_finger_markers(t))
    hand.set_data()
    return markers, hand


fig = pv.figure()

fig.plot_transform(np.eye(4), s=0.5)

markers = scatter(fig, dataset.get_markers(0), s=0.005)

mano2hand_markers = pt.transform_from_exponential_coordinates([0.048, 1.534, -0.092, -0.052, -0.031, 0.045])
betas = np.array([-2.424, -1.212, -1.869, -1.616, -4.091, -1.768, -0.808, 2.323, 1.111, 1.313])

hse = MarkerBasedRecordMapping(
    left=False, mano2hand_markers=mano2hand_markers, shape_parameters=betas,
    verbose=1)
hand = ManoHand(hse)
hand.add_artist(fig)

fig.view_init()
fig.animate(
    animation_callback, dataset.n_steps, loop=True,
    fargs=(markers, hand, hse, dataset))

fig.show()
