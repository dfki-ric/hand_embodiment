"""Example calls:
python examples/vis_markers_to_mano_trajectory.py --demo-file data/Qualisys_pnp/20151005_r_AV82_PickAndPlace_BesMan_labeled_02.tsv --mocap-config examples/config/markers/20151005_besman.yaml --mano-config examples/config/mano/20151005_besman.yaml
python examples/vis_markers_to_mano_trajectory.py --demo-file data/QualisysAprilTest/april_test_005.tsv
python examples/vis_markers_to_mano_trajectory.py --demo-file data/20210610_april/Measurement2.tsv --mocap-config examples/config/markers/20210610_april.yaml --mano-config examples/config/mano/20210610_april.yaml
"""

import argparse
import time
import numpy as np
import pytransform3d.visualizer as pv
from mocap.visualization import scatter

from hand_embodiment.record_markers import MarkerBasedRecordMapping
from hand_embodiment.vis_utils import ManoHand
from hand_embodiment.mocap_dataset import HandMotionCaptureDataset
from hand_embodiment.config import load_mano_config


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--demo-file", type=str,
        default="data/QualisysAprilTest/april_test_010.tsv",
        help="Demonstration that should be used.")
    parser.add_argument(
        "--mocap-config", type=str,
        default="examples/config/markers/20210520_april.yaml",
        help="MoCap configuration file.")
    parser.add_argument(
        "--mano-config", type=str,
        default="examples/config/mano/20210520_april.yaml",
        help="MANO configuration file.")
    parser.add_argument(
        "--start-idx", type=int, default=None, help="Start index.")
    parser.add_argument(
        "--end-idx", type=int, default=None, help="Start index.")
    parser.add_argument(
        "--hide-mano", action="store_true", help="Hide MANO mesh")
    parser.add_argument(
        "--skip-frames", type=int, default=1,
        help="Skip this number of frames between animated frames.")
    parser.add_argument(
        "--delay", type=float, default=0,
        help="Delay in seconds before starting the animation")

    return parser.parse_args()


def animation_callback(t, markers, hand, mbrm, dataset, delay):
    if t == 1:
        mbrm.reset()
        time.sleep(delay)
    markers.set_data(dataset.get_markers(t))
    if hand is not None:
        mbrm.estimate(dataset.get_hand_markers(t), dataset.get_finger_markers(t))
        hand.set_data()
        return markers, hand
    else:
        return markers


def main():
    args = parse_args()

    dataset = HandMotionCaptureDataset(
        args.demo_file, mocap_config=args.mocap_config,
        skip_frames=args.skip_frames, start_idx=args.start_idx,
        end_idx=args.end_idx)

    mano2hand_markers, betas = load_mano_config(args.mano_config)
    mbrm = MarkerBasedRecordMapping(
        left=False, mano2hand_markers=mano2hand_markers, shape_parameters=betas,
        verbose=1)

    fig = pv.figure()
    fig.plot_transform(np.eye(4), s=1)
    markers = scatter(fig, dataset.get_markers(0), s=0.006)

    if args.hide_mano:
        hand = None
    else:
        hand = ManoHand(mbrm, show_mesh=True, show_vertices=False)
        hand.add_artist(fig)

    fig.view_init(azim=45)
    fig.set_zoom(0.3)
    fig.animate(
        animation_callback, dataset.n_steps, loop=True,
        fargs=(markers, hand, mbrm, dataset, args.delay))

    fig.show()


if __name__ == "__main__":
    main()
