import argparse
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
        "--start-idx", type=int, default=None, help="Start index.")
    parser.add_argument(
        "--end-idx", type=int, default=None, help="Start index.")
    parser.add_argument(
        "--hide-mano", action="store_true", help="Hide MANO mesh")
    parser.add_argument(
        "--skip-frames", type=int, default=1,
        help="Skip this number of frames between animated frames.")

    return parser.parse_args()


def main():
    args = parse_args()

    dataset = HandMotionCaptureDataset(
        args.demo_file, mocap_config=args.mocap_config,
        skip_frames=args.skip_frames, start_idx=args.start_idx,
        end_idx=args.end_idx)

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

    mano2hand_markers, betas = load_mano_config(
        "examples/config/april_test_mano.yaml")
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


if __name__ == "__main__":
    main()
