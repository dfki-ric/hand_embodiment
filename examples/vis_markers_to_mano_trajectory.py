"""Example calls:
python examples/vis_markers_to_mano_trajectory.py --demo-file data/Qualisys_pnp/20151005_r_AV82_PickAndPlace_BesMan_labeled_02.tsv --mocap-config examples/config/markers/20151005_besman.yaml --mano-config examples/config/mano/20151005_besman.yaml
python examples/vis_markers_to_mano_trajectory.py --demo-file data/QualisysAprilTest/april_test_005.tsv
python examples/vis_markers_to_mano_trajectory.py --demo-file data/20210610_april/Measurement2.tsv --mocap-config examples/config/markers/20210610_april.yaml --mano-config examples/config/mano/20210610_april.yaml
python examples/vis_markers_to_mano_trajectory.py --demo-file data/20210616_april/Measurement16.tsv --mocap-config examples/config/markers/20210616_april.yaml --mano-config examples/config/mano/20210610_april.yaml --insole
"""

import argparse
import time
import numpy as np
import pytransform3d.visualizer as pv
from mocap.visualization import scatter

from hand_embodiment.mocap_dataset import HandMotionCaptureDataset
from hand_embodiment.pipelines import MoCapToRobot
from hand_embodiment.vis_utils import Insole


MARKER_COLORS = [
    (0.5, 0.5, 0.5), (0.5, 0.5, 0.5), (0.5, 0.5, 0.5),
    (1, 0, 0), (0.5, 0, 0),
    (0, 1, 0), (0, 0.5, 0),
    (0, 0, 1), (0, 0, 0.5),
    (1, 1, 0), (0.5, 0.5, 0),
    (0, 1, 1), (0, 0.5, 0.5),
    (1, 0, 1), (0.5, 0, 0.5),
    (0.5, 0.5, 0.5), (0.5, 0.5, 0.5), (0.5, 0.5, 0.5),
    (0.5, 0.5, 0.5), (0.5, 0.5, 0.5), (0.5, 0.5, 0.5),
    (0.5, 0.5, 0.5), (0.5, 0.5, 0.5), (0.5, 0.5, 0.5),
]


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
    parser.add_argument(
        "--insole", action="store_true", help="Visualize insole mesh.")

    return parser.parse_args()


class AnimationCallback:
    def __init__(self, fig, pipeline, args):
        self.fig = fig
        self.args = args

        if args.hide_mano:
            self.hand = None
        else:
            self.hand = pipeline.make_hand_artist()
            self.hand.add_artist(self.fig)

        if self.args.insole:
            self.mesh = Insole()
            self.mesh.add_artist(self.fig)

    def __call__(self, t, markers, dataset, pipeline):
        if t == 1:
            pipeline.reset()
            time.sleep(self.args.delay)

        markers.set_data(dataset.get_markers(t))

        artists = [markers]

        if self.args.insole:
            marker_names = dataset.config.get("additional_markers", ())
            additional_markers = dataset.get_additional_markers(t)
            insole_back = additional_markers[marker_names.index("insole_back")]
            insole_front = additional_markers[marker_names.index("insole_front")]
            self.mesh.set_data(insole_back, insole_front)
            artists.append(self.mesh)

        if not self.args.hide_mano:
            pipeline.estimate_hand(
                dataset.get_hand_markers(t), dataset.get_finger_markers(t))
            self.hand.set_data()
            artists.append(self.hand)

        return artists


def main():
    args = parse_args()

    dataset = HandMotionCaptureDataset(
        args.demo_file, mocap_config=args.mocap_config,
        skip_frames=args.skip_frames, start_idx=args.start_idx,
        end_idx=args.end_idx)

    pipeline = MoCapToRobot("mia", args.mano_config, dataset.finger_names,
                            verbose=1)

    fig = pv.figure()
    fig.plot_transform(np.eye(4), s=1)
    markers = dataset.get_markers(0)
    markers = scatter(fig, markers, s=0.006, c=MARKER_COLORS[:len(markers)])

    animation_callback = AnimationCallback(fig, pipeline, args)

    fig.view_init(azim=45)
    fig.set_zoom(0.3)
    fig.animate(
        animation_callback, dataset.n_steps, loop=True,
        fargs=(markers, dataset, pipeline))

    fig.show()


if __name__ == "__main__":
    main()
