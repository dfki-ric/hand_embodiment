"""Example calls:
python examples/vis_markers_to_mano_trajectory.py --demo-file data/Qualisys_pnp/20151005_r_AV82_PickAndPlace_BesMan_labeled_02.tsv --mocap-config examples/config/markers/20151005_besman.yaml --mano-config examples/config/mano/20151005_besman.yaml
python examples/vis_markers_to_mano_trajectory.py --demo-file data/QualisysAprilTest/april_test_005.tsv
python examples/vis_markers_to_mano_trajectory.py --demo-file data/20210610_april/Measurement2.tsv --mocap-config examples/config/markers/20210610_april.yaml --mano-config examples/config/mano/20210610_april.yaml
python examples/vis_markers_to_mano_trajectory.py --demo-file data/20210616_april/Measurement16.tsv --mocap-config examples/config/markers/20210616_april.yaml --mano-config examples/config/mano/20210610_april.yaml --insole
python examples/vis_markers_to_mano_trajectory.py --demo-file data/20210701_april/Measurement30.tsv --mocap-config examples/config/markers/20210616_april.yaml --mano-config examples/config/mano/20210610_april.yaml --insole
python examples/vis_markers_to_mano_trajectory.py --demo-file data/20210819_april/20210819_r_WK37_insole_set0.tsv --mocap-config examples/config/markers/20210819_april.yaml --mano-config examples/config/mano/20210610_april.yaml --insole
python examples/vis_markers_to_mano_trajectory.py --demo-file data/20210826_april/20210826_r_WK37_small_pillow_set0.tsv --mocap-config examples/config/markers/20210826_april.yaml --mano-config examples/config/mano/20210610_april.yaml --pillow
python examples/vis_markers_to_mano_trajectory.py --demo-file data/20211105_april/20211105_r_WK37_electronic_set0.tsv --mocap-config examples/config/markers/20211105_april.yaml --mano-config examples/config/mano/20210610_april.yaml --electronic
"""

import argparse
import numpy as np
import pytransform3d.visualizer as pv
from mocap.visualization import scatter

from hand_embodiment.mocap_dataset import HandMotionCaptureDataset
from hand_embodiment.pipelines import MoCapToRobot
from hand_embodiment.vis_utils import AnimationCallback

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
    parser.add_argument(
        "--pillow", action="store_true", help="Visualize pillow.")
    parser.add_argument(
        "--electronic", action="store_true",
        help="Visualize electronic components.")

    return parser.parse_args()


def main():
    args = parse_args()

    dataset = HandMotionCaptureDataset(
        args.demo_file, mocap_config=args.mocap_config,
        skip_frames=args.skip_frames, start_idx=args.start_idx,
        end_idx=args.end_idx)

    pipeline = MoCapToRobot("mia", args.mano_config, dataset.finger_names,
                            verbose=1)

    fig = pv.figure()
    fig.plot_transform(np.eye(4), s=0.5)
    markers = dataset.get_markers(0)
    markers = scatter(fig, markers, s=0.006, c=MARKER_COLORS[:len(markers)])

    animation_callback = AnimationCallback(fig, pipeline, args)

    fig.view_init(azim=45)
    fig.animate(
        animation_callback, dataset.n_steps, loop=True,
        fargs=(markers, dataset, pipeline))

    fig.show()


if __name__ == "__main__":
    main()
