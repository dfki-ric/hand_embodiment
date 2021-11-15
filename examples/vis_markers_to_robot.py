"""Example calls:
python examples/vis_markers_to_robot.py mia --demo-file data/Qualisys_pnp/20151005_r_AV82_PickAndPlace_BesMan_labeled_02.tsv --mocap-config examples/config/markers/20151005_besman.yaml --mano-config examples/config/mano/20151005_besman.yaml
python examples/vis_markers_to_robot.py mia --demo-file data/QualisysAprilTest/april_test_005.tsv
python examples/vis_markers_to_robot.py mia --demo-file data/20210610_april/Measurement2.tsv --mocap-config examples/config/markers/20210610_april.yaml --mano-config examples/config/mano/20210610_april.yaml --mia-thumb-adducted
python examples/vis_markers_to_robot.py mia --mocap-config examples/config/markers/20210616_april.yaml --mano-config examples/config/mano/20210616_april.yaml --skip-frames 1 --show-mano --demo-file data/20210616_april/Measurement24.tsv --mia-thumb-adducted
python examples/vis_markers_to_robot.py mia --mocap-config examples/config/markers/20210616_april.yaml --mano-config examples/config/mano/20210616_april.yaml --skip-frames 1 --demo-file data/20210701_april/Measurement30.tsv --insole --mia-thumb-adducted
python examples/vis_markers_to_robot.py mia --mocap-config examples/config/markers/20210819_april.yaml --mano-config examples/config/mano/20210616_april.yaml --skip-frames 1 --demo-file data/20210819_april/20210819_r_WK37_insole_set0.tsv --insole --show-mano --mia-thumb-adducted
python examples/vis_markers_to_robot.py mia --mocap-config examples/config/markers/20210826_april.yaml --mano-config examples/config/mano/20210610_april.yaml --skip-frames 1 --demo-file data/20210826_april/20210826_r_WK37_small_pillow_set0.tsv --show-mano --mia-thumb-adducted
python examples/vis_markers_to_robot.py shadow --mocap-config examples/config/markers/20211105_april.yaml --mano-config examples/config/mano/20210610_april.yaml --record-mapping-config examples/config/record_mapping/20211105_april.yaml --skip-frames 1 --demo-file data/20211105_april/20211105_r_WK37_electronic_set0.tsv --show-mano --mia-thumb-adducted --electronic
python examples/vis_markers_to_robot.py shadow --mocap-config examples/config/markers/20211112_april.yaml --mano-config examples/config/mano/20210610_april.yaml --record-mapping-config examples/config/record_mapping/20211105_april.yaml --skip-frames 1 --demo-file data/20211112_april/20211112_r_WK37_passport_set0.tsv --show-mano --passport
"""

import argparse
import time

import numpy as np
from pytransform3d import visualizer as pv
from mocap.visualization import scatter
from hand_embodiment.mocap_dataset import HandMotionCaptureDataset
from hand_embodiment.pipelines import MoCapToRobot
from hand_embodiment.vis_utils import AnimationCallback
from hand_embodiment.command_line import (
    add_animation_arguments, add_configuration_arguments,
    add_playback_control_arguments)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "hand", type=str,
        help="Name of the hand. Possible options: mia, shadow_hand")
    parser.add_argument(
        "--demo-file", type=str,
        default="data/QualisysAprilTest/april_test_010.tsv",
        help="Demonstration that should be used.")
    add_configuration_arguments(parser)
    add_playback_control_arguments(parser)
    parser.add_argument(
        "--show-mano", action="store_true", help="Show MANO mesh")
    parser.add_argument(
        "--mia-thumb-adducted", action="store_true",
        help="Adduct thumb of Mia hand.")
    parser.add_argument(
        "--measure-time", action="store_true",
        help="Measure time of record and embodiment mapping.")
    add_animation_arguments(parser)
    return parser.parse_args()


def main():
    args = parse_args()

    dataset = HandMotionCaptureDataset(
        args.demo_file, mocap_config=args.mocap_config,
        skip_frames=args.skip_frames, start_idx=args.start_idx,
        end_idx=args.end_idx)

    pipeline = MoCapToRobot(args.hand, args.mano_config, dataset.finger_names,
                            record_mapping_config=args.record_mapping_config,
                            verbose=1, measure_time=args.measure_time)

    if args.hand == "mia":
        angle = 1.0 if args.mia_thumb_adducted else -1.0
        pipeline.set_constant_joint("j_thumb_opp_binary", angle)

    fig = pv.figure()
    fig.plot_transform(np.eye(4), s=0.5)
    markers = scatter(fig, dataset.get_markers(0), s=0.006)

    animation_callback = AnimationCallback(fig, pipeline, args, show_robot=True)
    fig.view_init(azim=45)
    while fig.visualizer.poll_events():
        fig.animate(
            animation_callback, dataset.n_steps, loop=False,
            fargs=(markers, dataset, pipeline))
        if args.measure_time:
            print(f"Average frequency of record mapping: "
                  f"{1.0 / np.mean(pipeline.record_mapping_.timings_)} Hz")
            print(f"Average frequency of embodiment mapping: "
                  f"{1.0 / np.mean(pipeline.embodiment_mapping_.timings_)} Hz")
            pipeline.clear_timings()
            if fig.visualizer.poll_events():
                time.sleep(5)

    fig.show()


if __name__ == "__main__":
    main()
