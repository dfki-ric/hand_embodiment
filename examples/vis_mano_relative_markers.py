"""Visualize relative marker positions and estimated hand pose from it.

Example call:

    python examples/vis_mano_relative_markers.py sample3.txt --mocap-config examples/config/markers/20220328_april.yaml --mano-config examples/config/mano/20210701_april.yaml
"""
import argparse

import numpy as np
import pytransform3d.visualizer as pv

from hand_embodiment.record_markers import MarkerBasedRecordMapping
from hand_embodiment.config import load_mano_config, load_record_mapping_config
from hand_embodiment.command_line import add_configuration_arguments


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "filename", type=str,
        help="File that contains MANO-relative marker positions.")
    add_configuration_arguments(parser)
    return parser.parse_args()


def main():
    args = parse_args()

    finger_names = ["thumb", "index", "middle", "ring", "little"]

    mano2hand_markers, betas = load_mano_config(args.mano_config)
    record_mapping_config = args.record_mapping_config
    if record_mapping_config is not None:
        record_mapping_config = load_record_mapping_config(
            args.record_mapping_config)
    mbrm = MarkerBasedRecordMapping(
        left=False, mano2hand_markers=mano2hand_markers,
        shape_parameters=betas, record_mapping_config=record_mapping_config,
        use_fingers=finger_names, verbose=0, measure_time=False)

    sample = np.loadtxt(args.filename)
    finger_markers = {}
    idx = 0
    for finger_name in finger_names:
        finger_markers[finger_name] = sample[idx:idx + 6].reshape(2, 3)
        idx += 6
    mbrm.estimate_fingers(finger_names, finger_markers)
    mbrm.hand_state_.recompute_mesh(mesh2world=np.eye(4))

    fig = pv.figure()
    fig.plot_transform(np.eye(4), s=0.05)
    fig.scatter(sample.reshape(-1, 3), s=0.006)
    fig.add_geometry(mbrm.hand_state_.hand_mesh)
    fig.view_init()
    fig.show()


if __name__ == "__main__":
    main()
