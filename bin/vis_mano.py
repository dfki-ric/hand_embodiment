"""
Example call:
python bin/vis_mano.py \
    --config-filename examples/config/april_test_mano2.yaml \
    --show-tips --show-mesh --show-transforms
"""
import argparse
import numpy as np
import open3d as o3d
import pytransform3d.rotations as pr
import pytransform3d.transformations as pt
from pytransform3d import visualizer as pv
from mocap.mano import HandState

from hand_embodiment.vis_utils import make_coordinate_system
from hand_embodiment.config import load_mano_config
from hand_embodiment.record_markers import MANO_CONFIG, make_finger_kinematics

POSE = np.array([
    0, 0, 0,
    -0.068, 0, 0.068 + 1,
    0, 0.068, 0.068,
    0, 0, 0.615,
    0, 0.137, 0.068,
    0, 0, 0.137,
    0, 0, 0.683,
    0, 0.205, -0.137,
    0, 0.068, 0.205,
    0, 0, 0.205,
    0, 0.137, -0.137,
    0, -0.068, 0.273,
    0, 0, 0.478,
    0.615, 0.068, 0.273,
    0, 0, 0,
    0, 0, 0
])


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--vertices", type=int, nargs="*", default=[],
        help="Highlight vertices in red color.")
    parser.add_argument(
        "--joints", type=int, nargs="*", default=[],
        help="Highlight joints (they will be bigger).")
    parser.add_argument(
        "--show-mesh", action="store_true", help="Show mesh.")
    parser.add_argument(
        "--show-reference", action="store_true",
        help="Show coordinate frame for size reference.")
    parser.add_argument(
        "--show-transforms", action="store_true",
        help="Show reference frames of markers and MANO.")
    parser.add_argument(
        "--show-tips", action="store_true",
        help="Show tip vertices of fingers in green color.")
    parser.add_argument(
        "--color-fingers", action="store_true",
        help="Show finger vertices in uniform color.")
    parser.add_argument(
        "--zero-pose", action="store_true",
        help="Set all pose parameters to 0.")
    parser.add_argument(
        "--config-filename", type=str, default=None,
        help="MANO configuration that includes shape parameters.")

    return parser.parse_args()


def joint_poses(pose, J, kintree_table):
    """Computes global rotation and translation of the model.

    Parameters
    ----------
    pose : array, shape (n_parts * 3)
        Hand pose parameters

    J : array, shape (n_parts, 3)
        Joint positions

    kintree_table : array, shape (2, n_parts)
        Table that describes the kinematic tree of the hand.
        kintree_table[0, i] contains the index of the parent part of part i
        and kintree_table[1, :] does not matter for the MANO model.

    Returns
    -------
    J : list
        Poses of the joints as transformation matrices
    """
    id_to_col = {kintree_table[1, i]: i
                 for i in range(kintree_table.shape[1])}
    parent = {i: id_to_col[kintree_table[0, i]]
              for i in range(1, kintree_table.shape[1])}

    results = {0: pt.transform_from(
        pr.matrix_from_compact_axis_angle(pose[0, :]), J[0, :])}
    for i in range(1, kintree_table.shape[1]):
        T = pt.transform_from(pr.matrix_from_compact_axis_angle(
            pose[i, :]), J[i, :] - J[parent[i], :])
        results[i] = results[parent[i]].dot(T)

    results = [results[i] for i in sorted(results.keys())]
    return results


def main():
    args = parse_args()

    hand_state = HandState(left=False)

    if args.config_filename is None:
        mano2hand_markers, betas = np.eye(4), np.zeros(
            hand_state.n_shape_parameters)
    else:
        mano2hand_markers, betas = load_mano_config(args.config_filename)

    if args.zero_pose:
        pose = np.zeros_like(POSE)
    else:
        pose = POSE

    hand_state.betas[:] = betas
    hand_state.recompute_shape()
    hand_state.pose[:] = pose
    hand_state.recompute_mesh()

    J = joint_poses(pose.reshape(-1, 3), hand_state.pose_parameters["J"],
                    hand_state.pose_parameters["kintree_table"])

    pc = hand_state.hand_pointcloud

    if args.vertices:
        colors = []
        for i in range(len(pc.points)):
            if i in args.vertices:
                colors.append((1, 0, 0))
            else:
                colors.append((0, 0, 0))
        pc.colors = o3d.utility.Vector3dVector(colors)

    if args.color_fingers:
        colors = [
            (1, 0, 0),
            (1, 1, 0),
            (0, 1, 1),
            (0, 0, 1),
            (1, 0, 1),
        ]
        for finger, c in zip(MANO_CONFIG["vertex_indices_per_finger"], colors):
            kin = make_finger_kinematics(hand_state, finger)
            for index in kin.all_finger_vertex_indices:
                pc.colors[index] = c

    if args.show_tips:
        vipf = MANO_CONFIG["vertex_indices_per_finger"]
        for finger in vipf:
            indices = vipf[finger]
            kin = make_finger_kinematics(hand_state, finger)
            positions = kin.forward(pose[kin.finger_pose_param_indices])
            for i, index in enumerate(indices):
                pc.colors[index] = (0, 1, 0)
                for dist in range(1, 6):
                    if index - dist >= 0:
                        pc.colors[index - dist] = (0, 0, 1.0 / dist)
                    if index + dist < len(pc.colors):
                        pc.colors[index + dist] = (0, 0, 1.0 / dist)
                pc.points[index] = positions[i]

    fig = pv.figure()
    fig.add_geometry(pc)
    for i in range(len(J)):
        if i in args.joints:
            s = 0.05
        else:
            s = 0.01
        fig.plot_transform(J[i], s=s)
    if args.show_mesh:
        fig.add_geometry(hand_state.hand_mesh)
    if args.show_reference:
        coordinate_system = make_coordinate_system(s=0.2)
        fig.add_geometry(coordinate_system)
    if args.show_transforms:
        fig.plot_transform(np.eye(4), s=0.05)
        fig.plot_transform(pt.invert_transform(mano2hand_markers), s=0.05)
    fig.show()


if __name__ == "__main__":
    main()
