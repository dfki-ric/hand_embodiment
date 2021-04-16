import argparse
import numpy as np
import open3d as o3d
import pytransform3d.rotations as pr
import pytransform3d.transformations as pt
from pytransform3d import visualizer as pv
from mocap.mano import HandState

from hand_embodiment.vis_utils import make_coordinate_system

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
        "--zero-pose", action="store_true",
        help="Set all pose parameters to 0.")

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

    if args.zero_pose:
        pose = np.zeros_like(POSE)
    else:
        pose = POSE
    hand_state.pose[:] = pose
    hand_state.mesh_updated = True

    pose = pose.reshape(-1, 3)
    J = joint_poses(pose, hand_state.pose_parameters["J"],
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
    fig.show()


if __name__ == "__main__":
    main()
