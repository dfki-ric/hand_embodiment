"""Patches old MANO configurations with legacy transformation between MANO mesh and marker frame."""
import glob
import numpy as np
import pytransform3d.rotations as pr
import pytransform3d.transformations as pt
import pytransform3d.visualizer as pv
from hand_embodiment.mano import HandState
from hand_embodiment.config import load_mano_config, save_mano_config


def estimate_hand_pose_legacy(hand_top, hand_left, hand_right):
    """Estimate pose of the hand from markers on the back of the hand."""
    hand_middle = 0.5 * (hand_left + hand_right)
    hand_pose = np.eye(4)
    middle2top = hand_top - hand_middle
    middle2right = hand_right - hand_middle
    hand_pose[:3, :3] = pr.matrix_from_two_vectors(
        middle2top, middle2right).dot(
        pr.active_matrix_from_intrinsic_euler_xyz([0, 0.5 * np.pi, -0.5 * np.pi]))
    hand_pose[:3, 3] = hand_middle
    return hand_pose


def estimate_hand_pose(hand_top, hand_left, hand_right):
    """Estimate pose of the hand from markers on the back of the hand."""
    right2left = hand_left - hand_right
    approach = pr.norm_vector(hand_top - hand_right)
    orientation = pr.norm_vector(np.cross(approach, right2left))
    normal = np.cross(orientation, approach)
    hand_pose = np.eye(4)
    hand_pose[:3, :3] = np.column_stack((normal, orientation, approach))
    hand_pose[:3, 3] = hand_right
    return hand_pose


patch = False


hand_top = np.array([0.132351, 0.578986, 0.273875])
hand_left = np.array([0.120718, 0.62596, 0.252935])
hand_right = np.array([0.100843, 0.602364, 0.253115])
hand_markers_old2world = estimate_hand_pose_legacy(hand_top, hand_left, hand_right)
hand_markers_new2world = estimate_hand_pose(hand_top, hand_left, hand_right)
world2hand_markers_new = pt.invert_transform(hand_markers_new2world)
hand_markers_old2hand_markers_new = pt.concat(hand_markers_old2world, world2hand_markers_new)

filenames = glob.glob("examples/config/mano/*.yaml")
for filename in filenames:
    mano2hand_markers_old, betas = load_mano_config(filename)
    mano2hand_markers_new = pt.concat(mano2hand_markers_old, hand_markers_old2hand_markers_new)
    if patch:
        print(f"Patching '{filename}'")
        save_mano_config(filename, mano2hand_markers_new, betas)
    else:
        print(f"Not patching '{filename}'")
        print(f"mano2hand_markers_new=\n{repr(mano2hand_markers_new)}")

hand_state_old = HandState(left=False)
hand_state_old.shape_parameters = betas
hand_state_old.recompute_mesh(pt.concat(mano2hand_markers_old, hand_markers_old2world))
hand_mesh_old = hand_state_old.hand_mesh
hand_mesh_old.paint_uniform_color((1, 0, 0))

hand_state_new = HandState(left=False)
hand_state_new.shape_parameters = betas
hand_state_new.recompute_mesh(pt.concat(mano2hand_markers_new, hand_markers_new2world))
hand_mesh_new = hand_state_new.hand_pointcloud
hand_mesh_new.paint_uniform_color((0, 1, 0))


fig = pv.figure()
fig.scatter(np.row_stack((hand_top, hand_left, hand_right)), c=((1, 0, 0), (0, 1, 0), (0, 0, 1)), s=0.005)
fig.plot_transform(hand_markers_old2world, s=0.05)
fig.plot_transform(hand_markers_new2world, s=0.1)
fig.plot_transform(np.eye(4), s=0.3)
fig.add_geometry(hand_mesh_old)
fig.add_geometry(hand_mesh_new)
fig.view_init()
fig.show()
