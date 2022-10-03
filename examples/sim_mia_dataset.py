"""Simulate embodied trajectory with Mia hand in PyBullet."""
import argparse
import time
import numpy as np
import pybullet as pb
import pytransform3d.rotations as pr
import pytransform3d.transformations as pt
from hand_embodiment.target_dataset import RoboticHandDataset
from hand_embodiment.target_configurations import TARGET_CONFIG
from hand_embodiment.command_line import add_object_visualization_arguments
from hand_embodiment.vis_utils import ARTISTS


index_id = 1
little_id = 3
mrl_id = 4
ring_id = 6
thumb_opp_id = 7
thumb_fle_id = 9


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "trajectory_files", nargs="*", type=str,
        default=["test/data/mia_segment.csv"],
        help="Trajectories that should be used.")
    add_object_visualization_arguments(parser)
    return parser.parse_args()


def main():
    args = parse_args()

    dt = 0.01

    hand_config = TARGET_CONFIG["mia"]

    pb.connect(pb.GUI)
    pb.configureDebugVisualizer(pb.COV_ENABLE_GUI, 0)
    pb.configureDebugVisualizer(pb.COV_ENABLE_SHADOWS, 0)
    pb.configureDebugVisualizer(pb.COV_ENABLE_MOUSE_PICKING, 0)
    pb.resetDebugVisualizerCamera(0.5, 90, 20, [0, 0, 0])
    pb.resetSimulation(pb.RESET_USE_DEFORMABLE_WORLD)
    pb.setTimeStep(dt)
    pb.setGravity(0, 0, 0)

    hand = pb.loadURDF(
        "hand_embodiment/model/mia_hand_ros_pkgs/mia_hand_description/urdf/"
        "mia_hand.urdf.xacro",
        [0, 0, 0], [0, 0, 0, 1], useFixedBase=1,
        flags=pb.URDF_USE_SELF_COLLISION | pb.URDF_USE_SELF_COLLISION_EXCLUDE_PARENT)

    for artist in args.visual_objects:
        visual = ARTISTS[artist]()
        visual_uid = pb.createVisualShape(
            pb.GEOM_MESH, fileName=visual.mesh_filename, meshScale=1.0,
            rgbaColor=np.r_[visual.mesh_color, 1.0])
        assert visual_uid != -1
        collision_uid = pb.createCollisionShape(
            pb.GEOM_BOX, halfExtents=[0.001] * 3)
        mesh2markers = pt.invert_transform(visual.markers2mesh)
        pos = mesh2markers[:3, 3]
        orn = pr.quaternion_xyzw_from_wxyz(pr.quaternion_from_matrix(mesh2markers[:3, :3]))
        pb.createMultiBody(1e-5, collision_uid, visual_uid, basePosition=pos, baseOrientation=orn)

    hand_joint_indices = [index_id, mrl_id, ring_id, little_id, thumb_fle_id, thumb_opp_id]
    inertial2link_pos, inertial2link_orn = pb.getDynamicsInfo(hand, -1)[3:5]

    while pb.isConnected():
        for filename in args.trajectory_files:
            dataset = RoboticHandDataset.import_from_file(filename, hand_config)
            t = 0
            while t < dataset.n_steps:
                finger_joint_angles = dataset.get_finger_joint_angles(t)
                target_positions = [
                    finger_joint_angles["index"][0], finger_joint_angles["middle"][0],
                    finger_joint_angles["middle"][0], finger_joint_angles["middle"][0],
                    finger_joint_angles["thumb"][0],
                    dataset.additional_finger_joint_angles["j_thumb_opp"]]
                pb.setJointMotorControlArray(
                    hand, hand_joint_indices,
                    pb.POSITION_CONTROL,
                    targetPositions=target_positions,
                    targetVelocities=np.zeros(len(target_positions)))

                ee_pose = pt.pq_from_transform(dataset.get_ee_pose(t))
                link2world_pos = ee_pose[:3]
                link2world_orn = pr.quaternion_xyzw_from_wxyz(ee_pose[3:])
                inertial2world_pos, inertial2world_orn = pb.multiplyTransforms(
                    link2world_pos, link2world_orn, inertial2link_pos, inertial2link_orn)
                pb.resetBasePositionAndOrientation(
                    hand, inertial2world_pos, inertial2world_orn)

                pb.stepSimulation()

                time.sleep(dt)
                t += 1


if __name__ == "__main__":
    main()
