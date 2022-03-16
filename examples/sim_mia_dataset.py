"""Simulate embodied trajectory with Mia hand in PyBullet."""
import time
import numpy as np
import pybullet as pb
import pytransform3d.rotations as pr
import pytransform3d.transformations as pt
from hand_embodiment.target_dataset import RoboticHandDataset
from hand_embodiment.target_configurations import TARGET_CONFIG


dt = 0.01
filename = "test/data/mia_segment.csv"

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

index_id = 1
little_id = 3
mrl_id = 4
ring_id = 6
thumb_opp_id = 7
thumb_fle_id = 9
hand_joint_indices = [index_id, mrl_id, ring_id, little_id, thumb_fle_id, thumb_opp_id]
inertial2link_pos, inertial2link_orn = pb.getDynamicsInfo(hand, -1)[3:5]

dataset = RoboticHandDataset.import_from_file(filename, hand_config)
t = 0
while pb.isConnected():
    finger_joint_angles = dataset.get_finger_joint_angles(t % dataset.n_steps)
    target_positions = [
        finger_joint_angles["index"][0], finger_joint_angles["middle"][0],
        finger_joint_angles["middle"][0], finger_joint_angles["middle"][0],
        finger_joint_angles["thumb"][0], 0.0]
    pb.setJointMotorControlArray(
        hand, hand_joint_indices,
        pb.POSITION_CONTROL,
        targetPositions=target_positions,
        targetVelocities=np.zeros(len(target_positions)))

    ee_pose = pt.pq_from_transform(dataset.get_ee_pose(t % dataset.n_steps))
    link2world_pos = ee_pose[:3]
    link2world_orn = pr.quaternion_xyzw_from_wxyz(ee_pose[3:])
    inertial2world_pos, inertial2world_orn = pb.multiplyTransforms(
        link2world_pos, link2world_orn, inertial2link_pos, inertial2link_orn)
    pb.resetBasePositionAndOrientation(
        hand, inertial2world_pos, inertial2world_orn)

    pb.stepSimulation()

    time.sleep(dt)
    t += 1
