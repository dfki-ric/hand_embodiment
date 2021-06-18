import numpy as np
import pandas as pd
from pytransform3d import transformations as pt


class RoboticHandDataset:
    """Dataset that contains a trajectory of a robotic hand."""
    def __init__(self, finger_names):
        self.finger_names = finger_names

        self.n_samples = 0
        self.ee_poses = []
        self.finger_joint_angles = []
        self.additional_finger_joint_angles = {}

    def append(self, ee_pose, finger_joint_angles):
        self.n_samples += 1
        self.ee_poses.append(ee_pose)
        self.finger_joint_angles.append(finger_joint_angles)

    def add_constant_finger_joint(self, joint_name, angle):
        self.additional_finger_joint_angles[joint_name] = angle

    def export(self, filename, hand_config):
        pose_columns = ["base_x", "base_y", "base_z", "base_qw", "base_qx", "base_qy", "base_qz"]
        column_names = []
        for finger in self.finger_names:
            column_names += hand_config["joint_names"][finger]
        additional_joints = list(sorted(self.additional_finger_joint_angles.keys()))
        column_names += additional_joints
        column_names += pose_columns

        raw_data = []
        for t in range(self.n_samples):
            joint_angles = []
            for finger in self.finger_names:
                joint_angles += self.finger_joint_angles[t][finger].tolist()
            for joint_name in additional_joints:
                joint_angles.append(self.additional_finger_joint_angles[joint_name])
            pose = pt.pq_from_transform(self.ee_poses[t])
            raw_data.append(np.hstack((joint_angles, pose)))

        df = pd.DataFrame(raw_data, columns=column_names)
        df.to_csv(filename)
