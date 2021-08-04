import time
import tqdm
import copy
import numpy as np
import pandas as pd
from pytransform3d import transformations as pt


POSE_COLUMNS = ["base_x", "base_y", "base_z", "base_qw", "base_qx", "base_qy", "base_qz"]


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
        self.finger_joint_angles.append(copy.deepcopy(finger_joint_angles))

    def add_constant_finger_joint(self, joint_name, angle):
        self.additional_finger_joint_angles[joint_name] = angle

    def export(self, filename, hand_config):
        column_names = []
        for finger in self.finger_names:
            column_names += hand_config["joint_names"][finger]
        additional_joints = list(sorted(self.additional_finger_joint_angles.keys()))
        column_names += additional_joints
        column_names += POSE_COLUMNS

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

    @staticmethod
    def import_from_file(filename, hand_config):
        df = pd.read_csv(filename)
        ee_poses = [pt.transform_from_pq(df.iloc[t][POSE_COLUMNS])
                    for t in range(len(df))]

        df.drop(columns=POSE_COLUMNS, inplace=True)
        df = df[df.columns[1:]]  # drop index

        finger_to_joints = hand_config["joint_names"]
        finger_names = finger_to_joints.keys()

        finger_joint_angles = [dict() for _ in range(len(df))]
        for t in range(len(df)):
            row = df.iloc[t]
            for finger in finger_names:
                finger_joint_angles[t][finger] = []
                for joint in finger_to_joints[finger]:
                    finger_joint_angles[t][finger].append(row[joint])

        result = RoboticHandDataset(finger_names)
        result.ee_poses = ee_poses
        result.finger_joint_angles = finger_joint_angles
        result.n_samples = len(df)
        return result

    @property
    def n_steps(self):  # Compatibility to MotionCaptureDatasetBase
        return self.n_samples

    def get_ee_pose(self, t):
        return self.ee_poses[t]

    def get_finger_joint_angles(self, t):
        return self.finger_joint_angles[t]


def convert_mocap_to_robot(dataset, pipeline, ee2origin=None, verbose=0):
    output_dataset = RoboticHandDataset(finger_names=dataset.finger_names)
    pipeline.reset()

    start_time = time.time()
    for t in tqdm.tqdm(range(dataset.n_steps)):
        if ee2origin is not None and ee2origin.ndim == 3:
            ee2origin_t = ee2origin[t]
        else:
            ee2origin_t = ee2origin
        ee_pose, joint_angles = pipeline.estimate(
            dataset.get_hand_markers(t), dataset.get_finger_markers(t),
            ee2origin=ee2origin_t)
        output_dataset.append(ee_pose, joint_angles)

    if verbose:
        duration = time.time() - start_time
        time_per_frame = duration / dataset.n_steps
        frequency = dataset.n_steps / duration
        print(f"Embodiment mapping done after {duration:.2f} s, "
              f"{time_per_frame:.4f} s per frame, {frequency:.1f} Hz")
    return output_dataset
