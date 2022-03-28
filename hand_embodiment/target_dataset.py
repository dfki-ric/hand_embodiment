"""Dataset that contains a sequence of robotic hand states."""
import time
import tqdm
import copy
import numpy as np
import pandas as pd
from pytransform3d import transformations as pt


POSE_COLUMNS = ["base_x", "base_y", "base_z", "base_qw", "base_qx", "base_qy", "base_qz"]


class RoboticHandDataset:
    """Dataset that contains a trajectory of a robotic hand.

    Parameters
    ----------
    finger_names : list
        Names of fingers. Valid options: 'thumb', 'index', 'middle', 'ring',
        'little'.
    """
    def __init__(self, finger_names):
        self.finger_names = finger_names

        self.n_samples = 0
        self.ee_poses = []
        self.finger_joint_angles = []
        self.additional_finger_joint_angles = {}

    def append(self, ee_pose, finger_joint_angles):
        """Append sample to dataset.

        Parameters
        ----------
        ee_pose : array, shape (4, 4)
            Pose of the end effector.

        finger_joint_angles : dict
            Maps finger names to corresponding joint angles in the order that
            is given in the target configuration.
        """
        self.n_samples += 1
        self.ee_poses.append(ee_pose)
        self.finger_joint_angles.append(copy.deepcopy(finger_joint_angles))

    def add_constant_finger_joint(self, joint_name, angle):
        """Make finger joint constant.

        Parameters
        ----------
        joint_name : str
            Name of the robot's joint.

        angle : float
            Fixed angle of the joint.
        """
        self.additional_finger_joint_angles[joint_name] = angle

    def export(self, filename, hand_config):
        """Export dataset to csv file.

        Parameters
        ----------
        filename : str
            Name of the output file.

        hand_config : dict
            Configuration of the target hand. Must have a field 'joint_names'.
        """
        df = self.export_to_dataframe(hand_config)
        df.to_csv(filename)

    def export_to_dataframe(self, hand_config):
        """Export dataset to pandas dataframe.

        Parameters
        ----------
        hand_config : dict
            Configuration of the target hand. Must have a field 'joint_names'.

        Returns
        -------
        df : pd.DataFrame
            Dataframe.
        """
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
        return df

    @staticmethod
    def import_from_file(filename, hand_config):
        """Load dataset from file.

        Parameters
        ----------
        filename : str
            Name of the file that should be loaded.

        hand_config : dict
            Configuration of the target hand. Must have a field 'joint_names'.
        """
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
        """Number of steps."""
        return self.n_samples

    def get_ee_pose(self, t):
        """Get end-effector pose.

        Parameters
        ----------
        t : int
            Time step.

        Returns
        -------
        ee_pose : array, shape (4, 4)
            Pose of the end effector.
        """
        return self.ee_poses[t]

    def get_finger_joint_angles(self, t):
        """Get joint angles of the fingers.

        Returns
        -------
        finger_joint_angles : dict
            Joint angles per robotic finger.
        """
        return self.finger_joint_angles[t]


def convert_mocap_to_robot(dataset, pipeline, mocap_origin2origin=None, verbose=0):
    """Convert MoCap data to robot.

    Parameters
    ----------
    dataset : HandMotionCaptureDataset
        Motion capture dataset.

    pipeline : MoCapToRobot
        Converter from motion capture data to robot commands.

    mocap_origin2origin : array, shape (4, 4), optional (default: None)
        Frame conversion to transform end-effector poses to another coordinate
        system.

    verbose : int, optional (default: 0)
        Verbosity level.

    Returns
    -------
    output_dataset : RoboticHandDataset
        Converted motion.
    """
    output_dataset = RoboticHandDataset(finger_names=dataset.finger_names)
    pipeline.reset()

    start_time = time.time()
    for t in tqdm.tqdm(range(dataset.n_steps)):
        if mocap_origin2origin is not None and mocap_origin2origin.ndim == 3:
            mocap_origin2origin_t = mocap_origin2origin[t]
        else:
            mocap_origin2origin_t = mocap_origin2origin
        ee_pose, joint_angles = pipeline.estimate(
            dataset.get_hand_markers(t), dataset.get_finger_markers(t),
            mocap_origin2origin=mocap_origin2origin_t)
        output_dataset.append(ee_pose, joint_angles)

    if verbose:
        duration = time.time() - start_time
        time_per_frame = duration / dataset.n_steps
        frequency = dataset.n_steps / duration
        print(f"Embodiment mapping done after {duration:.2f} s, "
              f"{time_per_frame:.4f} s per frame, {frequency:.1f} Hz")
    return output_dataset
