import numpy as np
from mocap import qualisys, pandas_utils, conversion
from mocap.cleaning import median_filter, interpolate_nan


class HandMotionCaptureDataset:
    def __init__(self, filename, finger_names, hand_marker_names,
                 finger_marker_names, additional_markers=(), skip_frames=1):
        trajectory = qualisys.read_qualisys_tsv(filename=filename)

        marker_names = hand_marker_names + list(finger_marker_names.values()) + additional_markers
        trajectory = pandas_utils.extract_markers(
            trajectory, marker_names).copy()
        trajectory = self._convert_zeros_to_nans(
            trajectory, marker_names)
        trajectory = trajectory.iloc[::skip_frames]
        trajectory = median_filter(
            interpolate_nan(trajectory), 3).iloc[2:]

        self.n_steps = len(trajectory)

        self.hand_trajectories = []
        for marker_name in hand_marker_names:
            hand_column_names = pandas_utils.match_columns(
                trajectory, [marker_name], keep_time=False)
            hand_marker_trajectory = conversion.array_from_dataframe(
                trajectory, hand_column_names)
            self.hand_trajectories.append(hand_marker_trajectory)
        self.finger_trajectories = {}
        for finger_name in finger_names:
            finger_marker_name = finger_marker_names[finger_name]
            finger_column_names = pandas_utils.match_columns(
                trajectory, [finger_marker_name], keep_time=False)
            self.finger_trajectories[finger_name] = \
                conversion.array_from_dataframe(
                    trajectory, finger_column_names)
        self.additional_trajectories = []
        for marker_name in additional_markers:
            column_names = pandas_utils.match_columns(
                trajectory, [marker_name], keep_time=False)
            additional_trajectory = conversion.array_from_dataframe(
                trajectory, column_names)
            self.additional_trajectories.append(additional_trajectory)

    def _convert_zeros_to_nans(self, hand_trajectory, marker_names):
        column_names = pandas_utils.match_columns(
            hand_trajectory, marker_names, keep_time=False)
        for column_name in column_names:
            hand_trajectory[column_name].replace(0.0, np.nan, inplace=True)
        return hand_trajectory

    def get_hand_markers(self, t):
        return [ht[t] for ht in self.hand_trajectories]

    def get_finger_markers(self, t):
        return {fn: self.finger_trajectories[fn][t]
                for fn in self.finger_trajectories}

    def get_additional_markers(self, t):
        return [at[t] for at in self.additional_trajectories]

    def get_markers(self, t):
        hand_markers = self.get_hand_markers(t)
        finger_markers = self.get_finger_markers(t)
        additional_trajectories = self.get_additional_markers(t)
        return np.array(hand_markers + additional_trajectories + list(finger_markers.values()))