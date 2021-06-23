import warnings
import yaml
import numpy as np
from mocap import qualisys, pandas_utils, conversion
from mocap.cleaning import median_filter, interpolate_nan


class HandMotionCaptureDataset:
    """Hand motion capture dataset.

    Parameters
    ----------
    filename : str
        Name of the qualisys file (.tsv).

    finger_names : list of str, optional (default: None)
        Names of tracked fingers.

    hand_marker_names : list of str, optional (default: None)
        Names of hand markers that will be used to find the hand's pose.

    finger_marker_names : dict, optional (default: None)
        Mapping from finger names to corresponding marker.

    additional_markers : list, optional (default: [])
        Additional markers that have been tracked.

    mocap_config : str, optional (default: None)
        Path to configuration file that contains finger names, hand marker,
        names, finger marker names, and additional markers.

    skip_frames : int, optional (default: 1)
        Skip this number of frames when loading the motion capture data.

    start_idx : int, optional (default: None)
        Index of the first valid sample.

    end_idx : int, optional (default: None)
        Index of the last valid sample.

    interpolate_missing_markers : bool, optional (default: False)
        Interpolate unknown marker positions (indicated by nan).
    """
    def __init__(self, filename, mocap_config=None, skip_frames=1,
                 start_idx=None, end_idx=None,
                 interpolate_missing_markers=False, **kwargs):
        trajectory = qualisys.read_qualisys_tsv(filename=filename)

        if mocap_config is not None:
            with open(mocap_config, "r") as f:
                config = yaml.safe_load(f)
        else:
            config = dict()
        config.update(kwargs)

        self.finger_names = config["finger_names"]
        all_finger_marker_names = []
        for fn in self.finger_names:
            if fn in config["finger_marker_names"]:
                all_finger_marker_names.extend(
                    config["finger_marker_names"][fn])
        marker_names = (config["hand_marker_names"] + all_finger_marker_names
                        + config.get("additional_markers", []))
        trajectory = pandas_utils.extract_markers(
            trajectory, marker_names).copy()
        trajectory = self._convert_zeros_to_nans(trajectory, marker_names)
        trajectory = trajectory.iloc[slice(start_idx, end_idx)]
        trajectory = trajectory.iloc[::skip_frames]
        if interpolate_missing_markers:
            trajectory = interpolate_nan(trajectory)
            trajectory = median_filter(trajectory, 3).iloc[2:]

        self.n_steps = len(trajectory)

        self._hand_trajectories(config["hand_marker_names"], trajectory)
        self._finger_trajectories(
            config["finger_marker_names"], self.finger_names, trajectory)
        self._additional_trajectories(
            config.get("additional_markers", ()), trajectory)

    def _convert_zeros_to_nans(self, hand_trajectory, marker_names):
        column_names = pandas_utils.match_columns(
            hand_trajectory, marker_names, keep_time=False)
        for column_name in column_names:
            hand_trajectory[column_name].replace(0.0, np.nan, inplace=True)
        return hand_trajectory

    def _hand_trajectories(self, hand_marker_names, trajectory):
        self.hand_trajectories = []
        for marker_name in hand_marker_names:
            hand_column_names = pandas_utils.match_columns(
                trajectory, [marker_name], keep_time=False)
            hand_marker_trajectory = conversion.array_from_dataframe(
                trajectory, hand_column_names)
            self.hand_trajectories.append(hand_marker_trajectory)

    def _finger_trajectories(self, finger_marker_names, finger_names, trajectory):
        self.finger_trajectories = {}
        for finger_name in finger_names:
            markers = finger_marker_names[finger_name]
            finger_column_names = pandas_utils.match_columns(
                trajectory, markers, keep_time=False)
            arr = conversion.array_from_dataframe(
                trajectory, finger_column_names)
            self.finger_trajectories[finger_name] = arr.reshape(
                -1, len(markers), 3)

    def _additional_trajectories(self, additional_markers, trajectory):
        self.additional_trajectories = []
        for marker_name in additional_markers:
            try:
                column_names = pandas_utils.match_columns(
                    trajectory, [marker_name], keep_time=False)
            except ValueError:
                warnings.warn(
                    f"Could not find additional marker '{marker_name}'.")
                continue
            additional_trajectory = conversion.array_from_dataframe(
                trajectory, column_names)
            self.additional_trajectories.append(additional_trajectory)

    def get_hand_markers(self, t):
        """Get hand markers to extract pose of the hand."""
        return [ht[t] for ht in self.hand_trajectories]

    def get_finger_markers(self, t):
        """Get finger markers."""
        return {fn: self.finger_trajectories[fn][t]
                for fn in self.finger_names}

    def get_additional_markers(self, t):
        """Get additional markers."""
        return [at[t] for at in self.additional_trajectories]

    def get_markers(self, t):
        """Get positions of all markers."""
        hand_markers = self.get_hand_markers(t)
        finger_markers = self.get_finger_markers(t)
        additional_trajectories = self.get_additional_markers(t)
        finger_markers_flat = []
        for fn in self.finger_names:
            finger_markers_flat.extend(finger_markers[fn].tolist())
        return np.array(hand_markers + finger_markers_flat
                        + additional_trajectories)
