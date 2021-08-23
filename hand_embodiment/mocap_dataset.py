import warnings
import yaml
import numpy as np
import mocap
from mocap import qualisys, pandas_utils, conversion
from mocap.cleaning import median_filter, interpolate_nan


class MotionCaptureDatasetBase:
    """Base class of motion capture datasets.

    Parameters
    ----------
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
    """
    def __init__(self, mocap_config, **kwargs):
        if mocap_config is not None:
            with open(mocap_config, "r") as f:
                self.config = yaml.safe_load(f)
        else:
            self.config = dict()
        if "scale" not in self.config:
            self.config["scale"] = 1.0
        self.config.update(kwargs)

        self.finger_names = self.config["finger_names"]
        all_finger_marker_names = []
        for fn in self.finger_names:
            if fn in self.config["finger_marker_names"]:
                all_finger_marker_names.extend(
                    self.config["finger_marker_names"][fn])
        self.marker_names = (
                self.config["hand_marker_names"] + all_finger_marker_names
                + self.config.get("additional_markers", []))

        self.hand_trajectories = []
        self.finger_trajectories = {}
        self.additional_trajectories = []

    def _hand_trajectories(self, hand_marker_names, trajectory):
        hand_trajectories = []
        assert len(hand_marker_names) == 3, hand_marker_names
        for marker_name in hand_marker_names:
            hand_column_names = pandas_utils.match_columns(
                trajectory, [marker_name], keep_time=False)
            assert len(hand_column_names) == 3, hand_column_names
            hand_marker_trajectory = conversion.array_from_dataframe(
                trajectory, hand_column_names)
            hand_trajectories.append(hand_marker_trajectory)
        self.hand_trajectories = hand_trajectories

    def _finger_trajectories(self, finger_marker_names, finger_names, trajectory):
        finger_trajectories = {}
        for finger_name in finger_names:
            markers = finger_marker_names[finger_name]
            finger_column_names = pandas_utils.match_columns(
                trajectory, markers, keep_time=False)
            assert len(finger_column_names) % 3 == 0, finger_column_names
            arr = conversion.array_from_dataframe(
                trajectory, finger_column_names)
            finger_trajectories[finger_name] = arr.reshape(
                -1, len(markers), 3)
        self.finger_trajectories = finger_trajectories

    def _additional_trajectories(self, additional_markers, trajectory):
        additional_trajectories = []
        for marker_name in additional_markers:
            try:
                column_names = pandas_utils.match_columns(
                    trajectory, [marker_name], keep_time=False)
                assert len(column_names) == 3, column_names
            except ValueError:
                warnings.warn(
                    f"Could not find additional marker '{marker_name}'.")
                continue
            additional_trajectory = conversion.array_from_dataframe(
                trajectory, column_names)
            additional_trajectories.append(additional_trajectory)
        self.additional_trajectories = additional_trajectories

    def _convert_zeros_to_nans(self, hand_trajectory, marker_names):
        column_names = pandas_utils.match_columns(
            hand_trajectory, marker_names, keep_time=False)
        for column_name in column_names:
            hand_trajectory[column_name].replace(0.0, np.nan, inplace=True)
        return hand_trajectory

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


class HandMotionCaptureDataset(MotionCaptureDatasetBase):
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
        super(HandMotionCaptureDataset, self).__init__(mocap_config, **kwargs)

        trajectory = qualisys.read_qualisys_tsv(filename=filename)
        trajectory *= self.config["scale"]
        trajectory = pandas_utils.extract_markers(
            trajectory, self.marker_names).copy()
        trajectory = self._convert_zeros_to_nans(trajectory, self.marker_names)
        trajectory = trajectory.iloc[slice(start_idx, end_idx)]
        trajectory = trajectory.iloc[::skip_frames]
        if interpolate_missing_markers:
            trajectory = interpolate_nan(trajectory)
            trajectory = median_filter(trajectory, 3).iloc[2:]

        self.n_steps = len(trajectory)

        self._hand_trajectories(self.config["hand_marker_names"], trajectory)
        self._finger_trajectories(
            self.config["finger_marker_names"], self.finger_names, trajectory)
        self._additional_trajectories(
            self.config.get("additional_markers", ()), trajectory)


class SegmentedHandMotionCaptureDataset(MotionCaptureDatasetBase):
    """Segmented hand motion capture dataset.

    Parameters
    ----------
    filename : str
        Name of the metadata file (.json).

    segment_label : str
        Label of the segments that will be extracted.

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
    """
    def __init__(self, filename, segment_label, mocap_config=None, **kwargs):
        super(SegmentedHandMotionCaptureDataset, self).__init__(mocap_config, **kwargs)

        record = mocap.load(metadata=filename)
        streams = [f"{mn} .*" for mn in self.marker_names]
        self.segments = record.get_segments_as_dataframes(
            label=segment_label, streams=streams, label_field="l1",
            start_field="start_frame", end_field="end_frame")

        self.n_segments = len(self.segments)
        self.selected_segment = 0
        self.n_steps = 0

        self.select_segment(self.selected_segment)

    def select_segment(self, i):
        self.selected_segment = i

        trajectory = self.segments[self.selected_segment]

        self.n_steps = len(trajectory)
        self._hand_trajectories(self.config["hand_marker_names"], trajectory)
        self._finger_trajectories(
            self.config["finger_marker_names"], self.finger_names, trajectory)
        self._additional_trajectories(
            self.config.get("additional_markers", ()), trajectory)
