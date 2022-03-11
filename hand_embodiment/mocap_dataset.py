"""Motion capture dataset and preprocessing tools."""
import re
import warnings
import yaml
import numpy as np
import pandas as pd
from scipy.signal import medfilt


def read_qualisys_tsv(filename, unit="m", verbose=0):
    """Reads motion capturing data from tsv into pandas data frame.

    Parameters
    ----------
    filename : str
        Source file

    unit : str, optional (default: 'm')
        Unit to measure positions. Either meters 'm' or millimeters 'mm'.

    verbose : int, optional (default: 0)
        Verbosity level

    Returns
    -------
    df : DataFrame
        Raw data streams from source file
    """
    n_kv, n_meta = _header_sizes(filename)
    meta = pd.read_csv(
        filename, sep="\t", names=["Key", "Value"], header=None, nrows=7)
    meta = dict(zip(meta["Key"], meta["Value"]))
    if n_kv != n_meta:
        events = pd.read_csv(
            filename, sep="\t", header=None, skiprows=n_kv,
            names=["Event", "Type", "Frame", "Time"],
            nrows=n_meta - n_kv - 1)
    else:
        events = None

    df = pd.read_csv(
        filename, sep="\t", skiprows=n_meta, na_values=["null"])

    if unit == "m":
        # Get rid of "Time" and "Frame"
        marker_cols = df.columns[2:]
        df[marker_cols] /= 1000.0

    markers = [c[:-2] for c in df.columns if c.endswith(" X")]

    if verbose >= 1:
        print("[read_qualisys_tsv] Meta data:")
        print("  " + str(meta))
        print("[read_qualisys_tsv] Events:")
        print("  " + str(events))
        print("[read_qualisys_tsv] Available markers:")
        print("  " + (", ".join(markers)))
        print("[read_qualisys_tsv] Time delta: %g"
              % (1.0 / float(meta["FREQUENCY"])))

    return df


def _header_sizes(filename):
    """Determine number of lines in the header."""
    n_kv = 0    # Number of lines with metadata without events
    n_meta = 0  # Number of lines with metadata
    for i, l in enumerate(open(filename, "r")):
        if n_kv == 0 and l.startswith("EVENT"):
            n_kv = i
        elif l.startswith("Frame"):
            n_meta = i
            break
    if n_kv == 0:
        n_kv = n_meta
    return n_kv, n_meta


def array_from_dataframe(trajectory, columns):
    """Convert pandas DataFrame to numpy array.

    Parameters
    ----------
    trajectory : DataFrame
        Time series data

    columns : list of str
        Columns that should be extracted

    Returns
    -------
    array, shape (len(trajectories), len(columns))
        Extracted columns
    """
    return trajectory[columns].to_numpy()


def match_columns(trajectory, streams, keep_time=True):
    """Find columns of a dataframe that match regular expressions.

    Parameters
    ----------
    trajectory : DataFrame
        A collection of time series data

    streams : list of str
        Regular expressions that will be used to find matching streams in
        the columns of 'trajectory'. If None is given, we take all streams.

    keep_time : bool, optional (default: True)
        Keep the column with the name 'Time'

    Returns
    -------
    columns : list of str
        Columns that match given regular expressions (+ 'Time'). Columns are
        ordered first by given stream order and then by the dataframe's order
        of columns.
    """
    if streams is None:
        columns = list(trajectory.columns)
        columns.remove("Time")
    else:
        streams_re = [re.compile(s) for s in streams]
        columns = []
        for sre in streams_re:
            for c in trajectory.columns:
                if c not in columns and sre.match(c):
                    columns.append(c)

    if len(columns) == 0:
        raise ValueError(
            "No streams match the given patterns: %s.\n"
            "Available streams are: %s"
            % (", ".join(streams), ", ".join(trajectory.columns)))

    if keep_time and "Time" in trajectory:
        columns.append("Time")

    return columns


def extract_markers(trajectory, markers, keep_time=True):
    """Extract 3D marker streams (specific for Qualisys streams).

    Parameters
    ----------
    trajectory : DataFrame
        A collection of time series data

    markers : list of str
        Name of the Qualisys markers that will be used to find matching
        streams in the columns of 'trajectory'. We assume that each
        marker has three associated streams with ' X', ' Y', and ' Z' at
        the end of their names respectively.

    keep_time : bool, optional (default: True)
        Keep the column with the name 'Time'

    Returns
    -------
    trajectory : DataFrame
        A collection of time series data with only the given markers
    """
    columns = [m for m in trajectory.columns if m[:-2] in markers]
    if keep_time:
        columns = ["Time"] + columns
    return trajectory[columns]


def median_filter(X, window_size):
    """Median filter for trajectories.

    A median filter should be used to remove large jumps caused by noisy
    measurements or interpolation artifacts that often occur after
    normalization of orientation representations with ambiguities
    (such as quaternions).

    Parameters
    ----------
    X : array, shape (n_steps, n_dims) or DataFrame
        Trajectory

    Returns
    -------
    X : array, shape (n_steps, n_dims) or DataFrame
        Filtered trajectory
    """
    if isinstance(X, pd.DataFrame):
        return X.rolling(window_size).median()
    else:
        return np.column_stack(
            [medfilt(X[:, d], window_size) for d in range(X.shape[1])])


def interpolate_nan(X):
    """Remove NaNs with linear interpolation.

    This function accepts DataFrame objects and numpy arrays. When a NumPy
    array has to be converted, exact zeros are interpreted as NaNs, too.
    Furthermore an exception is thrown if the trajectory only contains NaNs.

    Parameters
    ----------
    X : array, shape (n_steps, n_dims) or DataFrame
        Trajectory

    Returns
    -------
    X : array, shape (n_steps, n_dims) or DataFrame
        Trajectory without NaN
    """
    if isinstance(X, pd.DataFrame):
        return X.interpolate(method="linear", limit_direction="both")
    else:
        nans = np.logical_or(np.isnan(X), X == 0.0)

        if np.all(nans):
            raise ValueError("Only NaN")

        for d in range(X.shape[1]):
            def x(y):
                return y.nonzero()[0]
            X[nans[:, d], d] = interp(x(nans[:, d]), x(~nans[:, d]),
                                      X[~nans[:, d], d])
        return X


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

    def _validate(self, trajectory, fail_on_error=False):
        markers = []
        for fn in self.config["finger_names"]:
            assert fn in self.config["finger_marker_names"]
            for mn in self.config["finger_marker_names"][fn]:
                markers.append(mn)

        markers += self.config["additional_markers"]
        for marker in markers:
            try:
                cols = match_columns(trajectory, [marker], keep_time=False)
            except ValueError:
                if fail_on_error:
                    raise Exception(f"Missing marker: '{marker}'.")
                else:
                    warnings.warn(f"Missing marker: '{marker}'.")

    def _scale(self, trajectory):
        data_columns = list(trajectory.columns)
        data_columns.remove("Time")
        new_trajectory = trajectory.copy()
        new_trajectory[data_columns] *= self.config["scale"]
        return new_trajectory

    def _hand_trajectories(self, hand_marker_names, trajectory):
        hand_trajectories = []
        assert len(hand_marker_names) == 3, hand_marker_names
        for marker_name in hand_marker_names:
            hand_column_names = match_columns(
                trajectory, [marker_name], keep_time=False)
            assert len(hand_column_names) == 3, hand_column_names
            hand_marker_trajectory = array_from_dataframe(
                trajectory, hand_column_names)
            hand_trajectories.append(hand_marker_trajectory)
        self.hand_trajectories = hand_trajectories

    def _finger_trajectories(self, finger_marker_names, finger_names, trajectory):
        finger_trajectories = {}
        for finger_name in finger_names:
            markers = finger_marker_names[finger_name]
            finger_column_names = match_columns(
                trajectory, markers, keep_time=False)
            assert len(finger_column_names) % 3 == 0, finger_column_names
            arr = array_from_dataframe(
                trajectory, finger_column_names)
            finger_trajectories[finger_name] = arr.reshape(
                -1, len(markers), 3)
        self.finger_trajectories = finger_trajectories

    def _additional_trajectories(self, additional_markers, trajectory):
        additional_trajectories = []
        for marker_name in additional_markers:
            try:
                column_names = match_columns(
                    trajectory, [marker_name], keep_time=False)
                assert len(column_names) == 3, column_names
            except ValueError:
                warnings.warn(
                    f"Could not find additional marker '{marker_name}'.")
                continue
            additional_trajectory = array_from_dataframe(
                trajectory, column_names)
            additional_trajectories.append(additional_trajectory)
        self.additional_trajectories = additional_trajectories

    def _convert_zeros_to_nans(self, hand_trajectory, marker_names):
        column_names = match_columns(
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

        trajectory = read_qualisys_tsv(filename=filename)
        trajectory = self._scale(trajectory)
        trajectory = extract_markers(trajectory, self.marker_names).copy()
        trajectory = self._convert_zeros_to_nans(trajectory, self.marker_names)
        trajectory = trajectory.iloc[slice(start_idx, end_idx)]
        trajectory = trajectory.iloc[::skip_frames]
        if interpolate_missing_markers:
            trajectory = interpolate_nan(trajectory)
            trajectory = median_filter(trajectory, 3)

        self.n_steps = len(trajectory)

        self._validate(trajectory)

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

    interpolate_missing_markers : bool, optional (default: False)
        Interpolate unknown marker positions (indicated by nan).
    """
    def __init__(self, filename, segment_label, mocap_config=None,
                 interpolate_missing_markers=False, **kwargs):
        super(SegmentedHandMotionCaptureDataset, self).__init__(mocap_config, **kwargs)
        self.interpolate_missing_markers = interpolate_missing_markers

        import mocap
        record = mocap.load(metadata=filename)
        streams = [f"{mn} .*" for mn in self.marker_names]
        try:
            self.segments = record.get_segments_as_dataframes(
                label=segment_label, streams=streams, label_field="l1",
                start_field="start_frame", end_field="end_frame")
        except KeyError:
            self.segments = record.get_segments_as_dataframes(
                label=segment_label, streams=streams, label_field="label 1",
                start_field="start index", end_field="end index")

        self.n_segments = len(self.segments)
        self.selected_segment = 0
        self.n_steps = 0

        self.select_segment(self.selected_segment)

    def select_segment(self, i):
        """Select a movement segment from the dataset.

        Parameters
        ----------
        i : int
            Index of the segment.
        """
        self.selected_segment = i

        trajectory = self.segments[self.selected_segment]
        if self.interpolate_missing_markers:
            trajectory = interpolate_nan(trajectory)
        trajectory = self._scale(trajectory)

        self.n_steps = len(trajectory)

        self._validate(trajectory)

        self._hand_trajectories(self.config["hand_marker_names"], trajectory)
        self._finger_trajectories(
            self.config["finger_marker_names"], self.finger_names, trajectory)
        self._additional_trajectories(
            self.config.get("additional_markers", ()), trajectory)
