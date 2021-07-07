import argparse
import getpass
import json
import numpy as np
from mocap import qualisys, pandas_utils, conversion, normalization
from mocap.cleaning import median_filter, interpolate_nan
from segmentation_library.MCI.run import vmci
from vmci_segmentation.preprocessing import normalize_differences


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--filename", type=str,
        default="data/QualisysAprilTest/april_test_010.tsv",
        help="Demonstration that should be used.")
    parser.add_argument(
        "--markers", type=list, default=["index_tip", "hand_top"],
        help="Markers that should be used.")
    parser.add_argument(
        "--frequency", type=float, default=40,
        help="Frequency at which the segmentation should be computed.")
    return parser.parse_args()


def segment(args):
    trajectory = qualisys.read_qualisys_tsv(filename=args.filename)
    trajectory = pandas_utils.extract_markers(trajectory, args.markers).copy()
    trajectory = interpolate_nan(trajectory)
    #trajectory = median_filter(trajectory, 3).iloc[2:]
    downsampled_trajectory = normalization.to_frequency(trajectory, args.frequency)

    downsampled_trajectory = downsampled_trajectory

    time = conversion.array_from_dataframe(downsampled_trajectory, ["Time"])
    columns = pandas_utils.match_columns(
        downsampled_trajectory, args.markers, keep_time=False)
    positions = conversion.array_from_dataframe(downsampled_trajectory, columns)
    velocities = np.empty((len(time), len(args.markers)))
    velocities[0] = 0.0
    for t in range(1, len(velocities)):
        for m in range(len(args.markers)):
            dp = (positions[t, m * 3:(m + 1) * 3]
                  - positions[t - 1, m * 3:(m + 1) * 3])
            dt = time[t] - time[t - 1]
            velocities[t, m] = np.linalg.norm(dp) / dt

    positions, velocities = normalize_differences(positions, velocities)

    _, segments = vmci(
        positions, velocities, time, save_dir=None,
        name_run="vMCI_results_", plot_results=False,
        demo_name=args.filename, set_mean=True, seed=0, verbose=10)

    # map back to indices of original time series
    changepoints = [
        int((trajectory["Time"] - segment.t[-1]).abs().argsort()[0])
        for segment in segments[:-1]]

    return trajectory, changepoints


# TODO move to mocap
LABELED_BY = "labeled_by"
SUBJECT = "subject"
FREQUENCY = "frequency"
RECORD_FILENAME = "record_filename"
PLATFORM_TYPE = "platform_type"
FILE_COMMENT = "file_comment"
SEGMENTS = "segments"

START_FRAME = "start_frame"
END_FRAME = "end_frame"
MARKER = "marker"
NONE = "None"
START = "start"
END = "end"
L1 = "l1"
L2 = "l2"
LENGTH = "length"
LABEL = "label"
SECOND_LABEL = "second_label"


def save_metadata(filename, markers, trajectory, changepoints, output_filename):
    out = dict()
    out[LABELED_BY] = getpass.getuser()  # current username
    out[SUBJECT] = None
    out[FREQUENCY] = None
    out[RECORD_FILENAME] = filename
    out[PLATFORM_TYPE] = ".{}".format(filename.split('.')[1])
    out[FILE_COMMENT] = None
    out[SEGMENTS] = []

    start_idx = 0
    for changepoint in changepoints + [len(trajectory) - 1]:
        # create new segment and add it
        start_time = float(trajectory["Time"].iloc[start_idx])
        end_time = float(trajectory["Time"].iloc[changepoint])
        segment = {
            START_FRAME: start_idx,
            START: start_time,
            END_FRAME: changepoint,
            END: end_time,
            L1: "unknown",
            L2: "",
            LENGTH: int(changepoint) - int(start_idx),
            MARKER: markers[0]
        }
        out[SEGMENTS].append(segment)
        start_idx = changepoint

    with open(output_filename, "w") as f:
        json.dump(out, f, indent=4)


def main():
    args = parse_args()
    trajectory, changepoints = segment(args)
    output_filename = args.filename.replace(".tsv", ".json")
    save_metadata(args.filename, args.markers, trajectory, changepoints,
                  output_filename)
    print(f"Saved segments to '{output_filename}'.")


if __name__ == "__main__":  # TODO make script part of segmentation library
    main()
