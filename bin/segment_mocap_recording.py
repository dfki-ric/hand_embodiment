"""
Example call:

python bin/segment_mocap_recording.py data/20210701_april/Measurement31.tsv  --verbose
"""
import argparse
import numpy as np
from mocap import qualisys, pandas_utils, conversion, normalization
from mocap.cleaning import median_filter, interpolate_nan
from mocap.metadata import save_metadata
from segmentation_library.MCI.run import vmci
from vmci_segmentation.preprocessing import normalize_differences


def main():
    args = parse_args()
    trajectory, changepoints = segment(args)
    output_filename = args.filename.replace(".tsv", ".json")
    save_metadata(args.filename, args.markers[0], trajectory, changepoints,
                  output_filename)
    print(f"Saved segments to '{output_filename}'.")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "filename", type=str, help="Demonstration that should be segmented.")
    parser.add_argument(
        "--markers", type=list, default=["index_tip", "hand_top"],
        help="Markers that should be used.")
    parser.add_argument(
        "--frequency", type=float, default=25,
        help="Frequency at which the segmentation should be computed.")
    parser.add_argument(
        "--verbose", action="store_true", help="Print information.")
    return parser.parse_args()


def segment(args):
    assert args.filename.endswith(".tsv"), f"Not a tsv file: {args.filename}"
    trajectory = qualisys.read_qualisys_tsv(
        filename=args.filename, verbose=int(args.verbose))
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
    changepoints = [0] + [
        int((trajectory["Time"] - segment.t[-1]).abs().argsort()[0])
        for segment in segments]

    return trajectory, changepoints


if __name__ == "__main__":  # TODO make script part of segmentation library
    main()
