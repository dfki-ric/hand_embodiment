"""Measure time for benchmarking."""
import time
import numpy as np


class TimeableMixin:
    """A timeable object stores measurements of duration times.

    Parameters
    ----------
    measure : bool
        Measure timing. Otherwise nothing will be measured.

    Attributes
    ----------
    timings_ : list of float
        All previous time measurements.
    """
    def __init__(self, measure):
        self._measure = measure
        self._start_time = None
        self._duration = None
        self.timings_ = []

    def start_measurement(self):
        """Start time measurement."""
        if self._measure:
            self._start_time = time.time()

    def stop_measurement(self):
        """Stop time measurement and store result."""
        if self._measure:
            end_time = time.time()
            self._duration = end_time - self._start_time
            self.timings_.append(self._duration)
            self._start_time = None

    def last_timing(self):
        """Get last time measurement."""
        return self._duration

    def clear_timings(self):
        """Clear time measurements."""
        self.timings_ = []


def timing_report(timeable, decimals=5, title=None):
    """Print timing report.

    Parameters
    ----------
    timeable : TimeableMixin
        Timeable object.

    decimals : int, optional (default: 5)
        Number of decimal places to print.

    title : str, optional (default: None)
        Title of the report.
    """
    timings = np.asarray(timeable.timings_)
    print("=" * 80)
    if title is not None:
        print(f"Timing report: {title}")
    n_measurements = len(timings)
    print(f"Number of measurements: {n_measurements}")
    average = np.mean(timings)
    print(f"Mean: {np.round(average, decimals)} s, {np.round(1.0 / average, decimals)} Hz")
    std = np.std(timings)
    print(f"Standard deviation: {np.round(std, decimals)} s, {np.round(1.0 / std, decimals)} Hz")
    median = np.median(timings)
    print(f"Median: {np.round(median, decimals)} s, {np.round(1.0 / median, decimals)} Hz")
    min = np.min(timings)
    max = np.max(timings)
    print(f"Range: [{np.round(min, decimals)} s, {np.round(max, decimals)} s], "
          f"[{np.round(1.0 / max, decimals)} Hz, {np.round(1.0 / min, decimals)} Hz]")
    print("=" * 80)
