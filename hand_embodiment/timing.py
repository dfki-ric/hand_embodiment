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
    print(f"Mean: {np.round(np.mean(timings), decimals)}")
    print(f"Standard deviation: {np.round(np.std(timings), decimals)}")
    print(f"Median: {np.round(np.median(timings), decimals)}")
    print(f"Range: [{np.round(np.min(timings), decimals)}, "
          f"{np.round(np.max(timings), decimals)}]")
    print("=" * 80)
