import time


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
