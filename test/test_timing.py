import time
from hand_embodiment.timing import TimeableMixin
from pytest import approx


class TimableMockup(TimeableMixin):
    def __init__(self):
        super(TimableMockup, self).__init__(True)

    def run(self):
        self.start_measurement()
        time.sleep(0.05)
        self.stop_measurement()


def test_timing():
    mockup = TimableMockup()
    mockup.run()
    assert approx(mockup.last_timing(), 0.05)


def test_multiple_timings_and_clear():
    mockup = TimableMockup()
    mockup.run()
    mockup.run()
    assert len(mockup.timings_) == 2
    mockup.clear_timings()
    assert len(mockup.timings_) == 0
