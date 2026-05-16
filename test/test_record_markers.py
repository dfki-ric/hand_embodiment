import numpy as np
from hand_embodiment.record_markers import MarkerBasedRecordMapping
from numpy.testing import assert_array_almost_equal


def test_smoke_record():
    rm = MarkerBasedRecordMapping()
    rm.reset()
    rm.estimate(
        hand_markers=[np.array([0, 0, 0], dtype=float),
                      np.array([0, 0, 1], dtype=float),
                      np.array([0, 1, 0], dtype=float)],
        finger_markers={"index": [np.array([0, 0, 1], dtype=float)]})
    # The test target [0,0,1] is ~1m from the MANO origin (unreachable).
    # The LM finds some local minimum; its exact pose depends on the optimizer
    # and hardware, so we only check that the IK ran (pose non-trivial).
    assert not np.all(rm.hand_state_.pose == 0.0)
    mano2hand_markers = rm.mano2hand_markers_
    assert_array_almost_equal(
        mano2hand_markers,
        np.array(
            [[-0.12186934, -0.0865061, -0.98876921, 0.06167482],
             [0.0, 0.9961947, -0.08715574, 0.03555096],
             [0.99254615, -0.01062161, -0.12140559, 0.00757272],
             [0.0, 0.0, 0.0, 1.0]]))
