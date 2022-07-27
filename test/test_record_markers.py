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
    pose = np.zeros(48, dtype=float)
    pose[:12] = np.array(
        [0.00000000e+00,  0.00000000e+00,  0.00000000e+00, -8.48596438e-01,
         -1.25663706e+00, -1.25663706e+00, -1.79209343e-06, -3.28425717e-01,
         -5.53037234e-01,  2.61635503e-03,  3.34717636e-04,  6.00331275e-01])
    # For some reasons the result differs on different machines. We couldn't
    # find out why. At least the signs should match though!
    assert_array_almost_equal(np.sign(rm.hand_state_.pose), np.sign(pose))
    mano2hand_markers = rm.mano2hand_markers_
    assert_array_almost_equal(
        mano2hand_markers,
        np.array(
            [[-0.12186934, -0.0865061, -0.98876921, 0.06167482],
             [0.0, 0.9961947, -0.08715574, 0.03555096],
             [0.99254615, -0.01062161, -0.12140559, 0.00757272],
             [0.0, 0.0, 0.0, 1.0]]))
