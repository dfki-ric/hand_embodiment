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
        [0.000000e+00, 0.000000e+00, 0.000000e+00, 4.461087e-02,
         -1.256637e+00, 4.029302e-01, 4.745406e-05, -1.230391e-01,
         -1.560938e-01, 2.713341e-04, -3.663346e-02, 5.782625e-01])
    # For some reasons the result differs on different machines. We couldn't
    # find out why. At least the signs should match though!
    assert_array_almost_equal(np.sign(rm.hand_state_.pose), np.sign(pose))
    mano2hand_markers = rm.mano2hand_markers_
    assert_array_almost_equal(
        mano2hand_markers,
        np.array(
            [[-1.218693e-01, -8.650610e-02, -9.887692e-01,  6.167482e-02],
             [-8.418254e-19,  9.961947e-01, -8.715574e-02,  3.555096e-02],
             [ 9.925462e-01, -1.062161e-02, -1.214056e-01,  7.572715e-03],
             [ 0.000000e+00,  0.000000e+00,  0.000000e+00,  1.000000e+00]]))
