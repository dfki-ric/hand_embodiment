import numpy as np
from hand_embodiment.record_markers import MarkerBasedRecordMapping
from numpy.testing import assert_array_almost_equal


def test_smoke_record():
    rm = MarkerBasedRecordMapping()
    rm.reset()
    rm.estimate(
        hand_markers=[np.array([0, 0, 0]), np.array([0, 0, 1]),
                      np.array([0, 1, 0])],
        finger_markers={"index": [np.array([0, 0, 1])]})
    pose = np.zeros(48)
    pose[:12] = np.array(
        [0.000000e+00, 0.000000e+00, 0.000000e+00, 4.458424e-02,
         -1.256637e+00, 4.028908e-01, 5.826873e-06, -1.230357e-01,
         -1.560019e-01, 2.920878e-04, -3.664338e-02, 5.782096e-01])
    assert_array_almost_equal(rm.hand_state_.pose, pose)
    mano2hand_markers = rm.mano2hand_markers_
    assert_array_almost_equal(
        mano2hand_markers,
        np.array(
            [[-1.218693e-01, -8.650610e-02, -9.887692e-01,  6.167482e-02],
             [-8.418254e-19,  9.961947e-01, -8.715574e-02,  3.555096e-02],
             [ 9.925462e-01, -1.062161e-02, -1.214056e-01,  7.572715e-03],
             [ 0.000000e+00,  0.000000e+00,  0.000000e+00,  1.000000e+00]]))
