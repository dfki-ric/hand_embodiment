import numpy as np
from hand_embodiment.mano import HandState


def test_mano():
    mano = HandState(left=False)

    random_state = np.random.RandomState(0)
    shape = random_state.randn(mano.n_shape_parameters)
    for i in range(mano.n_shape_parameters):
        mano.set_shape_parameter(i, shape[i])
    pose = random_state.randn(mano.n_pose_parameters)
    for i in range(mano.n_pose_parameters):
        mano.set_pose_parameter(i, pose[i])

    mano.recompute_mesh(np.eye(4))

    mesh = mano.hand_mesh
    assert len(mesh.vertices) == 778

    pc = mano.hand_pointcloud
    assert len(pc.points) == 778
