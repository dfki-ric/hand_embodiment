import numpy as np
import pytransform3d.visualizer as pv
import pytransform3d.transformations as pt
from hand_embodiment.mano import HandState
from hand_embodiment.embodiment import HandEmbodiment
from hand_embodiment.target_configurations import MIA_CONFIG, manobase2miabase


default_mano_pose = np.array([
    0, 0, 0,
    -0.068, 0, 0.068,
    0, 0.068, 0.068,
    0, 0, 0.615,
    0, 0.137, 0.068,
    0, 0, 0.137,
    0, 0, 0.683,
    0, 0.205, -0.137,
    0, 0.068, 0.205,
    0, 0, 0.205,
    0, 0.137, -0.137,
    0, -0.068, 0.273,
    0, 0, 0.478,
    0.615, 0.068, 0.273,
    0, 0, 0,
    0, 0, 0
])

mano_pose = np.copy(default_mano_pose)
mano_pose[5] = 1.0
mano_pose[14] = 1.0
mano_pose[40] = -0.5

fig = pv.figure()

hand_state = HandState(left=False)
hand_state.pose[:] = mano_pose
hand_state.recompute_mesh(manobase2miabase)
fig.add_geometry(hand_state.hand_mesh)

emb = HandEmbodiment(hand_state, MIA_CONFIG)

import time
start = time.time()

joint_angles, desired_positions = emb.solve(
    return_desired_positions=True,
    use_cached_forward_kinematics=False)

print(time.time() - start)

for finger_name in emb.finger_names_:
    actual_positions = emb.finger_forward_kinematics(
        finger_name, joint_angles[finger_name])
    for p in desired_positions[finger_name]:
        p = pt.translate_transform(np.eye(4), p)
        fig.plot_sphere(0.006, p, c=(0, 1, 0))
    for p in actual_positions:
        fig.plot_sphere(0.006, p, c=(1, 0, 0))

graph = pv.Graph(
    emb.target_kin.tm, MIA_CONFIG["base_frame"], show_frames=False,
    show_connections=False, show_visuals=True, show_collision_objects=False,
    show_name=False, s=0.02)
graph.add_artist(fig)

fig.show()
