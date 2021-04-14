import numpy as np
import pytransform3d.visualizer as pv
from mocap.mano import HandState
from hand_embodiment.embodiment import HandEmbodiment
from hand_embodiment.load_model import load_kinematic_model
from hand_embodiment.target_configurations import MIA_CONFIG, manobase2miabase, kinematic_model_hook_mia


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

fig = pv.figure()

hand_state = HandState(left=False)
hand_state.pose[:] = mano_pose
hand_state.recompute_mesh(manobase2miabase)
fig.add_geometry(hand_state.hand_mesh)

kin = load_kinematic_model(MIA_CONFIG)

emb = HandEmbodiment(hand_state, kin, MIA_CONFIG)

import time
start = time.time()

index_tip2miabase, q = emb.solve()

print(time.time() - start)

pose = emb.index_chain.forward(q)
fig.plot_sphere(0.005, index_tip2miabase, c=(0, 0, 0))
fig.plot_sphere(0.005, pose, c=(1, 0, 0))

graph = pv.Graph(kin.tm, MIA_CONFIG["base_frame"], show_frames=False, show_connections=False,
                 show_visuals=True, show_collision_objects=False, show_name=False,
                 s=0.02)
graph.add_artist(fig)

fig.show()
