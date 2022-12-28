import pytransform3d.trajectories
import numpy as np

with open("/home/dfki.uni-bremen.de/afabisch/noetic_ws/ur_policy_control/data.txt", "r") as f:
    data = f.read() 
pose = np.array(data) 
result = pytransform3d.trajectories.pqs_from_transforms(pose)

