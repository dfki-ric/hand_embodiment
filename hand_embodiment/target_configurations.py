import numpy as np
import pytransform3d.transformations as pt
import pytransform3d.rotations as pr


MIA_CONFIG = {
    "joint_names":
        {
            "thumb": ["jMCP1", "jmetacarpus"],
            "index": ["jMCP2"],
            "middle": ["jMCP3"],
            "ring": ["jMCP4"],
            "little": ["jMCP5"],
        }
}
manobase2miabase = pt.transform_from(
    R=pr.active_matrix_from_intrinsic_euler_xyz(np.array([-1.634, 1.662, -0.182])),
    p=np.array([0.002, 0.131, -0.024]))
