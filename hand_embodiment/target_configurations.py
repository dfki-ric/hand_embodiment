import numpy as np
import pytransform3d.transformations as pt
import pytransform3d.rotations as pr
from pkg_resources import resource_filename


def kinematic_model_hook_mia(kin):
    """Extends kinematic model to include links for embodiment mapping."""
    kin.tm.add_transform(
        "index_tip", "MCP2",
        np.array([
            [8.44588589e-02, -5.11060069e-01, -8.55385473e-01, -1.93295715e-02],
            [1.73649242e-01, 8.52865505e-01, -4.92408744e-01, 0.09],
            [9.81179210e-01, -1.06948758e-01, 1.60777241e-01, 5.84254784e-04],
            [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 1.00000000e+00]]))


manobase2miabase = pt.transform_from(
    R=pr.active_matrix_from_intrinsic_euler_xyz(np.array([-1.634, 1.662, -0.182])),
    p=np.array([0.002, 0.131, -0.024]))
MIA_CONFIG = {
    "joint_names":
        {
            "thumb": ["jMCP1", "jmetacarpus"],
            "index": ["jMCP2"],
            "middle": ["jMCP3"],
            "ring": ["jMCP4"],
            "little": ["jMCP5"],
        },
    "base_frame": "wrist",
    "ee_frames":
        {
            "index": "index_tip"
        },
    "handbase2robotbase": manobase2miabase,
    "model":
        {
            "urdf": resource_filename(
                "hand_embodiment",
                "model/mia_hand_description/urdf/mia_hand.urdf"),
            "package_dir": resource_filename("hand_embodiment", "model/"),
            "kinematic_model_hook": kinematic_model_hook_mia
        }
}
