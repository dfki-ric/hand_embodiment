import numpy as np
import pytransform3d.transformations as pt
import pytransform3d.rotations as pr
from pkg_resources import resource_filename


def kinematic_model_hook_mia(kin):
    """Extends kinematic model to include links for embodiment mapping."""
    kin.tm.add_transform(
        "thumb_tip", "MCP1",
        np.array([
            [1, 0, 0, 0.025],
            [0, 1, 0, 0.08],
            [0, 0, 1, 0],
            [0, 0, 0, 1]]))
    kin.tm.add_transform(
        "index_tip", "MCP2",
        np.array([
            [1, 0, 0, -0.02],
            [0, 1, 0, 0.09],
            [0, 0, 1, 0],
            [0, 0, 0, 1]]))
    kin.tm.add_transform(
        "middle_tip", "MCP3",
        np.array([
            [1, 0, 0, -0.02],
            [0, 1, 0, 0.09],
            [0, 0, 1, 0],
            [0, 0, 0, 1]]))
    kin.tm.add_transform(
        "ring_tip", "MCP4",
        np.array([
            [1, 0, 0, -0.013],
            [0, 1, 0, 0.083],
            [0, 0, 1, 0],
            [0, 0, 0, 1]]))
    kin.tm.add_transform(
        "little_tip", "MCP5",
        np.array([
            [1, 0, 0, -0.009],
            [0, 1, 0, 0.065],
            [0, 0, 1, 0],
            [0, 0, 0, 1]]))


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
            "thumb": "thumb_tip",
            "index": "index_tip",
            "middle": "middle_tip",
            "ring": "ring_tip",
            "little": "little_tip"
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
