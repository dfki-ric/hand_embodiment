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


manobase2shadowbase = pt.transform_from(
    R=pr.active_matrix_from_intrinsic_euler_xyz(np.array([-3.08, 1.427, 2.923])),
    p=np.array([0.01, -0.01, 0.36]))
SHADOW_HAND_CONFIG = {
    "joint_names":
        {  # wrist: rh_WRJ2, rh_WRJ1
            "thumb": ["rh_THJ5", "rh_THJ4", "rh_THJ3", "rh_THJ2", "rh_THJ1"],
            "index": ["rh_FFJ4", "rh_FFJ3", "rh_FFJ2", "rh_FFJ1"],
            "middle": ["rh_MFJ4", "rh_MFJ3", "rh_MFJ2", "rh_MFJ1"],
            "ring": ["rh_RFJ4", "rh_RFJ3", "rh_RFJ2", "rh_RFJ1"],
            "little": ["rh_LFJ5", "rh_LFJ4", "rh_LFJ3", "rh_LFJ2", "rh_LFJ1"],
        },
    "base_frame": "rh_forearm",
    "ee_frames":
        {
            "thumb": "rh_thtip",
            "index": "rh_fftip",
            "middle": "rh_mftip",
            "ring": "rh_rftip",
            "little": "rh_lftip"
        },
    "handbase2robotbase": manobase2shadowbase,
    "model":
        {
            "urdf": resource_filename(
                "hand_embodiment",
                "model/sr_common/sr_description/urdf/shadow_hand.urdf"),
            "package_dir": resource_filename(
                "hand_embodiment", "model/sr_common/"),
            "kinematic_model_hook": lambda x: x  # TODO
        }
}
