import numpy as np
import pytransform3d.transformations as pt
import pytransform3d.rotations as pr
from pkg_resources import resource_filename


###############################################################################
# Mia hand
###############################################################################

def kinematic_model_hook_mia(kin):
    """Extends kinematic model to include links for embodiment mapping."""
    # adjust index finger limit
    joint_info = kin.tm._joints["j_index_fle"]
    joint_info = (joint_info[0], joint_info[1], joint_info[2], joint_info[3],
                  (0.0, joint_info[4][1]), joint_info[5])
    kin.tm._joints["j_index_fle"] = joint_info

    kin.tm.add_transform(
        "thumb_tip", "thumb_fle",
        np.array([
            [1, 0, 0, 0.03],
            [0, 1, 0, 0.07],
            [0, 0, 1, 0],
            [0, 0, 0, 1]]))
    kin.tm.add_transform(
        "thumb_middle", "thumb_fle",
        np.array([
            [1, 0, 0, 0.03],
            [0, 1, 0, 0.015],
            [0, 0, 1, 0],
            [0, 0, 0, 1]]))
    kin.tm.add_transform(
        "index_tip", "index_fle",
        np.array([
            [1, 0, 0, -0.015],
            [0, 1, 0, 0.09],
            [0, 0, 1, 0],
            [0, 0, 0, 1]]))
    kin.tm.add_transform(
        "index_middle", "index_fle",
        np.array([
            [1, 0, 0, 0.02],
            [0, 1, 0, 0.015],
            [0, 0, 1, 0],
            [0, 0, 0, 1]]))
    kin.tm.add_transform(
        "middle_tip", "middle_fle",
        np.array([
            [1, 0, 0, -0.015],
            [0, 1, 0, 0.09],
            [0, 0, 1, 0],
            [0, 0, 0, 1]]))
    kin.tm.add_transform(
        "middle_middle", "middle_fle",
        np.array([
            [1, 0, 0, 0.02],
            [0, 1, 0, 0.015],
            [0, 0, 1, 0],
            [0, 0, 0, 1]]))
    kin.tm.add_transform(
        "ring_tip", "ring_fle",
        np.array([
            [1, 0, 0, -0.012],
            [0, 1, 0, 0.083],
            [0, 0, 1, 0],
            [0, 0, 0, 1]]))
    kin.tm.add_transform(
        "ring_middle", "ring_fle",
        np.array([
            [1, 0, 0, 0.017],
            [0, 1, 0, 0.015],
            [0, 0, 1, 0],
            [0, 0, 0, 1]]))
    kin.tm.add_transform(
        "little_tip", "little_fle",
        np.array([
            [1, 0, 0, -0.01],
            [0, 1, 0, 0.068],
            [0, 0, 1, 0],
            [0, 0, 0, 1]]))
    kin.tm.add_transform(
        "little_middle", "little_fle",
        np.array([
            [1, 0, 0, 0.015],
            [0, 1, 0, 0.015],
            [0, 0, 1, 0],
            [0, 0, 0, 1]]))


class MiaVirtualThumbJoint:
    """Positive values will be mapped to max limit, negative values to min."""
    def __init__(self, real_joint_name):
        self.real_joint_name = real_joint_name
        self.min_angle = -0.628
        self.max_angle = 0.0
        self.angle_threshold = 0.5 * (self.min_angle + self.max_angle)

    def make_virtual_joint(self, joint_name, tm):
        limits = tm.get_joint_limits(self.real_joint_name)

        return (joint_name + "_from", joint_name + "_to", np.eye(4),
                np.array([0, 0, 0]), limits, "revolute")

    def __call__(self, value):
        if value >= 0:
            angle = self.max_angle
        else:
            angle = self.min_angle
        return {self.real_joint_name: angle}


# TODO make this configurable
#manobase2miabase = pt.transform_from(
#    R=pr.active_matrix_from_intrinsic_euler_xyz(np.array([-1.634, 1.662, -0.182])),
#    p=np.array([0.002, 0.131, -0.024]))
manobase2miabase = pt.transform_from_exponential_coordinates(
    [-1.348, 0.865, -1.38, -0.105, 0.12, 0.041])
#manobase2miabase = pt.transform_from_exponential_coordinates(
#    [-1.348, 0.476, -1.707, -0.088, 0.078, 0.014])
MIA_CONFIG = {
    "joint_names":
        {  # map finger names to a list of joint names that control the finger
            "thumb": ["j_thumb_fle"],
            "index": ["j_index_fle"],
            "middle": ["j_mrl_fle"],
            "ring": ["j_ring_fle"],
            "little": ["j_little_fle"],
        },
    "base_frame": "palm",  # base frame of the hand
    "ee_frames":
        {  # map finger name to the name of the tip frame in the kinematic model
            "thumb": "thumb_tip",
            "index": "index_tip",
            "middle": "middle_tip",
            "ring": "ring_tip",
            "little": "little_tip"
        },
    "intermediate_frames":
        {
            "thumb": "thumb_middle",
            "index": "index_middle",
            "middle": "middle_middle",
            "ring": "ring_middle",
            "little": "little_middle"
        },
    "handbase2robotbase": manobase2miabase,  # transform from MANO base to hand base
    "model":  # kinematic model definition
        {
            # this xacro is actually just plain urdf:
            "urdf": resource_filename(
                "hand_embodiment",
                "model/mia_hand_description/urdf/mia_hand.urdf.xacro"),
            "package_dir": resource_filename("hand_embodiment", "model/"),
            "kinematic_model_hook": kinematic_model_hook_mia
        },
    "virtual_joints_callbacks":
        {  # here we can introduce virtual joints that compute the state of real joints in a callback
            "j_thumb_opp_binary": MiaVirtualThumbJoint("j_thumb_opp"),
        },
    "coupled_joints":
    [  # coupled joints always have the same angle
        ("middle", "j_mrl_fle"),
        ("ring", "j_ring_fle"),
        ("little", "j_little_fle"),
    ]
}


###############################################################################
# Shadow dexterous hand
###############################################################################

def kinematic_model_hook_shadow(kin):
    """Extends kinematic model to include links for embodiment mapping."""
    kin.tm.add_transform(
        "thumb_tip", "rh_thtip",
        np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0.013],
            [0, 0, 1, -0.01],
            [0, 0, 0, 1]]))
    kin.tm.add_transform(
        "thumb_middle", "rh_thmiddle",
        np.array([
            [1, 0, 0, 0.015],
            [0, 1, 0, 0],
            [0, 0, 1, 0.01],
            [0, 0, 0, 1]]))
    kin.tm.add_transform(
        "index_tip", "rh_fftip",
        np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0.01],
            [0, 0, 1, -0.01],
            [0, 0, 0, 1]]))
    kin.tm.add_transform(
        "index_middle", "rh_ffproximal",
        np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0.015],
            [0, 0, 1, 0.02],
            [0, 0, 0, 1]]))
    kin.tm.add_transform(
        "middle_tip", "rh_mftip",
        np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0.01],
            [0, 0, 1, 0],
            [0, 0, 0, 1]]))
    kin.tm.add_transform(
        "middle_middle", "rh_mfproximal",
        np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0.015],
            [0, 0, 1, 0.02],
            [0, 0, 0, 1]]))
    kin.tm.add_transform(
        "ring_tip", "rh_rftip",
        np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0.01],
            [0, 0, 1, -0.013],
            [0, 0, 0, 1]]))
    kin.tm.add_transform(
        "ring_middle", "rh_rfproximal",
        np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0.015],
            [0, 0, 1, 0.02],
            [0, 0, 0, 1]]))
    kin.tm.add_transform(
        "little_tip", "rh_lftip",
        np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0.01],
            [0, 0, 1, -0.025],
            [0, 0, 0, 1]]))
    kin.tm.add_transform(
        "little_middle", "rh_lfproximal",
        np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0.015],
            [0, 0, 1, 0.02],
            [0, 0, 0, 1]]))


class ShadowVirtualF0Joint:
    def __init__(self, first_real_joint_name, second_real_joint_name):
        self.first_real_joint_name = first_real_joint_name
        self.second_real_joint_name = second_real_joint_name
        self.first_joint_max = 0.5 * np.pi

    def make_virtual_joint(self, joint_name, tm):
        first_limits = tm.get_joint_limits(self.first_real_joint_name)
        second_limits = tm.get_joint_limits(self.second_real_joint_name)

        joint_range = (second_limits[1] - second_limits[0] +
                       first_limits[1] - first_limits[0])
        return (joint_name + "_from", joint_name + "_to", np.eye(4),
                np.array([0, 0, 0]), (0, joint_range), "revolute")

    def __call__(self, value):
        if value > self.first_joint_max:
            first_joint_value = self.first_joint_max
            second_joint_value = value - self.first_joint_max
        else:
            first_joint_value = value
            second_joint_value = 0.0
        return {self.first_real_joint_name: first_joint_value,
                self.second_real_joint_name: second_joint_value}


manobase2shadowbase = pt.transform_from(
    R=pr.active_matrix_from_intrinsic_euler_xyz(np.array([-3.17, 1.427, 3.032])),
    p=np.array([0.011, -0.014, 0.36]))
SHADOW_HAND_CONFIG = {
    "joint_names":
        {  # wrist: rh_WRJ2, rh_WRJ1
            "thumb": ["rh_THJ5", "rh_THJ4", "rh_THJ3", "rh_THJ2", "rh_THJ1"],
            "index": ["rh_FFJ4", "rh_FFJ3", "rh_FFJ0"],
            "middle": ["rh_MFJ4", "rh_MFJ3", "rh_MFJ0"],
            "ring": ["rh_RFJ4", "rh_RFJ3", "rh_RFJ0"],
            "little": ["rh_LFJ5", "rh_LFJ4", "rh_LFJ3", "rh_LFJ0"],
        },
    "base_frame": "rh_forearm",
    "ee_frames":
        {
            "thumb": "thumb_tip",
            "index": "index_tip",
            "middle": "middle_tip",
            "ring": "ring_tip",
            "little": "little_tip"
        },
    "intermediate_frames":
        {
            "thumb": "thumb_middle",
            "index": "index_middle",
            "middle": "middle_middle",
            "ring": "ring_middle",
            "little": "little_middle"
        },
    "handbase2robotbase": manobase2shadowbase,
    "model":
        {
            "urdf": resource_filename(
                "hand_embodiment",
                "model/sr_common/sr_description/urdf/shadow_hand.urdf"),
            "package_dir": resource_filename(
                "hand_embodiment", "model/sr_common/"),
            "kinematic_model_hook": kinematic_model_hook_shadow
        },
    "virtual_joints_callbacks":
        {
            "rh_FFJ0": ShadowVirtualF0Joint("rh_FFJ2", "rh_FFJ1"),
            "rh_MFJ0": ShadowVirtualF0Joint("rh_MFJ2", "rh_MFJ1"),
            "rh_RFJ0": ShadowVirtualF0Joint("rh_RFJ2", "rh_RFJ1"),
            "rh_LFJ0": ShadowVirtualF0Joint("rh_LFJ2", "rh_LFJ1"),
        }
}


###############################################################################
# Selection
###############################################################################

TARGET_CONFIG = {
    "mia": MIA_CONFIG,
    "shadow_hand": SHADOW_HAND_CONFIG,
    "shadow": SHADOW_HAND_CONFIG
}
