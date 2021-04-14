import numpy as np

from .kinematics import Kinematics
from .record_markers import make_finger_kinematics
import pytransform3d.transformations as pt


class HandEmbodiment:
    def __init__(
            self, hand_state, target_config,
            use_fingers=("thumb", "index", "middle"),
            mano_finger_kinematics=None):
        self.hand_state = hand_state
        if mano_finger_kinematics is None:
            self.mano_finger_kinematics = {}
            for finger_name in use_fingers:
                self.mano_finger_kinematics[finger_name] = \
                    make_finger_kinematics(self.hand_state, finger_name)
        else:
            self.mano_finger_kinematics = mano_finger_kinematics
        self.handbase2robotbase = target_config["handbase2robotbase"]
        for finger_name in use_fingers:
            assert finger_name in self.mano_finger_kinematics

        self.target_kin = load_kinematic_model(target_config)
        self.target_finger_chains = {}
        self.joint_angles = {}
        for finger_name in use_fingers:
            assert finger_name in target_config["joint_names"]
            assert finger_name in target_config["ee_frames"]
            self.target_finger_chains[finger_name] = \
                self.target_kin.create_chain(
                    target_config["joint_names"][finger_name],
                    target_config["base_frame"],
                    target_config["ee_frames"][finger_name])
            self.joint_angles[finger_name] = \
                np.zeros(len(target_config["joint_names"][finger_name]))

    def solve(self):
        # TODO for each finger
        index_tip_in_manobase = self.mano_finger_kinematics["index"].forward(
            self.hand_state.pose[self.mano_finger_kinematics["index"].finger_pose_param_indices])
        index_tip_in_handbase = pt.transform(
            self.handbase2robotbase, pt.vector_to_point(index_tip_in_manobase))
        self.joint_angles["index"] = self.target_finger_chains["index"].inverse_position(index_tip_in_handbase[:3], self.joint_angles["index"])
        return pt.translate_transform(np.eye(4), index_tip_in_handbase), self.joint_angles


def load_kinematic_model(hand_config):
    model = hand_config["model"]
    with open(model["urdf"], "r") as f:
        kin = Kinematics(urdf=f.read(), package_dir=model["package_dir"])
    if "kinematic_model_hook" in model:
        model["kinematic_model_hook"](kin)
    return kin
