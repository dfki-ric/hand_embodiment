import time
import numpy as np

from .kinematics import Kinematics
from .record_markers import make_finger_kinematics
import pytransform3d.transformations as pt


class HandEmbodiment:
    """Solves embodiment mapping from MANO model to robotic hand.

    Parameters
    ----------
    hand_state : HandState
        State of the MANO mesh, defined internally by pose and shape
        parameters, and whether it is a left or right hand. It also
        stores the mesh.

    target_config : dict
        Configuration for the target system.

    use_fingers : tuple of str, optional (default: ('thumb', 'index', 'middle'))
        Fingers for which we compute the embodiment mapping.

    mano_finger_kinematics : list, optional (default: None)
        If finger kinematics are already available, e.g., from the record
        mapping, these can be passed here. Otherwise they will be created.

    initial_handbase2world : array-like, shape (4, 4), optional (default: None)
        Initial transform from hand base to world coordinates.

    verbose : int, optional (default: 0)
        Verbosity level
    """
    def __init__(
            self, hand_state, target_config,
            use_fingers=("thumb", "index", "middle"),
            mano_finger_kinematics=None, initial_handbase2world=None,
            verbose=0):
        self.finger_names = use_fingers
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
        self.base_frame = target_config["base_frame"]
        for finger_name in use_fingers:
            assert finger_name in target_config["joint_names"]
            assert finger_name in target_config["ee_frames"]
            self.target_finger_chains[finger_name] = \
                self.target_kin.create_chain(
                    target_config["joint_names"][finger_name],
                    self.base_frame,
                    target_config["ee_frames"][finger_name])
            self.joint_angles[finger_name] = \
                np.zeros(len(target_config["joint_names"][finger_name]))

        self._update_hand_base_pose(initial_handbase2world)

        self.verbose = verbose

    def solve(self, handbase2world=None, return_desired_positions=False):
        if self.verbose:
            start = time.time()

        if return_desired_positions:
            desired_positions = {}

        for finger_name in self.finger_names:
            # MANO forward kinematics
            finger_tip_in_manobase = self.mano_finger_kinematics[finger_name].forward(
                self.hand_state.pose[self.mano_finger_kinematics[finger_name].finger_pose_param_indices])
            finger_tip_in_handbase = pt.transform(
                self.handbase2robotbase,
                pt.vector_to_point(finger_tip_in_manobase))[:3]

            # Hand inverse kinematics
            self.joint_angles[finger_name] = \
                self.target_finger_chains[finger_name].inverse_position(
                    finger_tip_in_handbase, self.joint_angles[finger_name])

            if return_desired_positions:
                desired_positions[finger_name] = finger_tip_in_handbase

        self._update_hand_base_pose(handbase2world)

        if self.verbose:
            stop = time.time()
            duration = stop - start
            print(f"[{type(self).__name__}] Time for optimization: "
                  f"{duration:.4f} s")

        if return_desired_positions:
            return self.joint_angles, desired_positions
        else:
            return self.joint_angles

    def _update_hand_base_pose(self, handbase2world):
        if handbase2world is None:
            world2robotbase = np.eye(4)
        else:
            world2robotbase = pt.concat(
                pt.invert_transform(handbase2world, check=False),
                self.handbase2robotbase)
        self.target_kin.tm.add_transform(
            "world", self.base_frame, world2robotbase)


def load_kinematic_model(hand_config):
    model = hand_config["model"]
    with open(model["urdf"], "r") as f:
        kin = Kinematics(urdf=f.read(), package_dir=model["package_dir"])
    if "kinematic_model_hook" in model:
        model["kinematic_model_hook"](kin)
    return kin
