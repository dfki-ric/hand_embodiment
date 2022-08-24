"""Embodiment mapping.

Maps MANO states to robotic hands.
"""
import warnings
import numpy as np

from .kinematics import Kinematics
from .record_markers import make_finger_kinematics
from .target_configurations import TARGET_CONFIG
from .timing import TimeableMixin
import pytransform3d.transformations as pt


class HandEmbodiment(TimeableMixin):
    """Solves embodiment mapping from MANO model to robotic hand.

    Parameters
    ----------
    hand_state : HandState
        State of the MANO mesh, defined internally by pose and shape
        parameters, and whether it is a left or right hand. It also
        stores the mesh.

    target_config : dict or str
        Configuration for the target system as a dictionary or string that
        identifies the target hand.

    use_fingers : tuple of str, optional (default: ('thumb', 'index', 'middle'))
        Fingers for which we compute the embodiment mapping.

    mano_finger_kinematics : list, optional (default: None)
        If finger kinematics are already available, e.g., from the record
        mapping, these can be passed here. Otherwise they will be created.

    initial_handbase2world : array-like, shape (4, 4), optional (default: None)
        Initial transform from hand base to world coordinates.

    only_tip : bool, optional (default: False)
        Use only tip frames in inverse kinematics.

    verbose : int, optional (default: 0)
        Verbosity level

    measure_time : bool
        Measure computation time for each frame.

    Attributes
    ----------
    finger_names_ : tuple of str
        Fingers for which we compute the embodiment mapping.

    hand_state_ : mocap.mano.HandState
        MANO hand state. This state should be updated by the record mapping
        so that we can perform a subsequent embodiment mapping based on the
        current state.

    transform_manager_ : pytransform3d.transform_manager.TransformManager
        Exposes transform manager that represents the target system.
        It holds information about all links, joints, visual objects, and
        collision objects of the target system (robotic hand). It is used
        to compute forward and inverse kinematics and can be used for
        visualization. This representation of the target system's state will
        be updated with each embodiment mapping.
    """
    def __init__(
            self, hand_state, target_config,
            use_fingers=("thumb", "index", "middle"),
            mano_finger_kinematics=None, initial_handbase2world=None,
            only_tip=False, verbose=0, measure_time=False):
        super(HandEmbodiment, self).__init__(verbose or measure_time)

        if isinstance(target_config, str):
            target_config = TARGET_CONFIG[target_config]
        self.finger_names_ = use_fingers
        self.hand_state_ = hand_state
        if mano_finger_kinematics is None:
            self.mano_finger_kinematics = {}
            for finger_name in use_fingers:
                self.mano_finger_kinematics[finger_name] = \
                    make_finger_kinematics(self.hand_state_, finger_name)
        else:
            self.mano_finger_kinematics = mano_finger_kinematics
        self.handbase2robotbase = target_config["handbase2robotbase"]
        for finger_name in use_fingers:
            assert finger_name in self.mano_finger_kinematics

        self.target_kin, self.vis_kin = load_kinematic_model(target_config)
        self.target_finger_chains = {}
        self.vis_finger_chains = {}
        self.joint_angles = {}
        self.base_frame = target_config["base_frame"]
        for finger_name in target_config["ee_frames"].keys():
            assert finger_name in target_config["joint_names"]
            assert finger_name in target_config["ee_frames"]
            ee_frames = [target_config["ee_frames"][finger_name]]
            if not only_tip and "intermediate_frames" in target_config:
                ee_frames.append(
                    target_config["intermediate_frames"][finger_name])
            self.target_finger_chains[finger_name] = \
                self.target_kin.create_multi_chain(
                    target_config["joint_names"][finger_name],
                    self.base_frame, ee_frames)
            self.vis_finger_chains[finger_name] = \
                self.vis_kin.create_multi_chain(
                    target_config["joint_names"][finger_name],
                    self.base_frame, ee_frames)
            self.joint_angles[finger_name] = \
                np.zeros(len(target_config["joint_names"][finger_name]))

        self._update_hand_base_pose(initial_handbase2world)

        self.coupled_joints = target_config.get("coupled_joints", None)
        self.post_embodiment_hook = target_config.get(
            "post_embodiment_hook", None)

        self.verbose = verbose

    def solve(self, handbase2world=None, return_desired_positions=False,
              use_cached_forward_kinematics=False):
        """Solve embodiment.

        Internally we rely on the attribute hand_state_ to be updated
        before this function is called. The hand state will contain all
        necessary information to perform the embodiment of finger
        configurations.

        Parameters
        ----------
        handbase2world : array-like, shape (4, 4), optional (default: None)
            Transform from MANO base to world.

        return_desired_positions : bool, optional (default: False)
            Return desired position for each finger in robot's base frame.

        use_cached_forward_kinematics : bool, optional (defalt: False)
            If the underlying MANO model was previously fitted from data,
            you can use the cached results. If you want to use this option
            you should provide the constructor argument
            'mano_finger_kinematics'.

        Returns
        -------
        joint_angles : dict
            Maps finger names to corresponding joint angles in the order that
            is given in the target configuration.

        desired_positions : dict, optional
            Maps finger names to desired finger tip positions in robot base
            frame.
        """
        self.start_measurement()

        desired_positions = self._mano_forward_kinematics(use_cached_forward_kinematics)
        self._robotic_hand_inverse_kinematics(desired_positions)
        self._update_hand_base_pose(handbase2world)

        self.stop_measurement()
        if self.verbose:
            print(f"[{type(self).__name__}] Time for optimization: "
                  f"{self.last_timing():.4f} s")

        if self.vis_kin is not self.target_kin:
            for finger_name in self.finger_names_:
                self.vis_finger_chains[finger_name].forward(
                    self.joint_angles[finger_name])

        if return_desired_positions:
            return self.joint_angles, desired_positions
        else:
            return self.joint_angles

    def _mano_forward_kinematics(self, use_cached_forward_kinematics):
        """MANO forward kinematics.

        Parameters
        ----------
        use_cached_forward_kinematics : bool
            Reused cached result of forward kinematics from record mapping
            in MANO model.

        Returns
        -------
        desired_positions : dict
            Desired positions of expected marker positions in frame of the
            robotic target system.
        """
        desired_positions = {}
        for finger_name in self.finger_names_:
            if use_cached_forward_kinematics:
                # because it has been computed during record mapping
                markers_in_handbase = self.mano_finger_kinematics[finger_name].forward(
                    None, return_cached_result=True)
            else:
                markers_in_handbase = self.mano_finger_kinematics[finger_name].forward(
                    self.hand_state_.pose[
                        self.mano_finger_kinematics[
                            finger_name].finger_pose_param_indices])

            markers_in_robotbase = pt.transform(
                self.handbase2robotbase,
                pt.vectors_to_points(markers_in_handbase))[:, :3]
            n_ee_frames = len(self.target_finger_chains[finger_name].ee_frames)
            desired_positions[finger_name] = markers_in_robotbase[:n_ee_frames]
        return desired_positions

    def _robotic_hand_inverse_kinematics(self, desired_positions):
        """Robotic hand inverse kinematics.

        Parameters
        ----------
        desired_positions : dict
            Desired positions of expected marker positions in frame of the
            robotic target system.
        """
        for finger_name in self.finger_names_:
            self.joint_angles[finger_name] = \
                self.target_finger_chains[finger_name].inverse_position(
                    desired_positions[finger_name],
                    self.joint_angles[finger_name])

        updated_fingers = set()
        if self.coupled_joints is not None:
            updated_fingers.update(self._average_coupled_joints())

        if self.post_embodiment_hook is not None:
            updated_fingers.update(self.post_embodiment_hook(self.joint_angles))

        for finger_name in updated_fingers:
            self.finger_forward_kinematics(
                finger_name, self.joint_angles[finger_name])

    def _average_coupled_joints(self):
        """Average joint angles of coupled joints that move together."""
        angle_sum = 0.0
        n_joints = 0
        for finger_name, joint_name in self.coupled_joints:
            if finger_name not in self.finger_names_:
                continue
            angle_sum += self.joint_angles[finger_name][
                self.target_finger_chains[
                    finger_name].joint_names.index(joint_name)]
            n_joints += 1

        if n_joints == 0:
            return

        average_angle = angle_sum / n_joints
        updated_fingers = set()
        for finger_name, joint_name in self.coupled_joints:
            self.joint_angles[finger_name][
                self.target_finger_chains[finger_name].joint_names.index(
                    joint_name)] = average_angle
            updated_fingers.add(finger_name)
        return updated_fingers

    def _update_hand_base_pose(self, handbase2world):
        """Compute pose of the base of the robotic target hand.

        Parameters
        ----------
        handbase2world : array, shape (4, 4)
            Transform from base of the robotic hand to world frame.
        """
        if handbase2world is None:
            world2robotbase = np.eye(4)
        else:
            world2robotbase = pt.concat(
                pt.invert_transform(handbase2world, check=False),
                self.handbase2robotbase)
        self.target_kin.tm.add_transform(
            "world", self.base_frame, world2robotbase)

        if self.vis_kin is not self.target_kin:
            self.vis_kin.tm.add_transform(
                "world", self.base_frame, world2robotbase)

    @property
    def transform_manager_(self):
        """Get transform manager."""
        return self.vis_kin.tm

    def finger_forward_kinematics(self, finger_name, joint_angles):
        """Forward kinematics for a finger of the target system.

        Parameters
        ----------
        finger_name : str
            Name of the finger.

        joint_angles : array-like, shape (n_joints,)
            Angles of the finger joints.

        Returns
        -------
        """
        return self.target_finger_chains[finger_name].forward(joint_angles)


def load_kinematic_model(hand_config, unscaled_visual_model=True):
    """Load kinematic model of a robotic hand.

    Parameters
    ----------
    hand_config : dict
        Configuration of robotic target hand.

    unscaled_visual_model : bool, optional (default: True)
        Load an unscaled visual model of the hand in addition to the scaled
        version.

    Returns
    -------
    kin : Kinematics
        Forward and inverse kinematics.

    vis_kin : Kinematics
        Forward and inverse kinematics for visualization.
    """
    model = hand_config["model"]
    extra_args = {}
    if "package_dir" in model:
        extra_args["package_dir"] = model["package_dir"]
    if "mesh_path" in model:
        extra_args["mesh_path"] = model["mesh_path"]
    extra_args["scale"] = hand_config.get("scale", 1.0)
    with open(model["urdf"], "r") as f:
        kin = Kinematics(urdf=f.read(), **extra_args)
    if unscaled_visual_model:
        extra_args.pop("scale")
        with open(model["urdf"], "r") as f:
            vis_kin = Kinematics(urdf=f.read(), **extra_args)
    else:
        vis_kin = kin
    if "kinematic_model_hook" in model:
        kinematic_model_hook_args = hand_config.get("kinematic_model_hook_args", {})
        model["kinematic_model_hook"](kin, **kinematic_model_hook_args)
        if unscaled_visual_model:
            model["kinematic_model_hook"](vis_kin)
    if "virtual_joints_callbacks" in hand_config:
        for joint_name, callback in hand_config["virtual_joints_callbacks"].items():
            kin.tm.add_virtual_joint(joint_name, callback)
            if unscaled_visual_model:
                vis_kin.tm.add_virtual_joint(joint_name, callback)
    if "world" in kin.tm.nodes:
        warnings.warn(
            "'world' frame is already in URDF. Removing all connections to it.")
        invalid_connections = []
        for from_frame, to_frame in kin.tm.transforms:
            if "world" in (from_frame, to_frame):
                invalid_connections.append((from_frame, to_frame))
        for from_frame, to_frame in invalid_connections:
            kin.tm.remove_transform(from_frame, to_frame)
            if unscaled_visual_model:
                vis_kin.tm.remove_transform(from_frame, to_frame)
    return kin, vis_kin
