"""Pipelines are high-level interfaces that map human data to robotic hands."""
import json
import os

import numpy as np
import pytransform3d.transformations as pt
import yaml

from hand_embodiment.target_configurations import TARGET_CONFIG
from hand_embodiment.config import load_mano_config, load_record_mapping_config
from hand_embodiment.record_markers import MarkerBasedRecordMapping
from hand_embodiment.embodiment import HandEmbodiment


class MoCapToRobot:
    """Combines record mapping for motion capture and embodiment mapping.

    Parameters
    ----------
    hand : str
        Name of the hand: 'mia' or 'shadow'

    mano_config : str
        Path to MANO configuration

    use_fingers : list of str
        Fingers that should be used

    record_mapping_config : str, optional (default: None)
        Path to record mapping configuration

    verbose : int, optional (default: 0)
        Verbosity level

    measure_time : bool
        Measure computation time for each frame.

    robot_config : str, optional (default: None)
        Target system configuration.
    """
    def __init__(self, hand, mano_config, use_fingers,
                 record_mapping_config=None, verbose=0, measure_time=False,
                 robot_config=None, objective='cartesian', smooth_w=None,
                 adapt_handbase=False, adapt_lr=1e-3, adapt_reg=1e-4,
                 adapt_max_m=0.05, handbase_cache=None):
        self.hand_config_ = self._hand_config(hand, robot_config)
        self._handbase_cache = handbase_cache
        mano2hand_markers, betas = load_mano_config(mano_config)

        if record_mapping_config is not None:
            record_mapping_config = load_record_mapping_config(
                record_mapping_config)

        self.record_mapping_ = MarkerBasedRecordMapping(
            left=False, mano2hand_markers=mano2hand_markers,
            shape_parameters=betas,
            record_mapping_config=record_mapping_config,
            use_fingers=use_fingers, verbose=verbose,
            measure_time=measure_time)
        if handbase_cache is not None and os.path.isfile(handbase_cache):
            self._load_handbase_cache(handbase_cache)
        self.embodiment_mapping_ = HandEmbodiment(
            self.record_mapping_.hand_state_, self.hand_config_,
            use_fingers=use_fingers,
            mano_finger_kinematics=self.record_mapping_.mano_finger_kinematics_,
            initial_handbase2world=self.record_mapping_.mano2world_,
            verbose=verbose, measure_time=measure_time,
            objective=objective, smooth_w=smooth_w,
            adapt_handbase=adapt_handbase, adapt_lr=adapt_lr,
            adapt_reg=adapt_reg, adapt_max_m=adapt_max_m)

    def save_handbase_cache(self, path=None):
        """Save adapted handbase2robotbase to a JSON cache file.

        Parameters
        ----------
        path : str, optional
            File path. Defaults to self._handbase_cache if set.
        """
        path = path or self._handbase_cache
        if path is None:
            raise ValueError("No cache path specified.")
        T = self.embodiment_mapping_.handbase2robotbase
        data = {"handbase2robotbase": pt.exponential_coordinates_from_transform(T).tolist()}
        if hasattr(self.embodiment_mapping_, '_handbase2robotbase_init'):
            T_init = self.embodiment_mapping_._handbase2robotbase_init
            data["original_handbase2robotbase"] = (
                pt.exponential_coordinates_from_transform(T_init).tolist())
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    def _load_handbase_cache(self, path):
        """Load handbase2robotbase from a JSON cache file into hand_config_."""
        with open(path, "r") as f:
            data = json.load(f)
        if "handbase2robotbase" in data:
            self.hand_config_["handbase2robotbase"] = \
                pt.transform_from_exponential_coordinates(
                    np.array(data["handbase2robotbase"]))

    def _hand_config(self, hand, robot_config):
        hand_config_ = TARGET_CONFIG[hand]
        if robot_config is not None:
            with open(robot_config, "r") as f:
                hand_config = yaml.safe_load(f)
                if "handbase2robotbase" in hand_config:
                    hand_config["handbase2robotbase"] = \
                        pt.transform_from_exponential_coordinates(
                            hand_config["handbase2robotbase"])
                hand_config_.update(hand_config)
        return hand_config_

    @property
    def transform_manager_(self):
        """Graph that represents state of robotic hand."""
        return self.embodiment_mapping_.transform_manager_

    def reset(self):
        """Reset record mapping."""
        self.record_mapping_.reset()

    def set_constant_joint(self, joint_name, angle):
        """Set constant joint angle of target hand.

        Parameters
        ----------
        joint_name : str
            Name of the joint in the URDF.

        angle : float
            Joint angle.
        """
        self.transform_manager_.set_joint(joint_name, angle)

    def estimate_hand(self, hand_markers, finger_markers):
        """Estimate MANO pose and joint angles.

        Parameters
        ----------
        hand_markers : list
            Markers on hand in order 'hand_top', 'hand_left', 'hand_right'.

        finger_markers : dict (str to array-like)
            Positions of markers on fingers.
        """
        assert len(hand_markers) == 3, hand_markers
        self.record_mapping_.estimate(hand_markers, finger_markers)

    def estimate_robot(self, mocap_origin2origin=None):
        """Estimate end-effector pose and joint angles of target system from MANO.

        Parameters
        ----------
        mocap_origin2origin : array, shape (4, 4)
            Transform that will be applied to end-effector pose.

        Returns
        -------
        ee_pose : array, shape (4, 4)
            Pose of the end effector.

        joint_angles : dict
            Maps finger names to corresponding joint angles in the order that
            is given in the target configuration.
        """
        joint_angles = self.embodiment_mapping_.solve(
            self.record_mapping_.mano2world_,
            use_cached_forward_kinematics=True)
        ee2mocap_origin = self.transform_manager_.get_transform(
            self.hand_config_["base_frame"], "world")
        if mocap_origin2origin is not None:
            ee2mocap_origin = pt.concat(ee2mocap_origin, mocap_origin2origin)
        return ee2mocap_origin, joint_angles

    def estimate(self, hand_markers, finger_markers, mocap_origin2origin=None):
        """Estimate state of target system from MoCap markers.

        Parameters
        ----------
        hand_markers : list
            Markers on hand in order 'hand_top', 'hand_left', 'hand_right'.

        finger_markers : dict (str to array-like)
            Positions of markers on fingers.

        mocap_origin2origin : array, shape (4, 4), optional (default: None)
            Transform that will be applied to end-effector pose.

        Returns
        -------
        ee_pose : array, shape (4, 4)
            Pose of the end effector.

        joint_angles : dict
            Maps finger names to corresponding joint angles in the order that
            is given in the target configuration.
        """
        self.estimate_hand(hand_markers, finger_markers)
        return self.estimate_robot(mocap_origin2origin)

    def make_hand_artist(self, show_expected_markers=False):
        """Create artist that visualizes internal state of the hand.

        Parameters
        ----------
        show_expected_markers : bool, optional (default: False)
            Show expected marker positions at hand model.

        Returns
        -------
        artist : ManoHand
            Artist for pytransform3d's visualizer.
        """
        from hand_embodiment.vis_utils import ManoHand
        return ManoHand(
            self.record_mapping_, show_mesh=True, show_vertices=False,
            show_expected_markers=show_expected_markers)

    def make_robot_artist(self):
        """Create artist that visualizes state of the target system.

        Returns
        -------
        graph : pytransform3d.visualizer.Graph
            Representation of the robotic hand.
        """
        from pytransform3d import visualizer as pv
        return pv.Graph(
            self.transform_manager_, "world", show_frames=True,
            whitelist=[self.hand_config_["base_frame"]], show_connections=False,
            show_visuals=True, show_collision_objects=False, show_name=False,
            s=0.02)

    def clear_timings(self):
        """Clear time measurements."""
        self.record_mapping_.clear_timings()
        self.embodiment_mapping_.clear_timings()
