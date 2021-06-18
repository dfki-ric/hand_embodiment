from hand_embodiment.target_configurations import TARGET_CONFIG
from hand_embodiment.config import load_mano_config
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

    verbose : int, optional (default: 0)
        Verbosity level
    """
    def __init__(self, hand, mano_config, use_fingers, verbose=0):
        self.hand_config_ = TARGET_CONFIG[hand]
        mano2hand_markers, betas = load_mano_config(mano_config)
        self.record_mapping_ = MarkerBasedRecordMapping(
            left=False, mano2hand_markers=mano2hand_markers,
            shape_parameters=betas, verbose=verbose)
        self.embodiment_mapping_ = HandEmbodiment(
            self.record_mapping_.hand_state_, self.hand_config_,
            use_fingers=use_fingers,
            mano_finger_kinematics=self.record_mapping_.mano_finger_kinematics_,
            initial_handbase2world=self.record_mapping_.mano2world_,
            verbose=verbose)

    @property
    def transform_manager_(self):
        """Graph that represents state of robotic hand."""
        return self.embodiment_mapping_.transform_manager_

    def reset(self):
        """Reset record mapping."""
        self.record_mapping_.reset()

    def set_constant_joint(self, joint_name, angle):
        self.transform_manager_.set_joint(joint_name, angle)

    def estimate_hand(self, hand_markers, finger_markers):
        """Estimate MANO pose and joint angles."""
        self.record_mapping_.estimate(hand_markers, finger_markers)

    def estimate_robot(self):
        """Estimate end-effector pose and joint angles of target system from MANO."""
        joint_angles = self.embodiment_mapping_.solve(
            self.record_mapping_.mano2world_,
            use_cached_forward_kinematics=True)
        ee_pose = self.transform_manager_.get_transform(
            self.hand_config_["base_frame"], "world")
        return ee_pose, joint_angles

    def estimate(self, hand_markers, finger_markers):
        """Estimate state of target system from MoCap markers."""
        self.estimate_hand(hand_markers, finger_markers)
        return self.estimate_robot()

    def make_hand_artist(self):
        """Create artist that visualizes internal state of the hand."""
        from hand_embodiment.vis_utils import ManoHand
        return ManoHand(
            self.embodiment_mapping_, show_mesh=True, show_vertices=False)

    def make_robot_artist(self):
        """Create artist that visualizes state of the target system."""
        from pytransform3d import visualizer as pv
        return pv.Graph(
            self.transform_manager_, "world", show_frames=True,
            whitelist=[self.hand_config_["base_frame"]], show_connections=False,
            show_visuals=True, show_collision_objects=False, show_name=False,
            s=0.02)
