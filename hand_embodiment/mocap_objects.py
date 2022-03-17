import numpy as np
from .vis_utils import (electronic_target_pose, electronic_object_pose,
                        pillow_pose, insole_pose)


class ElectronicTargetMarkers:
    default_marker_positions = {
        "target_top": np.array([0, 0, 0]),
        "target_bottom": np.array([1, 0, 0])
    }
    marker_names = tuple(default_marker_positions.keys())

    @staticmethod
    def pose_from_markers(target_top, target_bottom):
        return electronic_target_pose(target_top, target_bottom)


class ElectronicObjectMarkers:
    default_marker_positions = {
        "object_left": np.zeros(3),
        "object_right": np.array([1, 0, 0]),
        "object_top": np.array([1, 1, 0])
    }
    marker_names = tuple(default_marker_positions.keys())

    @staticmethod
    def pose_from_markers(object_left, object_right, object_top):
        return electronic_object_pose(object_left, object_right, object_top)


class PillowMarkers:
    default_marker_positions = {
        "pillow_left": np.array([-0.11, 0.13, 0]),
        "pillow_right": np.array([-0.11, -0.13, 0]),
        "pillow_top": np.array([0.11, -0.13, 0])
    }
    marker_names = tuple(default_marker_positions.keys())

    @staticmethod
    def pose_from_markers(pillow_left, pillow_right, pillow_top):
        return pillow_pose(pillow_left, pillow_right, pillow_top)


class InsoleMarkers:
    default_marker_positions = {
        "insole_back": np.zeros(3),
        "insole_front": np.array([0.19, 0, 0])
    }
    marker_names = tuple(default_marker_positions.keys())

    @staticmethod
    def pose_from_markers(insole_back, insole_front):
        return insole_pose(insole_back, insole_front)
