import numpy as np
from hand_embodiment.target_dataset import RoboticHandDataset
from hand_embodiment.target_configurations import MIA_CONFIG


def test_robotic_hand_dataset():
    finger_names = ["thumb", "index", "middle", "ring", "little"]
    dataset = RoboticHandDataset(finger_names)
    ee_pose = np.array([0, 0, 0, 1, 0, 0, 0])
    finger_joint_angles = {
        "thumb": [0],
        "index": [0],
        "middle": [0],
        "ring": [0],
        "little": [0],
    }
    dataset.append(ee_pose, finger_joint_angles)
    dataset.add_constant_finger_joint("j_thumb_opp", 0)
    assert dataset.n_steps == 1
    # TODO test export (without file)
    # TODO test import
