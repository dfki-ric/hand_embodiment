import numpy as np
from hand_embodiment.target_dataset import RoboticHandDataset
from hand_embodiment.target_configurations import MIA_CONFIG
from numpy.testing import assert_array_almost_equal


def test_robotic_hand_dataset():
    finger_names = ["thumb", "index", "middle", "ring", "little"]
    dataset = RoboticHandDataset(finger_names)
    ee_pose = np.eye(4)
    finger_joint_angles = {
        "thumb": np.array([1]),
        "index": np.array([2]),
        "middle": np.array([3]),
        "ring": np.array([4]),
        "little": np.array([5]),
    }
    dataset.append(ee_pose, finger_joint_angles)
    dataset.add_constant_finger_joint("j_thumb_opp", 0)
    assert dataset.n_steps == 1
    df = dataset.export_to_dataframe(MIA_CONFIG)
    assert len(df) == 1
    assert df["j_thumb_opp"].iloc[0] == 0.0
    assert df["j_thumb_fle"].iloc[0] == 1.0
    assert df["j_index_fle"].iloc[0] == 2.0
    assert df["j_mrl_fle"].iloc[0] == 3.0
    assert df["j_ring_fle"].iloc[0] == 4.0
    assert df["j_little_fle"].iloc[0] == 5.0


def test_import_robotic_dataset():
    dataset = RoboticHandDataset.import_from_file(
        "test/data/mia_segment.csv", MIA_CONFIG)
    assert dataset.n_steps == 260
    expected_ee_pose = np.array([
        [-0.945719, -0.308479, 0.10226, 0.057908],
        [-0.299843, 0.949587, 0.091533, -0.382703],
        [-0.125341, 0.055903, -0.990538, 0.040661],
        [0.0, 0.0, 0.0, 1.0]])
    assert_array_almost_equal(dataset.get_ee_pose(55), expected_ee_pose)
    joint_angles = dataset.get_finger_joint_angles(258)
    assert_array_almost_equal(joint_angles["thumb"], 0.424181)
    assert_array_almost_equal(joint_angles["index"], 0.0)
    assert_array_almost_equal(joint_angles["middle"], 0.01723)
    assert_array_almost_equal(joint_angles["ring"], 0.01723)
    assert_array_almost_equal(joint_angles["little"], 0.01723)
