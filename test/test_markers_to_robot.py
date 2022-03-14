import numpy as np
from hand_embodiment.mocap_dataset import HandMotionCaptureDataset
from hand_embodiment.pipelines import MoCapToRobot


def test_markers_to_robot_mia():
    hand = "mia"
    mia_thumb_adducted = True
    interpolate_missing_markers = True
    demo_file = "test/data/recording.tsv"
    mocap_config = "examples/config/markers/20210826_april.yaml"
    mano_config = "examples/config/mano/20210616_april.yaml"
    record_mapping_config = \
        "examples/config/record_mapping/20211105_april.yaml"

    _test_markers_to_robot(
        hand, demo_file, mocap_config, record_mapping_config, mano_config,
        interpolate_missing_markers, mia_thumb_adducted)


def test_markers_to_robot_shadow():
    hand = "shadow"
    interpolate_missing_markers = True
    demo_file = "test/data/recording.tsv"
    mocap_config = "examples/config/markers/20210826_april.yaml"
    mano_config = "examples/config/mano/20210616_april.yaml"
    record_mapping_config = \
        "examples/config/record_mapping/20211105_april.yaml"

    _test_markers_to_robot(
        hand, demo_file, mocap_config, record_mapping_config, mano_config,
        interpolate_missing_markers)


def _test_markers_to_robot(hand, demo_file, mocap_config, record_mapping_config, mano_config,
                           interpolate_missing_markers, mia_thumb_adducted=None):
    dataset = HandMotionCaptureDataset(
        demo_file, mocap_config=mocap_config,
        skip_frames=100, start_idx=100, end_idx=-1,
        interpolate_missing_markers=interpolate_missing_markers)

    pipeline = MoCapToRobot(
        hand, mano_config, dataset.finger_names,
        record_mapping_config=record_mapping_config, verbose=0)

    if mia_thumb_adducted is not None:
        angle = 1.0 if mia_thumb_adducted else -1.0
        pipeline.set_constant_joint("j_thumb_opp_binary", angle)

    for t in range(10):
        hand_markers = dataset.get_hand_markers(t)
        finger_markers = dataset.get_finger_markers(t)
        ee_pose, joint_angles = pipeline.estimate(
            hand_markers, finger_markers)
        assert not np.any(np.isnan(ee_pose))
        for finger in joint_angles:
            assert not np.any(np.isnan(joint_angles[finger]))
