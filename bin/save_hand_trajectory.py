"""
Example call:

python bin/save_hand_trajectory.py mia --mia-thumb-adducted --demo-file data/QualisysAprilTest/april_test_009.tsv --output trajectory_009.csv
"""
import argparse
import time
import numpy as np
import tqdm
from hand_embodiment.record_markers import MarkerBasedRecordMapping
from hand_embodiment.embodiment import HandEmbodiment
from hand_embodiment.mocap_dataset import HandMotionCaptureDataset
from hand_embodiment.target_configurations import TARGET_CONFIG
from hand_embodiment.config import load_mano_config
from hand_embodiment.target_dataset import RoboticHandDataset


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "hand", type=str,
        help="Name of the hand. Possible options: mia, shadow_hand")
    parser.add_argument(
        "--demo-file", type=str,
        default="data/QualisysAprilTest/april_test_010.tsv",
        help="Demonstration that should be used.")
    parser.add_argument(
        "--mocap-config", type=str,
        default="examples/config/markers/20210520_april.yaml",
        help="MoCap configuration file.")
    parser.add_argument(
        "--mano-config", type=str,
        default="examples/config/mano/20210520_april.yaml",
        help="MANO configuration file.")
    parser.add_argument(
        "--output", type=str, default="trajectory.csv",
        help="Output file (.csv).")
    parser.add_argument(
        "--start-idx", type=int, default=None, help="Start index.")
    parser.add_argument(
        "--end-idx", type=int, default=None, help="Start index.")
    parser.add_argument(
        "--show-mano", action="store_true", help="Show MANO mesh")
    parser.add_argument(
        "--skip-frames", type=int, default=1,
        help="Skip this number of frames between animated frames.")
    parser.add_argument(
        "--mia-thumb-adducted", action="store_true",
        help="Adduct thumb of Mia hand.")

    return parser.parse_args()


def main():
    args = parse_args()

    dataset = HandMotionCaptureDataset(
        args.demo_file, mocap_config=args.mocap_config,
        skip_frames=args.skip_frames, start_idx=args.start_idx,
        end_idx=args.end_idx)

    hand_config = TARGET_CONFIG[args.hand]

    mano2hand_markers, betas = load_mano_config(args.mano_config)
    mbrm = MarkerBasedRecordMapping(
        left=False, mano2hand_markers=mano2hand_markers,
        shape_parameters=betas, verbose=0)
    emb = HandEmbodiment(
        mbrm.hand_state_, hand_config,
        use_fingers=dataset.finger_names,
        mano_finger_kinematics=mbrm.mano_finger_kinematics_,
        initial_handbase2world=mbrm.mano2world_, verbose=0)
    if args.hand == "mia":
        if args.mia_thumb_adducted:
            emb.target_kin.tm.set_joint("j_thumb_opp_binary", 1.0)
        else:
            emb.target_kin.tm.set_joint("j_thumb_opp_binary", -1.0)

    output_dataset = RoboticHandDataset(finger_names=dataset.finger_names)
    start_time = time.time()
    for t in tqdm.tqdm(range(dataset.n_steps)):
        mbrm.estimate(dataset.get_hand_markers(t),
                      dataset.get_finger_markers(t))
        joint_angles = emb.solve(
            mbrm.mano2world_, use_cached_forward_kinematics=True)
        ee_pose = emb.transform_manager_.get_transform(
            hand_config["base_frame"], "world")
        joint_angles_t = joint_angles.copy()
        if args.hand == "mia":  # TODO refactor
            j_min, j_max = emb.transform_manager_.get_joint_limits("j_thumb_opp")
            if args.mia_thumb_adducted:
                thumb_opp = j_max
            else:
                thumb_opp = j_min
            joint_angles_t["thumb"] = np.hstack((joint_angles_t["thumb"], [thumb_opp]))
        output_dataset.append(ee_pose, joint_angles_t)

    duration = time.time() - start_time
    time_per_frame = duration / dataset.n_steps
    frequency = dataset.n_steps / duration
    print(f"Embodiment mapping done after {duration:.2f} s, "
          f"{time_per_frame:.4f} s per frame, {frequency:.1f} Hz")

    output_dataset.export(args.output, args.hand, hand_config)
    # TODO convert frequency
    print(f"Saved demonstration to '{args.output}'")


if __name__ == "__main__":
    main()
