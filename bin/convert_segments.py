"""Convert MoCap segments to a robotic hand: record and embodiment mapping.

Example call:

python bin/convert_segments.py mia close --mia-thumb-adducted --mocap-config examples/config/markers/20210616_april.yaml --demo-file data/20210616_april/metadata/Measurement16.json --output dataset_16_segment_%d.csv
python bin/convert_segments.py mia close --mia-thumb-adducted --mocap-config examples/config/markers/20210616_april.yaml --demo-file data/20210616_april/metadata/Measurement23.json --output dataset_23_segment_%d.csv
python bin/convert_segments.py mia close --mia-thumb-adducted --mocap-config examples/config/markers/20210616_april.yaml --demo-file data/20210616_april/metadata/Measurement24.json --output dataset_24_segment_%d.csv
python bin/convert_segments.py mia close --mia-thumb-adducted --mocap-config examples/config/markers/20210616_april.yaml --demo-file data/20210701_april/Measurement30.json --output dataset_30_segment_%d.csv --insole-hack
python bin/convert_segments.py mia close --mia-thumb-adducted --mocap-config examples/config/markers/20210819_april.yaml --demo-file data/20210819_april/20210819_r_WK37_insole_set0.json --output 20210819_r_WK37_insole_set0_%d.csv --insole-hack
python bin/convert_segments.py mia close --mia-thumb-adducted --mocap-config examples/config/markers/20210826_april.yaml --demo-file data/20210826_april/20210826_r_WK37_small_pillow_set0.json --output 20210826_r_WK37_small_pillow_set0_%d.csv --pillow-hack --measure-time

# grasp insole
python bin/convert_segments.py mia close --mia-thumb-adducted --mocap-config examples/config/markers/20210819_april.yaml --demo-file data/20210819_april/2021*_r_WK37_insole_set*.json --output 2021_r_WK37_insole_%d.csv --insole-hack --measure-time
python bin/convert_segments.py shadow close --mocap-config examples/config/markers/20210819_april.yaml --mano-config examples/config/mano/20211105_april.yaml --demo-file data/20210819_april/2021*_r_WK37_insole_set*.json --output 2021_r_WK37_insole_%d.csv --insole-hack --measure-time
# grasp small pillow
python bin/convert_segments.py mia close --mia-thumb-adducted --mocap-config examples/config/markers/20210826_april.yaml --demo-file data/20210826_april/2021*_r_WK37_small_pillow_set*.json --output 2021_r_WK37_small_pillow_%d.csv --pillow-hack --measure-time
python bin/convert_segments.py shadow close --mocap-config examples/config/markers/20210826_april.yaml --mano-config examples/config/mano/20211105_april.yaml --demo-file data/20210826_april/2021*_r_WK37_small_pillow_set*.json --output 2021_r_WK37_small_pillow_%d.csv --pillow-hack --measure-time
# grasp electronic component
python bin/convert_segments.py mia grasp --mia-thumb-adducted --mocap-config examples/config/markers/20211105_april.yaml --demo-file data/20211105_april/2021*_r_WK37_electronic_set*.json --output 2021_r_WK37_electronic_%d.csv --electronic-object-hack --measure-time --interpolate-missing-markers
python bin/convert_segments.py shadow grasp --mocap-config examples/config/markers/20211105_april.yaml --demo-file data/20211105_april/2021*_r_WK37_electronic_set*.json --output 2021_r_WK37_electronic_%d.csv --electronic-object-hack --measure-time --interpolate-missing-markers
# assembly of electronic components
python bin/convert_segments.py mia insert --mia-thumb-adducted --mocap-config examples/config/markers/20211105_april.yaml --demo-file data/20211105_april/2021*_r_WK37_electronic_set*.json --output 2021_r_WK37_electronic_insert_%d.csv --electronic-target-hack --measure-time --interpolate-missing-markers
python bin/convert_segments.py shadow insert --mocap-config examples/config/markers/20211105_april.yaml --demo-file data/20211105_april/2021*_r_WK37_electronic_set*.json --output 2021_r_WK37_electronic_insert_%d.csv --electronic-target-hack --measure-time --interpolate-missing-markers
# flip pages of a passport
python bin/convert_segments.py mia flip --mia-thumb-adducted --mocap-config examples/config/markers/20211112_april.yaml --demo-file data/20211112_april/20211112_r_WK37_passport_set*.json --output 2021_r_WK37_flip_passport_%d.csv --passport-hack --measure-time --interpolate-missing-markers
python bin/convert_segments.py shadow flip --mocap-config examples/config/markers/20211112_april.yaml --demo-file data/20211112_april/20211112_r_WK37_passport_set*.json --output 2021_r_WK37_flip_passport_%d.csv --passport-hack --measure-time --interpolate-missing-markers
# grasp big pillow
python bin/convert_segments.py mia grasp --mia-thumb-adducted --mocap-config examples/config/markers/20211126_april_pillow.yaml --demo-file data/20211126_april_pillow/20211126_r_WK37_big_pillow_set*.json --output 2021_r_WK37_big_pillow_%d.csv --measure-time --interpolate-missing-markers
python bin/convert_segments.py shadow grasp --mocap-config examples/config/markers/20211126_april_pillow.yaml --demo-file data/20211126_april_pillow/20211126_r_WK37_big_pillow_set*.json --output 2021_r_WK37_big_pillow_%d.csv --measure-time --interpolate-missing-markers
# insert insole
python bin/convert_segments.py mia insert --mocap-config examples/config/markers/20211126_april_insole.yaml --demo-file data/20211126_april_insole/20211126_r_WK37_insert_insole_set*.json --output 2021_r_WK37_insert_insole_%d.csv --measure-time --interpolate-missing-markers
python bin/convert_segments.py shadow insert --mocap-config examples/config/markers/20211126_april_insole.yaml --demo-file data/20211126_april_insole/20211126_r_WK37_insert_insole_set*.json --output 2021_r_WK37_insert_insole_%d.csv --measure-time --interpolate-missing-markers
# insert passport in a box
python bin/convert_segments.py mia insert --mia-thumb-adducted --mocap-config examples/config/markers/20211217_april.yaml --demo-file data/20211217_april/20211217_r_WK37_passport_box_set*.json --output 2021_r_WK37_insert_passport_%d.csv --passport-box-hack --measure-time --interpolate-missing-markers
python bin/convert_segments.py shadow insert --mocap-config examples/config/markers/20211217_april.yaml --demo-file data/20211217_april/20211217_r_WK37_passport_box_set*.json --output 2021_r_WK37_insert_passport_%d.csv --passport-box-hack --measure-time --interpolate-missing-markers

# insole dataset with labels: grasp point (front, middle, back) and grasp type (spherical, pinch, lateral)
python bin/convert_segments.py mia grasp_spherical_insole_middle --demo-files data/20210819_april/*.json data/20211119_april/*.json data/20220328_april/*.json --label-field l2 --mocap-config examples/config/markers/20211119_april.yaml --output 2022_r_WK37_insole_spherical_middle_%d.csv --insole-hack --measure-time
python bin/convert_segments.py mia grasp_spherical_insole_front --demo-files data/20210819_april/*.json data/20211119_april/*.json data/20220328_april/*.json --label-field l2 --mocap-config examples/config/markers/20210819_april.yaml --output 2022_r_WK37_insole_spherical_front_%d.csv --insole-hack --measure-time
python bin/convert_segments.py mia grasp_spherical_insole_back --demo-files data/20210819_april/*.json data/20211119_april/*.json data/20220328_april/*.json --label-field l2 --mocap-config examples/config/markers/20210819_april.yaml --output 2022_r_WK37_insole_spherical_back_%d.csv --insole-hack --measure-time
python bin/convert_segments.py mia grasp_pinch_insole_back --demo-files data/20210819_april/*.json data/20211119_april/*.json data/20220328_april/*.json --label-field l2 --mia-thumb-adducted --mocap-config examples/config/markers/20220328_april.yaml --output 2022_r_WK37_insole_pinch_back_%d.csv --insole-hack --measure-time
python bin/convert_segments.py mia grasp_pinch_insole_front --demo-files data/20210819_april/*.json data/20211119_april/*.json data/20220328_april/*.json --label-field l2 --mia-thumb-adducted --mocap-config examples/config/markers/20220328_april.yaml --output 2022_r_WK37_insole_pinch_front_%d.csv --insole-hack --measure-time
python bin/convert_segments.py mia grasp_lateral_insole_back --demo-files data/20210819_april/*.json data/20211119_april/*.json data/20220328_april/*.json --label-field l2 --mocap-config examples/config/markers/20220328_april.yaml --output 2022_r_WK37_insole_lateral_back_%d.csv --insole-hack --measure-time
python bin/convert_segments.py mia grasp_lateral_insole_front --demo-files data/20210819_april/*.json data/20211119_april/*.json data/20220328_april/*.json --label-field l2 --mocap-config examples/config/markers/20220328_april.yaml --output 2022_r_WK37_insole_lateral_front_%d.csv --insole-hack --measure-time
"""
import argparse
import warnings

from hand_embodiment.mocap_dataset import SegmentedHandMotionCaptureDataset
from hand_embodiment.pipelines import MoCapToRobot
from hand_embodiment.target_dataset import convert_mocap_to_robot
from hand_embodiment.timing import timing_report
from hand_embodiment.command_line import (
    add_hand_argument, add_configuration_arguments)
from hand_embodiment.mocap_objects import (
    InsoleMarkers, PillowMarkers, ElectronicTargetMarkers, ElectronicObjectMarkers,
    PassportMarkers, PassportClosedMarkers, PassportBoxMarkers,
    extract_mocap_origin2object)


def parse_args():
    parser = argparse.ArgumentParser()
    add_hand_argument(parser)
    parser.add_argument(
        "segment_label", type=str,
        help="Label of the segment that should be used.")
    parser.add_argument(
        "--demo-files", type=str, nargs="*",
        default=["data/20210616_april/metadata/Measurement24.json"],
        help="Demonstrations that should be used.")
    add_configuration_arguments(parser)
    parser.add_argument(
        "--label-field", type=str, default="label 1",
        help="Name of the label field in metadata file.")
    parser.add_argument(
        "--output", type=str, default="segment_%02d.csv",
        help="Output file pattern (.csv).")
    parser.add_argument(
        "--show-mano", action="store_true", help="Show MANO mesh")
    parser.add_argument(
        "--skip-frames", type=int, default=1,
        help="Skip this number of frames between animated frames.")
    parser.add_argument(
        "--interpolate-missing-markers", action="store_true",
        help="Interpolate NaNs.")
    parser.add_argument(
        "--mia-thumb-adducted", action="store_true",
        help="Adduct thumb of Mia hand.")
    parser.add_argument(
        "--measure-time", action="store_true",
        help="Measure time of record and embodiment mapping.")
    parser.add_argument(
        "--insole-hack", action="store_true",
        help="Insole-relative end-effector coordinates.")
    parser.add_argument(
        "--pillow-hack", action="store_true",
        help="Pillow-relative end-effector coordinates.")
    parser.add_argument(
        "--electronic-object-hack", action="store_true",
        help="Electronic-object-relative end-effector coordinates.")
    parser.add_argument(
        "--electronic-target-hack", action="store_true",
        help="Electronic-target-relative end-effector coordinates.")
    parser.add_argument(
        "--passport-hack", action="store_true",
        help="Passport-relative end-effector coordinates.")
    parser.add_argument(
        "--passport-closed-hack", action="store_true",
        help="Passport-relative end-effector coordinates.")
    parser.add_argument(
        "--passport-box-hack", action="store_true",
        help="Passport-box-relative end-effector coordinates.")

    return parser.parse_args()


def main():
    args = parse_args()

    dataset = SegmentedHandMotionCaptureDataset(
        args.demo_files[0], args.segment_label, mocap_config=args.mocap_config,
        label_field=args.label_field)
    pipeline = MoCapToRobot(args.hand, args.mano_config, dataset.finger_names,
                            record_mapping_config=args.record_mapping_config,
                            measure_time=args.measure_time)

    total_segment_idx = 0
    for demo_file in args.demo_files:
        dataset = SegmentedHandMotionCaptureDataset(
            demo_file, args.segment_label, mocap_config=args.mocap_config,
            interpolate_missing_markers=args.interpolate_missing_markers,
            label_field=args.label_field)
        if dataset.n_segments == 0:
            continue

        if args.hand == "mia":
            angle = 1.0 if args.mia_thumb_adducted else -1.0
            pipeline.set_constant_joint("j_thumb_opp_binary", angle)

        for i in range(dataset.n_segments):
            dataset.select_segment(i)

            if args.insole_hack:
                mocap_origin2origin = extract_mocap_origin2object(dataset, InsoleMarkers)
            elif args.pillow_hack:
                mocap_origin2origin = extract_mocap_origin2object(dataset, PillowMarkers)
            elif args.electronic_object_hack:
                mocap_origin2origin = extract_mocap_origin2object(dataset, ElectronicObjectMarkers)
            elif args.electronic_target_hack:
                mocap_origin2origin = extract_mocap_origin2object(dataset, ElectronicTargetMarkers)
            elif args.passport_hack:
                mocap_origin2origin = extract_mocap_origin2object(dataset, PassportMarkers)
            elif args.passport_closed_hack:
                mocap_origin2origin = extract_mocap_origin2object(dataset, PassportClosedMarkers)
            elif args.passport_box_hack:
                mocap_origin2origin = extract_mocap_origin2object(dataset, PassportBoxMarkers)
            else:
                mocap_origin2origin = None

            output_dataset = convert_mocap_to_robot(
                dataset, pipeline, mocap_origin2origin=mocap_origin2origin,
                verbose=1)

            if args.hand == "mia":
                j_min, j_max = pipeline.transform_manager_.get_joint_limits("j_thumb_opp")
                thumb_opp = j_max if args.mia_thumb_adducted else j_min
                output_dataset.add_constant_finger_joint("j_thumb_opp", thumb_opp)

            output_filename = args.output % total_segment_idx
            output_dataset.export(output_filename, pipeline.hand_config_)
            # TODO convert frequency
            print(f"Saved demonstration to '{output_filename}'")
            total_segment_idx += 1

    if args.measure_time:
        timing_report(pipeline.record_mapping_, title="record mapping")
        timing_report(pipeline.embodiment_mapping_, title="embodiment mapping")


if __name__ == "__main__":
    main()
