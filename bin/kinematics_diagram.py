"""Create kinematic diagram from a URDF."""
import argparse
from hand_embodiment.tools.graphviz_urdf import write_png
from hand_embodiment.embodiment import load_kinematic_model
from hand_embodiment.target_configurations import TARGET_CONFIG
from hand_embodiment.command_line import add_hand_argument


def main():
    parser = argparse.ArgumentParser()
    add_hand_argument(parser)
    parser.add_argument(
        "--show-visuals", action="store_true",
        help="Show visual geometries.")
    parser.add_argument(
        "--show-collision-objects", action="store_true",
        help="Show collision objects.")
    parser.add_argument(
        "--show-inertial-frames", action="store_true",
        help="Show collision objects.")
    parser.add_argument(
        "--show-matrix", action="store_true",
        help="Show transformation matrices.")

    args = parser.parse_args()

    hand_config = TARGET_CONFIG[args.hand]

    kin = load_kinematic_model(hand_config, unscaled_visual_model=False)[0]

    filename = f"{args.hand}.png"
    write_png(
        kin.tm, filename, prog="fdp", show_visuals=args.show_visuals,
        show_collision_objects=args.show_collision_objects,
        show_inertial_frames=args.show_inertial_frames,
        show_matrix=args.show_matrix)
    print(f"Result has been written to {filename}")


if __name__ == "__main__":
    main()
