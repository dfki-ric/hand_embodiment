import argparse
from hand_embodiment.tools.graphviz_urdf import write_png
from hand_embodiment.embodiment import load_kinematic_model
from hand_embodiment.target_configurations import MIA_CONFIG


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "hand", type=str,
        help="Hand for which we plot the kinematics diagram. Possible "
             "options: 'mia'")
    parser.add_argument(
        "--show-visuals", action="store_true",
        help="Show visual geometries.")
    parser.add_argument(
        "--show-collision-objects", action="store_true",
        help="Show collision objects.")
    parser.add_argument(
        "--show-matrix", action="store_true",
        help="Show transformation matrices.")

    args = parser.parse_args()

    if args.hand == "mia":
        hand_config = MIA_CONFIG
    else:
        raise Exception(f"Unknown hand: '{args.hand}'")

    kin = load_kinematic_model(hand_config)

    filename = f"{args.hand}.png"
    write_png(
        kin.tm, filename, prog="fdp", show_visuals=args.show_visuals,
        show_collision_objects=args.show_collision_objects,
        show_matrix=args.show_matrix)
    print(f"Result has been written to {filename}")


if __name__ == "__main__":
    main()
