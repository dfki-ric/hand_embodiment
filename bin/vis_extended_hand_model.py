import argparse
import pytransform3d.visualizer as pv
from hand_embodiment.embodiment import load_kinematic_model
from hand_embodiment.target_configurations import MIA_CONFIG, SHADOW_HAND_CONFIG


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "hand", type=str,
        help="Hand for which we show the extended model. Possible "
             "options: 'mia', 'shadow_hand'")
    args = parser.parse_args()

    fig = pv.figure()

    if args.hand == "shadow_hand":
        hand_config = SHADOW_HAND_CONFIG
    elif args.hand == "mia":
        hand_config = MIA_CONFIG
    else:
        raise Exception(f"Unknown hand: '{args.hand}'")

    kin = load_kinematic_model(hand_config)
    for finger_name in hand_config["ee_frames"].keys():
        finger2base = kin.tm.get_transform(
            hand_config["ee_frames"][finger_name], hand_config["base_frame"])
        fig.plot_sphere(radius=0.005, A2B=finger2base, c=(1, 0, 0))
    if "intermediate_frames" in hand_config:
        for finger_name in hand_config["intermediate_frames"].keys():
            finger2base = kin.tm.get_transform(
                hand_config["intermediate_frames"][finger_name],
                hand_config["base_frame"])
            fig.plot_sphere(radius=0.005, A2B=finger2base, c=(1, 0, 0))

    graph = pv.Graph(
        kin.tm, hand_config["base_frame"], show_frames=False,
        show_connections=False, show_visuals=True, show_collision_objects=False,
        show_name=False, s=0.02)
    graph.add_artist(fig)

    fig.show()


if __name__ == "__main__":
    main()
