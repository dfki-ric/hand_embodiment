import pytransform3d.visualizer as pv
from hand_embodiment.embodiment import load_kinematic_model
from hand_embodiment.target_configurations import MIA_CONFIG, SHADOW_HAND_CONFIG


fig = pv.figure()

hand = "mia"
if hand == "shadow_hand":
    hand_config = SHADOW_HAND_CONFIG
elif hand == "mia":
    hand_config = MIA_CONFIG
else:
    raise Exception(f"Unknown hand: '{hand}'")

kin = load_kinematic_model(hand_config)
for finger_name in hand_config["ee_frames"].keys():
    finger2base = kin.tm.get_transform(
        hand_config["ee_frames"][finger_name], hand_config["base_frame"])
    fig.plot_sphere(radius=0.005, A2B=finger2base, c=(1, 0, 0))

graph = pv.Graph(
    kin.tm, hand_config["base_frame"], show_frames=False,
    show_connections=False, show_visuals=True, show_collision_objects=False,
    show_name=False, s=0.02)
graph.add_artist(fig)

fig.show()
