import pytransform3d.visualizer as pv
from hand_embodiment.embodiment import load_kinematic_model
from hand_embodiment.target_configurations import MIA_CONFIG


fig = pv.figure()

kin = load_kinematic_model(MIA_CONFIG)
for finger_name in MIA_CONFIG["ee_frames"].keys():
    finger2base = kin.tm.get_transform(
        MIA_CONFIG["ee_frames"][finger_name], MIA_CONFIG["base_frame"])
    fig.plot_sphere(radius=0.005, A2B=finger2base, c=(1, 0, 0))

graph = pv.Graph(
    kin.tm, MIA_CONFIG["base_frame"], show_frames=False,
    show_connections=False, show_visuals=True, show_collision_objects=False,
    show_name=False, s=0.02)
graph.add_artist(fig)

fig.show()
