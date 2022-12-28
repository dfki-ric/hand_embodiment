"""Live visualization of data streamed from MoCap system."""
import json
import numpy as np
import pytransform3d.visualizer as pv
from hand_embodiment.record_markers import MarkerBasedRecordMapping
from hand_embodiment.embodiment import HandEmbodiment
from hand_embodiment.config import load_mano_config
from hand_embodiment.target_configurations import MIA_CONFIG, SHADOW_HAND_CONFIG
from hand_embodiment.vis_utils import ManoHand


def animation_callback(step, markers, mbrm, hand, emb, robot):
    try:
        with open("comm.json", "r") as f:
            result = json.load(f)
    except:
        return []

    marker_pos = np.empty((len(result), 3))
    for i, marker_name in enumerate(result):
        marker_pos[i] = result[marker_name]

    # Filter positions
    nan = np.isnan(marker_pos)
    valid = not np.any(nan)
    marker_pos[nan] = 0.0

    markers.set_data(marker_pos)

    if not valid:
        return markers

    hand_markers = [np.array(result["hand_top"]),
                    np.array(result["hand_left"]),
                    np.array(result["hand_right"])]
    finger_markers = {
        "thumb": np.array(result["thumb_tip"]),
        "index": np.array(result["index_tip"]),
        "middle": np.array(result["middle_tip"]),
        "ring": np.array(result["ring_tip"]),
        "little": np.array(result["little_tip"])
    }
    mbrm.estimate(hand_markers, finger_markers)

    emb.solve(mbrm.mano2world_, use_cached_forward_kinematics=True)

    artists = [markers]
    if show_mano:
        hand.set_data()
        artists.append(hand)
    if show_robot:
        robot.set_data()
        artists.append(robot)
    return artists


hand = "mia"
show_mano = True
show_robot = True


if hand == "shadow_hand":
    hand_config = SHADOW_HAND_CONFIG
elif hand == "mia":
    hand_config = MIA_CONFIG
else:
    raise Exception(f"Unknown hand: '{hand}'")


#mano2hand_markers, betas = load_mano_config(
#    "examples/config/april_test_mano.yaml")
mano2hand_markers, betas = load_mano_config(
    "examples/config/mano/20220815_april_trha.yaml")
mbrm = MarkerBasedRecordMapping(
    left=False, mano2hand_markers=mano2hand_markers,
    shape_parameters=betas, verbose=1)
emb = HandEmbodiment(
    mbrm.hand_state_, hand_config,
    use_fingers=("thumb", "index", "middle", "ring", "little"),
    mano_finger_kinematics=mbrm.mano_finger_kinematics_,
    initial_handbase2world=mbrm.mano2world_, verbose=1)
hand = ManoHand(mbrm, show_mesh=True, show_vertices=False)
robot = pv.Graph(
    emb.target_kin.tm, "world", show_frames=True, whitelist=[hand_config["base_frame"]],
    show_connections=False, show_visuals=True, show_collision_objects=False,
    show_name=False, s=0.02)

fig = pv.figure()
fig.plot_transform(s=0.5)
marker_color = [
[0, 0, 1],
[0, 0, 0],
[0, 0, 0],
[0, 0, 0],
[0, 0, 0],
[0, 0, 0],
[0, 0, 0],
[0, 0, 0],
[0, 0, 0],
[0, 0, 0],
[0, 0, 0],
[0, 0, 0],
[0, 0, 0],
]

markers = fig.scatter(np.zeros((13, 3)), s=0.006, c=marker_color)
if show_mano:
    hand.add_artist(fig)
if show_robot:
    robot.add_artist(fig)
fig.view_init(azim=-70)
fig.animate(
    animation_callback, 1, loop=True,
    fargs=(markers, mbrm, hand, emb, robot))
fig.show()
