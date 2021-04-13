from pkg_resources import resource_filename
from .kinematics import Kinematics


def model_arguments(hand):
    if hand == "mia":
        urdf = resource_filename(
            "hand_embodiment", "model/mia_hand_description/urdf/mia_hand.urdf")
        package_dir = resource_filename("hand_embodiment", "model/")
    else:
        raise ValueError("Hand unknown: '{hand}'")
    return {"urdf": urdf, "package_dir": package_dir}


def load_kinematic_model(hand):
    args = model_arguments(hand)
    with open(args["urdf"], "r") as f:
        kin = Kinematics(urdf=f.read(), package_dir=args["package_dir"])
    return kin
