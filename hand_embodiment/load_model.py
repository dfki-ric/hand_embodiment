from .kinematics import Kinematics


def model_arguments(hand_config):
    return hand_config["model"]


def load_kinematic_model(hand_config):
    model = model_arguments(hand_config)
    with open(model["urdf"], "r") as f:
        kin = Kinematics(urdf=f.read(), package_dir=model["package_dir"])
    if "kinematic_model_hook" in model:
        model["kinematic_model_hook"](kin)
    return kin
