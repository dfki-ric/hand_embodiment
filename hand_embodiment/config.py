"""Interface to MANO and record mapping configuration."""
import numpy as np
import yaml
import pytransform3d.transformations as pt


def load_mano_config(filename):
    """Load MANO configuration for record mapping.

    Parameters
    ----------
    filename : str
        Configuration file

    Returns
    -------
    mano2hand_markers : array, shape (4, 4)
        Transformation from MANO base to hand marker frame

    betas : array, shape (10,)
        Shape parameters of MANO mesh
    """
    with open(filename, "r") as f:
        config = yaml.safe_load(f)
    mano2hand_markers = pt.transform_from_exponential_coordinates(
        config["mano2hand_markers"])
    betas = np.array(config["betas"])
    return mano2hand_markers, betas


def save_mano_config(filename, mano2hand_markers, betas):
    """Save MANO configuration for record mapping.

    Parameters
    ----------
    filename : str
        Configuration file

    mano2hand_markers : array, shape (4, 4)
        Transformation from MANO base to hand marker frame

    betas : array, shape (10,)
        Shape parameters of MANO mesh
    """
    mano2hand_markers = pt.exponential_coordinates_from_transform(mano2hand_markers)
    config = {
        "mano2hand_markers": mano2hand_markers.tolist(),
        "betas": betas.tolist()
    }
    with open(filename, "w") as f:
        yaml.dump(config, f)


def load_record_mapping_config(filename):
    """Load record mapping configuration.

    Parameters
    ----------
    filename : str
        Configuration file

    Returns
    -------
    record_mapping_config : dict
        Configuration
    """
    with open(filename, "r") as f:
        record_mapping_config = yaml.safe_load(f)
    for finger_name in record_mapping_config["action_weights_per_finger"]:
        record_mapping_config["action_weights_per_finger"][finger_name] = \
            np.asarray(record_mapping_config["action_weights_per_finger"][finger_name])
    for finger_name in record_mapping_config["pose_parameters_per_finger"]:
        record_mapping_config["pose_parameters_per_finger"][finger_name] = \
            np.asarray(record_mapping_config["pose_parameters_per_finger"][finger_name])
    for finger_name in record_mapping_config["tip_vertex_offsets_per_finger"]:
        record_mapping_config["tip_vertex_offsets_per_finger"][finger_name] = \
            np.asarray(record_mapping_config["tip_vertex_offsets_per_finger"][finger_name])
    return record_mapping_config
