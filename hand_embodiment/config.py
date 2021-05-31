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
        config = yaml.load(f)
    mano2hand_markers = pt.transform_from_exponential_coordinates(
        config["mano2hand_markers"])
    betas = np.array(config["betas"])
    return mano2hand_markers, betas
