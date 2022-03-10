import json
import pickle
import numpy as np


def prepare_mano_json(input_filename, output_filename):
    """Convert a MANO model to JSON."""
    with open(input_filename, "rb") as f:
        data = pickle.load(f, encoding="latin1")

    output = {}
    for key in ["hands_components", "f", "kintree_table", "J", "hands_coeffs",
                "weights", "posedirs", "hands_mean", "v_template"]:
        output[key] = data[key].tolist()

    for key in ["bs_style", "bs_type"]:
        output[key] = data[key]

    output["shapedirs"] = np.asarray(data["shapedirs"]).tolist()

    output["J_regressor"] = {
        "data": data["J_regressor"].data.tolist(),
        "indices": data["J_regressor"].indices.tolist(),
        "indptr": data["J_regressor"].indptr.tolist()
    }

    with open(output_filename, "w") as f:
        json.dump(output, f)


if __name__ == '__main__':
    MANO_CONF_ROOT_DIR = "./"

    # Convert 'HAND_MESH_MODEL_PATH' to 'HAND_MESH_MODEL_PATH_JSON' with 'prepare_mano.py'
    HAND_MESH_MODEL_LEFT_PATH_JSON = MANO_CONF_ROOT_DIR + "mano_left.json"
    HAND_MESH_MODEL_RIGHT_PATH_JSON = MANO_CONF_ROOT_DIR + "mano_right.json"

    OFFICIAL_MANO_LEFT_PATH = MANO_CONF_ROOT_DIR + "MANO_LEFT.pkl"
    OFFICIAL_MANO_RIGHT_PATH = MANO_CONF_ROOT_DIR + "MANO_RIGHT.pkl"

    prepare_mano_json(OFFICIAL_MANO_LEFT_PATH, HAND_MESH_MODEL_LEFT_PATH_JSON)
    prepare_mano_json(OFFICIAL_MANO_RIGHT_PATH, HAND_MESH_MODEL_RIGHT_PATH_JSON)
