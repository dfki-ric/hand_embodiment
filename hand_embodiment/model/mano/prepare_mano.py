import pickle

import numpy as np
import json


class MANOHandJoints:
    labels = [
        'W', #0
        'I0', 'I1', 'I2', #3
        'M0', 'M1', 'M2', #6
        'L0', 'L1', 'L2', #9
        'R0', 'R1', 'R2', #12
        'T0', 'T1', 'T2', #15
        'I3', 'M3', 'L3', 'R3', 'T3' #20, tips are manually added (not in MANO)
    ]

    # finger tips are not joints in MANO, we label them on the mesh manually
    mesh_mapping = {16: 333, 17: 444, 18: 672, 19: 555, 20: 744}

    parents = [
        None,
        0, 1, 2,
        0, 4, 5,
        0, 7, 8,
        0, 10, 11,
        0, 13, 14,
        3, 6, 9, 12, 15
    ]

MANO_CONF_ROOT_DIR = './'

# Convert 'HAND_MESH_MODEL_PATH' to 'HAND_MESH_MODEL_PATH_JSON' with 'prepare_mano.py'
HAND_MESH_MODEL_LEFT_PATH_JSON = MANO_CONF_ROOT_DIR + 'mano_left.json'
HAND_MESH_MODEL_RIGHT_PATH_JSON = MANO_CONF_ROOT_DIR + 'mano_right.json'

OFFICIAL_MANO_LEFT_PATH = MANO_CONF_ROOT_DIR + 'MANO_LEFT.pkl'
OFFICIAL_MANO_RIGHT_PATH = MANO_CONF_ROOT_DIR + 'MANO_RIGHT.pkl'


def prepare_mano_json(left=True):
    """Use this function to convert a mano_handstate model (from MANO-Hand Project) to the hand model we want to use in the project.
    """

    if left:
        with open(OFFICIAL_MANO_LEFT_PATH, 'rb') as f:
            data = pickle.load(f, encoding='latin1')
    else:
        with open(OFFICIAL_MANO_RIGHT_PATH, 'rb') as f:
            data = pickle.load(f, encoding='latin1')

    output = {}
    output['verts'] = data['v_template']
    output['faces'] = data.pop("f")
    output['mesh_basis'] = np.transpose(data["shapedirs"], (2, 0, 1))

    j_regressor = np.zeros([21, 778])
    j_regressor[:16] = data["J_regressor"].toarray()
    for k, v in MANOHandJoints.mesh_mapping.items():
        j_regressor[k, v] = 1
    output['j_regressor'] = j_regressor
    output['joints'] = np.matmul(output['j_regressor'], output['verts'])

    raw_weights = data["weights"]
    weights = [None] * 21
    weights[0] = raw_weights[:, 0]
    for j in 'IMLRT':
        weights[MANOHandJoints.labels.index(j + '0')] = np.zeros(778)
        for k in [1, 2, 3]:
            src_idx = MANOHandJoints.labels.index(j + str(k - 1))
            tar_idx = MANOHandJoints.labels.index(j + str(k))
            weights[tar_idx] = raw_weights[:, src_idx]
    output['weights'] = np.expand_dims(np.stack(weights, -1), -1)

    # save in json_file
    if left:
        with open(HAND_MESH_MODEL_LEFT_PATH_JSON, 'w') as f:
            mano_data_string = convert_to_plain(output)
            json.dump(mano_data_string, f)
    else:
        with open(HAND_MESH_MODEL_RIGHT_PATH_JSON, 'w') as f:
            mano_data_string = convert_to_plain(output)
            json.dump(mano_data_string, f)


def convert_to_plain(hand):
    plain = {}
    for k in ["verts", "faces", "weights", "joints", "mesh_basis", "j_regressor"]:
        plain[k] = np.array(hand[k]).tolist()
    return plain


if __name__ == '__main__':
    prepare_mano_json(left=False)
    prepare_mano_json(left=True)
