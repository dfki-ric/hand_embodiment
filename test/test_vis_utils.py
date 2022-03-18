import numpy as np
from hand_embodiment.vis_utils import (
    Insole, PillowSmall, Passport, PassportClosed, PassportBox)
from numpy.testing import assert_array_almost_equal


def test_default_insole_pose():
    artist = Insole()
    assert_array_almost_equal(artist.markers2origin, np.eye(4))


def test_default_pillow_pose():
    artist = PillowSmall()
    assert_array_almost_equal(artist.markers2origin, np.eye(4))


def test_default_passport_pose():
    artist = Passport()
    assert_array_almost_equal(artist.markers2origin, np.eye(4))


def test_default_closed_passport_pose():
    artist = PassportClosed()
    assert_array_almost_equal(artist.markers2origin, np.eye(4))


def test_default_passport_box_pose():
    artist = PassportBox()
    assert_array_almost_equal(artist.markers2origin, np.eye(4))


def test_load_meshes():
    for Object in [Insole, PillowSmall, Passport]:
        mesh = Object().load_mesh()
        assert mesh.vertices
        assert mesh.triangles
        assert mesh.triangle_normals


def test_pose_from_markers():
    for Object in [Insole, PillowSmall, Passport]:
        obj = Object()
        obj.pose_from_markers(**Object.default_marker_positions)
        assert_array_almost_equal(obj.markers2origin, np.eye(4))


def test_transform_from_insole_mesh_to_origin():
    artist = Insole()
    artist.set_data(insole_back=np.array([0, 0, 0.1]),
                    insole_front=np.array([0.19, 0, 0.1]))
    p_in_origin = artist.transform_from_mesh_to_origin([0.1, 0.2, 0.3])
    assert_array_almost_equal(p_in_origin, [0.049615, -0.134307, -0.207])


def test_transform_from_pillow_mesh_to_origin():
    pillow_left = np.array([-0.11, 0.13, 0.2])
    pillow_right = np.array([-0.11, -0.13, 0.2])
    pillow_top = np.array([0.11, -0.13, 0.2])
    artist = PillowSmall()
    artist.set_data(pillow_left=pillow_left, pillow_right=pillow_right,
                    pillow_top=pillow_top)
    p_in_origin = artist.transform_from_mesh_to_origin([0.1, 0.2, 0.3])
    assert_array_almost_equal(p_in_origin, [0.22, -0.1, 0.405])
