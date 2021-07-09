from pkg_resources import resource_filename
import numpy as np
import open3d as o3d
import pytransform3d.transformations as pt
from pytransform3d import visualizer as pv
from mocap.visualization import scatter
from hand_embodiment.vis_utils import Insole, make_coordinate_system


def main():
    # TODO compare to location in PyBullet
    insole_back = np.array([244.92, 539.448, 130.386]) / 1000.0
    insole_front = np.array([183.12, 362.562, 131.816]) / 1000.0

    fig = pv.figure()
    coordinate_system = make_coordinate_system(s=0.3)
    fig.add_geometry(coordinate_system)
    markers = scatter(fig, [insole_back, insole_front], s=0.006)
    insole = Insole(insole_back, insole_front)
    insole.add_artist(fig)
    mesh2origin = pt.concat(pt.invert_transform(insole.insole_mesh2insole), insole.insole2origin)
    fig.plot_transform(mesh2origin, s=0.15)
    fig.plot_transform(insole.insole_mesh2insole, s=0.1)
    fig.plot_transform(np.eye(4), s=0.1)
    markers2_locations = pt.transform(pt.invert_transform(mesh2origin),
                                      pt.vectors_to_points(np.array([insole_back, insole_front])))[:, :3]
    markers2 = scatter(fig, markers2_locations, s=0.006)
    filename = resource_filename("hand_embodiment", "model/objects/insole.stl")
    mesh = o3d.io.read_triangle_mesh(filename)
    scale = 0.30 / 0.27486159  # TODO find exact length of insole
    mesh.vertices = o3d.utility.Vector3dVector(np.array(mesh.vertices) * scale)
    mesh.paint_uniform_color(np.array([0.37, 0.28, 0.26]))
    mesh.compute_triangle_normals()
    fig.add_geometry(mesh)
    pc = o3d.geometry.PointCloud(mesh.vertices)
    fig.add_geometry(pc)

    fig.show()


if __name__ == "__main__":
    main()
