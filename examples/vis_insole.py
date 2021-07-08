from pkg_resources import resource_filename
import numpy as np
import open3d as o3d
import pytransform3d.transformations as pt
from pytransform3d import visualizer as pv
from mocap.visualization import scatter
from hand_embodiment.vis_utils import Insole, make_coordinate_system


def main():
    insole_back = np.array([244.92, 539.448, 130.386]) / 1000.0
    insole_front = np.array([183.12, 362.562, 131.816]) / 1000.0

    fig = pv.figure()
    coordinate_system = make_coordinate_system(s=0.3)
    fig.add_geometry(coordinate_system)
    markers = scatter(fig, [insole_back, insole_front], s=0.006)
    # TODO rescale insole
    # TODO check marker positions
    # TODO compare to location in PyBullet
    insole = Insole(insole_back, insole_front)
    insole.add_artist(fig)
    mesh2origin = pt.concat(insole.insole_mesh2insole, insole.insole2origin)
    fig.plot_transform(mesh2origin, s=0.15)  # TODO what is that????
    fig.plot_transform(insole.insole2origin, s=0.1)
    fig.plot_transform(np.eye(4), s=0.1)
    filename = resource_filename("hand_embodiment", "model/objects/insole.stl")
    mesh = o3d.io.read_triangle_mesh(filename)
    mesh.paint_uniform_color(np.array([0.37, 0.28, 0.26]))
    mesh.compute_triangle_normals()
    fig.add_geometry(mesh)
    pc = o3d.geometry.PointCloud(mesh.vertices)
    fig.add_geometry(pc)

    fig.show()


if __name__ == "__main__":
    main()
