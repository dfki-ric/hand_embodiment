"""
Example call:

python bin/gui_mano_shape.py examples/config/mano/20210610_april.yaml --mocap-filename data/20210610_april/Measurement2.tsv --mocap-config examples/config/markers/20210610_april.yaml --start-idx 3000 --fit-fingers
"""
import argparse
import os
from functools import partial
import numpy as np
import open3d as o3d
from open3d.visualization import gui
import pytransform3d.transformations as pt

from hand_embodiment.record_markers import MarkerBasedRecordMapping
from hand_embodiment.vis_utils import make_coordinate_system
from hand_embodiment.mocap_dataset import HandMotionCaptureDataset
from hand_embodiment.config import load_mano_config, save_mano_config


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "filename", type=str,
        help="Configuration file. Modifies existing one or creates new one.")
    parser.add_argument(
        "--mocap-filename", type=str, help="Path to motion capture recording.",
        default="data/QualisysAprilTest/april_test_010.tsv")
    parser.add_argument(
        "--mocap-config", type=str,
        default="examples/config/markers/20210520_april.yaml",
        help="MoCap configuration file.")
    parser.add_argument(
        "--start-idx", type=int, default=100,
        help="Index of frame that we visualize.")
    parser.add_argument(
        "--fit-fingers", action="store_true",
        help="Fit fingers to marker configuration.")
    return parser.parse_args()


class Figure:
    def __init__(self, window_name, width, height, config_filename, ax_s=1.0):
        self.window_name = window_name
        self.width = width
        self.height = height
        self.config_filename = config_filename

        gui.Application.instance.initialize()
        self.window = gui.Application.instance.create_window(
            title=self.window_name, width=self.width, height=self.height)

        em = self.window.theme.font_size
        self.layout = gui.TabControl()
        self.tab1 = gui.Vert(0, gui.Margins(0.5 * em, 0.5 * em, 0.5 * em, 0.5 * em))
        self.layout.add_tab("MANO Shape", self.tab1)
        self.tab2 = gui.Vert(0, gui.Margins(0.5 * em, 0.5 * em, 0.5 * em, 0.5 * em))
        self.layout.add_tab("Transform", self.tab2)

        self.scene_widget = gui.SceneWidget()
        self.scene_widget.scene = o3d.visualization.rendering.Open3DScene(self.window.renderer)
        self.scene_widget.scene.set_background((0.8, 0.8, 0.8, 1))
        self.bounds = o3d.geometry.AxisAlignedBoundingBox(-ax_s * np.ones(3), ax_s * np.ones(3))
        self.scene_widget.setup_camera(60, self.bounds, self.bounds.get_center())

        self.window.set_on_layout(self._on_layout)
        self.window.add_child(self.scene_widget)
        self.window.add_child(self.layout)

        self.menu = gui.Menu()
        SAVE_ID = 1
        self.menu.add_item("Save MANO parameters", SAVE_ID)
        QUIT_ID = 2
        self.menu.add_item("Quit", QUIT_ID)
        self.main_menu = gui.Menu()
        self.main_menu.add_menu("Menu", self.menu)
        gui.Application.instance.menubar = self.main_menu

        self.window.set_on_menu_item_activated(
            SAVE_ID, self.save_mano_parameters)
        self.window.set_on_menu_item_activated(
            QUIT_ID, gui.Application.instance.quit)

        self.main_scene = self.scene_widget.scene
        self.geometry_names = []
        self.mbrm = None

    def _on_layout(self, layout_context):
        # The on_layout callback should set the frame (position + size) of every
        # child correctly. After the callback is done the window will layout
        # the grandchildren.
        r = self.window.content_rect
        self.scene_widget.frame = r
        width = 30 * layout_context.theme.font_size
        height = min(
            r.height,
            self.layout.calc_preferred_size(
                layout_context, gui.Widget.Constraints()).height)
        self.layout.frame = gui.Rect(r.get_right() - width, r.y, width, height)

    def show(self):
        gui.Application.instance.run()

    def add_geometry(self, geometry, material=None):
        """Add geometry to visualizer.

        Parameters
        ----------
        geometry : Geometry
            Open3D geometry.

        material : Material, optional (default: None)
            Open3D material.
        """
        name = str(len(self.geometry_names))
        self.geometry_names.append(name)
        if material is None:
            material = o3d.visualization.rendering.Material()
        self.main_scene.add_geometry(name, geometry, material)

    def add_hand_mesh(self, mesh, material):
        self.main_scene.add_geometry("MANO", mesh, material)

    def save_mano_parameters(self):
        if self.mbrm is None:
            print("No parameters available")
        else:
            save_mano_config(
                self.config_filename, self.mbrm.mano2hand_markers_,
                self.mbrm.hand_state_.betas)

    def make_mano_widgets(self, mbrm):
        self.mbrm = mbrm
        self.tab1.add_child(gui.Label("MANO shape"))
        mano_change = OnManoChange(self, mbrm)
        for i in range(mbrm.hand_state_.n_shape_parameters):
            pose_control_layout = gui.Horiz()
            pose_control_layout.add_child(gui.Label(f"{(i + 1):02d}"))
            slider = gui.Slider(gui.Slider.DOUBLE)
            slider.set_limits(-10, 10)
            slider.double_value = mbrm.hand_state_.betas[i]
            slider.set_on_value_changed(partial(mano_change.shape_changed, i=i))
            pose_control_layout.add_child(slider)
            self.tab1.add_child(pose_control_layout)

        self.tab2.add_child(gui.Label("MANO to Hand Markers / Exponential Coordinates"))
        names = ["o_1", "o_2", "o_3", "v_1", "v_2", "v_3"]
        ranges = [2, 2, 2, 0.1, 0.1, 0.1]
        initial_pose = pt.exponential_coordinates_from_transform(mbrm.mano2hand_markers_)
        for i, (name, r) in enumerate(zip(names, ranges)):
            pose_control_layout = gui.Horiz()
            pose_control_layout.add_child(gui.Label(name))
            slider = gui.Slider(gui.Slider.DOUBLE)
            slider.set_limits(initial_pose[i] - r, initial_pose[i] + r)
            slider.double_value = initial_pose[i]
            slider.set_on_value_changed(partial(mano_change.pos_changed, i=i))
            pose_control_layout.add_child(slider)
            self.tab2.add_child(pose_control_layout)
        mano_change.update_mesh()


class OnMano:
    def __init__(self, fig, mbrm):
        self.fig = fig
        self.mbrm = mbrm

    def redraw(self):
        self.fig.main_scene.remove_geometry("MANO")
        self.fig.add_hand_mesh(
            self.mbrm.hand_state_.hand_mesh, self.mbrm.hand_state_.material)


class OnManoChange(OnMano):
    def __init__(self, fig, mbrm):
        super(OnManoChange, self).__init__(fig, mbrm)
        self.mbrm = mbrm
        self.update_mesh()
        self.redraw()

    def shape_changed(self, value, i):
        self.mbrm.hand_state_.betas[i] = value
        self.update_mesh()
        self.redraw()

    def pos_changed(self, value, i):
        pose = pt.exponential_coordinates_from_transform(self.mbrm.mano2hand_markers_)
        pose[i] = value
        self.mbrm.mano2hand_markers_ = pt.transform_from_exponential_coordinates(pose)
        self.update_mesh()
        self.redraw()

    def update_mesh(self):
        self.mbrm.hand_state_.recompute_shape()
        self.mbrm.hand_state_.recompute_mesh(self.mbrm.mano2hand_markers_)


def main():
    args = parse_args()

    dataset = HandMotionCaptureDataset(
        args.mocap_filename, mocap_config=args.mocap_config)

    config_filename = args.filename
    if os.path.exists(config_filename):
        mano2hand_markers, betas = load_mano_config(config_filename)
    else:
        mano2hand_markers, betas = np.eye(4), np.zeros(10)

    mbrm = MarkerBasedRecordMapping(
        left=False, shape_parameters=betas, mano2hand_markers=mano2hand_markers)

    if args.fit_fingers:
        finger_markers = dataset.get_finger_markers(args.start_idx)
    else:
        finger_markers = {}
    mbrm.estimate(dataset.get_hand_markers(args.start_idx), finger_markers)
    world2mano = pt.invert_transform(mbrm.mano2world_)
    markers_in_world = dataset.get_markers(args.start_idx)
    markers_in_mano = pt.transform(
        mano2hand_markers, pt.transform(
            world2mano, pt.vectors_to_points(markers_in_world)))[:, :3]

    fig = Figure("MANO shape", 1920, 1080, config_filename, ax_s=0.2)

    for p in markers_in_mano:  # TODO refactor
        marker = o3d.geometry.TriangleMesh.create_sphere(radius=0.006)
        n_vertices = len(marker.vertices)
        colors = np.zeros((n_vertices, 3))
        colors[:] = (0.3, 0.3, 0.3)
        marker.vertex_colors = o3d.utility.Vector3dVector(colors)
        marker.translate(p)
        marker.compute_vertex_normals()
        marker.compute_triangle_normals()
        fig.add_geometry(marker)

    coordinate_system = make_coordinate_system(s=0.2)
    fig.add_geometry(coordinate_system)

    fig.make_mano_widgets(mbrm)

    fig.show()


if __name__ == "__main__":
    main()
