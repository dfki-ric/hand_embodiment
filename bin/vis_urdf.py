import argparse
import numpy as np
import open3d as o3d
import pytransform3d.visualizer as pv
from open3d.visualization import gui
from hand_embodiment.embodiment import load_kinematic_model
from hand_embodiment.target_configurations import (
    MIA_CONFIG, SHADOW_HAND_CONFIG)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "hand", type=str,
        help="Name of the hand. Possible options: mia, shadow_hand")

    return parser.parse_args()


class Figure:
    def __init__(self, window_name, width, height, ax_s=1.0):
        self.window_name = window_name
        self.width = width
        self.height = height

        gui.Application.instance.initialize()
        self.window = gui.Application.instance.create_window(
            title=self.window_name, width=self.width, height=self.height)

        em = self.window.theme.font_size
        self.layout = gui.Vert(0, gui.Margins(0.5 * em, 0.5 * em, 0.5 * em, 0.5 * em))

        self.scene_widget = gui.SceneWidget()
        self.scene_widget.scene = o3d.visualization.rendering.Open3DScene(self.window.renderer)
        self.scene_widget.scene.set_background((0.8, 0.8, 0.8, 1))
        self.bounds = o3d.geometry.AxisAlignedBoundingBox(-ax_s * np.ones(3), ax_s * np.ones(3))
        self.scene_widget.setup_camera(60, self.bounds, self.bounds.get_center())

        self.window.set_on_layout(self._on_layout)
        self.window.add_child(self.scene_widget)
        self.window.add_child(self.layout)

        self.menu = gui.Menu()
        QUIT_ID = 1
        self.menu.add_item("Quit", QUIT_ID)
        self.main_menu = gui.Menu()
        self.main_menu.add_menu("Menu", self.menu)
        gui.Application.instance.menubar = self.main_menu
        self.window.set_on_menu_item_activated(QUIT_ID, gui.Application.instance.quit)
        self.main_scene = self.scene_widget.scene
        self.geometry_names = []

    def _on_layout(self, theme):
        r = self.window.content_rect
        self.scene_widget.frame = r
        width = 17 * theme.font_size
        height = min(
            r.height, self.layout.calc_preferred_size(theme).height)
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


def make_widgets(fig, graph, tm):
    em = fig.window.theme.font_size

    fig.layout.add_child(gui.Label("Options"))
    fig.layout.add_fixed(em)
    fig.layout.add_child(gui.Label("Joint Angles"))
    for joint_name, joint_data in tm._joints.items():
        joint_limits = joint_data[-2]

        joint_control_layout = gui.Horiz()
        joint_control_layout.add_child(gui.Label(joint_name))
        slider = gui.Slider(gui.Slider.DOUBLE)
        slider.set_limits(*joint_limits)
        slider.set_on_value_changed(OnSlider(fig, graph, tm, joint_name))
        joint_control_layout.add_child(slider)

        fig.layout.add_child(joint_control_layout)

    fig.layout.add_fixed(em)
    fig.layout.add_child(gui.Label("Display"))
    fig.layout.add_fixed(em)

    frames_checkbox = gui.Checkbox("Show frames")
    frames_checkbox.checked = True
    frames_checkbox.set_on_checked(OnFramesCheckbox(fig, graph, tm))
    fig.layout.add_child(frames_checkbox)

    visuals_checkbox = gui.Checkbox("Show visuals")
    visuals_checkbox.checked = True
    visuals_checkbox.set_on_checked(OnVisualsCheckbox(fig, graph, tm))
    fig.layout.add_child(visuals_checkbox)

    collision_objects_checkbox = gui.Checkbox("Show collision objects")
    collision_objects_checkbox.checked = True
    collision_objects_checkbox.set_on_checked(OnCollisionObjectsCheckbox(fig, graph, tm))
    fig.layout.add_child(collision_objects_checkbox)


class On:
    def __init__(self, fig, graph, tm):
        self.fig = fig
        self.graph = graph
        self.tm = tm

    def redraw(self):
        if not self.graph.show_visuals:
            visuals = self.graph.visuals
            self.graph.visuals = {}
        if not self.graph.show_collision_objects:
            collision_objects = self.graph.collision_objects
            self.graph.collision_objects = {}

        self.graph.set_data()

        # self.fig.main_scene.update_geometry(g) does not exist yet
        # remove and add everything
        for g in self.fig.geometry_names:
            self.fig.main_scene.remove_geometry(g)
        self.fig.geometry_names = []
        self.graph.add_artist(self.fig)

        if not self.graph.show_visuals:
            self.graph.visuals = visuals
        if not self.graph.show_collision_objects:
            self.graph.collision_objects = collision_objects


class OnSlider(On):
    def __init__(self, fig, graph, tm, joint_name):
        super(OnSlider, self).__init__(fig, graph, tm)
        self.joint_name = joint_name

    def __call__(self, value):
        self.tm.set_joint(self.joint_name, value)
        self.redraw()


class OnFramesCheckbox(On):
    def __call__(self, is_checked):
        self.graph.show_frames = is_checked
        self.redraw()


class OnVisualsCheckbox(On):
    def __init__(self, fig, graph, tm):
        super(OnVisualsCheckbox, self).__init__(fig, graph, tm)

    def __call__(self, is_checked):
        self.graph.show_visuals = is_checked
        self.redraw()


class OnCollisionObjectsCheckbox(On):
    def __init__(self, fig, graph, tm):
        super(OnCollisionObjectsCheckbox, self).__init__(fig, graph, tm)
        self.collision_objects = {}

    def __call__(self, is_checked):
        self.graph.show_collision_objects = is_checked
        self.redraw()


def main(args):
    fig = Figure("URDF visualization", 1920, 1080, ax_s=0.2)

    if args.hand == "shadow_hand":
        hand_config = SHADOW_HAND_CONFIG
    elif args.hand == "mia":
        hand_config = MIA_CONFIG
    else:
        raise Exception(f"Unknown hand: '{args.hand}'")

    tm = load_kinematic_model(hand_config).tm
    graph = pv.Graph(
        tm, hand_config["base_frame"], show_frames=True,
        show_connections=False, show_visuals=True, show_collision_objects=True,
        show_name=False, s=0.02)
    graph.add_artist(fig)
    make_widgets(fig, graph, tm)

    fig.show()


if __name__ == "__main__":
    args = parse_args()
    main(args)
