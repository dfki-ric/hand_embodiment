"""Interactive embodiment mapping.

Example call:

python bin/gui_robot_embodiment.py mia --mano-config examples/config/mano/20220718_april.yaml --robot-config examples/config/robot/20220718_april_mia.yaml
"""
import argparse
from functools import partial
import numpy as np
import open3d as o3d
from open3d.visualization import gui
import pytransform3d.visualizer as pv
import pytransform3d.transformations as pt
import yaml
from hand_embodiment.mano import HandState, apply_shape_parameters
from hand_embodiment.record_markers import make_finger_kinematics
from hand_embodiment.target_configurations import TARGET_CONFIG
from hand_embodiment.embodiment import HandEmbodiment
from hand_embodiment.config import load_mano_config
from hand_embodiment.command_line import add_hand_argument


class Figure:
    def __init__(self, window_name, robot_config, width, height, ax_s=1.0):
        self.window_name = window_name
        self.robot_config = robot_config
        self.width = width
        self.height = height

        self.pose = None

        gui.Application.instance.initialize()
        self.window = gui.Application.instance.create_window(
            title=self.window_name, width=self.width, height=self.height)

        em = self.window.theme.font_size
        self.layout = gui.TabControl()
        self.tab1 = gui.Vert(0, gui.Margins(0.5 * em, 0.5 * em, 0.5 * em, 0.5 * em))
        self.layout.add_tab("Control URDF", self.tab1)
        self.tab2 = gui.Vert(0, gui.Margins(0.5 * em, 0.5 * em, 0.5 * em, 0.5 * em))
        self.layout.add_tab("MANO Transf.", self.tab2)
        self.tab3 = gui.Vert(0, gui.Margins(0.5 * em, 0.5 * em, 0.5 * em, 0.5 * em))
        self.layout.add_tab("MANO Param.", self.tab3)

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
        self.menu.add_item("Save robot parameters", SAVE_ID)
        QUIT_ID = 2
        self.menu.add_item("Quit", QUIT_ID)
        self.main_menu = gui.Menu()
        self.main_menu.add_menu("Menu", self.menu)
        gui.Application.instance.menubar = self.main_menu

        self.window.set_on_menu_item_activated(
            SAVE_ID, self.save_robot_parameters)
        self.window.set_on_menu_item_activated(
            QUIT_ID, gui.Application.instance.quit)

        self.main_scene = self.scene_widget.scene
        self.geometry_names = []

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

    def register_new_pose(self, pose):
        self.pose = pose

    def save_robot_parameters(self):
        with open(self.robot_config, "w") as f:
            yaml.dump({"handbase2robotbase": self.pose.tolist()}, f)

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
            try:  # Open3D <= 0.13
                material = o3d.visualization.rendering.Material()
                material.shader = "defaultLit"
            except AttributeError:  # Open3d >= 0.14
                material = o3d.visualization.rendering.MaterialRecord()
                material.shader = "defaultLit"
        self.main_scene.add_geometry(name, geometry, material)

    def add_hand_mesh(self, mesh, material):
        self.main_scene.add_geometry("MANO", mesh, material)


def make_mia_widgets(fig, graph, tm):
    em = fig.window.theme.font_size

    fig.tab1.add_child(gui.Label("Options"))
    fig.tab1.add_fixed(em)
    fig.tab1.add_child(gui.Label("Joint Angles"))
    for joint_name, joint_data in tm._joints.items():
        joint_limits = joint_data[-2]

        joint_control_layout = gui.Horiz()
        joint_control_layout.add_child(gui.Label(joint_name))
        slider = gui.Slider(gui.Slider.DOUBLE)
        slider.set_limits(*joint_limits)
        slider.set_on_value_changed(OnSlider(fig, graph, tm, joint_name))
        joint_control_layout.add_child(slider)

        fig.tab1.add_child(joint_control_layout)

    fig.tab1.add_fixed(em)
    fig.tab1.add_child(gui.Label("Display"))
    fig.tab1.add_fixed(em)

    frames_checkbox = gui.Checkbox("Show frames")
    frames_checkbox.checked = True
    frames_checkbox.set_on_checked(OnFramesCheckbox(fig, graph, tm))
    fig.tab1.add_child(frames_checkbox)

    visuals_checkbox = gui.Checkbox("Show visuals")
    visuals_checkbox.checked = True
    visuals_checkbox.set_on_checked(OnVisualsCheckbox(fig, graph, tm))
    fig.tab1.add_child(visuals_checkbox)

    collision_objects_checkbox = gui.Checkbox("Show collision objects")
    collision_objects_checkbox.checked = True
    collision_objects_checkbox.set_on_checked(OnCollisionObjectsCheckbox(fig, graph, tm))
    fig.tab1.add_child(collision_objects_checkbox)


def make_mano_widgets(fig, hand_state, graph, tm, embodiment, show_mano, hand_config, mano_pose):
    em = fig.window.theme.font_size

    fig.tab2.add_child(gui.Label("MANO transformation"))
    mano_pos_state = OnManoPoseSlider(
        fig, hand_state, graph, tm, embodiment, show_mano, hand_config, mano_pose)
    for i in range(3):
        pose_control_layout = gui.Horiz()
        pose_control_layout.add_child(gui.Label(f"{i + 1}"))
        slider = gui.Slider(gui.Slider.DOUBLE)
        slider.set_limits(-np.pi, np.pi)
        slider.double_value = mano_pos_state.pose[i]
        slider.set_on_value_changed(partial(mano_pos_state.pos_changed, i=i))
        pose_control_layout.add_child(slider)
        fig.tab2.add_child(pose_control_layout)
    for i in range(3):
        pose_control_layout = gui.Horiz()
        pose_control_layout.add_child(gui.Label(f"{i + 4}"))
        slider = gui.Slider(gui.Slider.DOUBLE)
        slider.set_limits(-0.5, 0.5)
        slider.double_value = mano_pos_state.pose[i + 3]
        slider.set_on_value_changed(partial(mano_pos_state.pos_changed, i=i + 3))
        pose_control_layout.add_child(slider)
        fig.tab2.add_child(pose_control_layout)

    joint_limits = (-np.pi, np.pi)
    for pose_parameter_i in range(0, hand_state.n_pose_parameters, 3):
        joint_control_layout = gui.Horiz()
        for pose_parameter_j in range(3):
            slider = gui.Slider(gui.Slider.DOUBLE)
            slider.set_limits(*joint_limits)
            slider.double_value = mano_pos_state.hand_state.pose[pose_parameter_i + pose_parameter_j]
            slider.set_on_value_changed(partial(mano_pos_state.joint_changed, i=pose_parameter_i + pose_parameter_j))

            joint_control_layout.add_fixed(em)
            joint_control_layout.add_child(gui.Label("%02d" % (pose_parameter_i + pose_parameter_j + 1)))
            joint_control_layout.add_child(slider)
        fig.tab3.add_child(joint_control_layout)


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


class OnMano(On):
    def __init__(self, fig, hand_state, graph, tm, show_mano):
        super(OnMano, self).__init__(fig, graph, tm)
        self.fig = fig
        self.hand_state = hand_state
        self.show_mano = show_mano

    def redraw(self):
        if self.show_mano:
            self.fig.main_scene.remove_geometry("MANO")
            self.fig.add_hand_mesh(self.hand_state.hand_mesh, self.hand_state.material)
        super(OnMano, self).redraw()


class OnManoPoseSlider(OnMano):
    def __init__(self, fig, hand_state, graph, tm, embodiment, show_mano, hand_config, mano_pose):
        super(OnManoPoseSlider, self).__init__(
            fig, hand_state, graph, tm, show_mano)
        mano2robot = hand_config["handbase2robotbase"]
        self.pose = pt.exponential_coordinates_from_transform(mano2robot)
        self.hand_state.pose[:] = mano_pose
        self.embodiment = embodiment
        self.mano_index_kin = make_finger_kinematics(hand_state, "index")

    def pos_changed(self, value, i):
        self.pose[i] = value
        self.fig.register_new_pose(self.pose)
        self.update_pose()
        self.redraw()

    def joint_changed(self, value, i):
        self.hand_state.pose[i] = value

        self.embodiment.solve(use_cached_forward_kinematics=False)

        self.update_pose()
        self.redraw()

    def update_pose(self):
        self.hand_state.recompute_mesh(
            pt.transform_from_exponential_coordinates(self.pose))


def parse_args():
    parser = argparse.ArgumentParser()
    add_hand_argument(parser)
    parser.add_argument(
        "--hide-mano", action="store_true", help="Don't show MANO mesh")
    parser.add_argument(
        "--mano-config", type=str,
        default="examples/config/april_test_mano.yaml",
        help="MANO configuration file.")
    parser.add_argument(
        "--robot-config", type=str, default=None,
        help="Target system configuration file.")

    return parser.parse_args()


def main():
    args = parse_args()

    hand_config = TARGET_CONFIG[args.hand]
    if args.robot_config is not None:
        with open(args.robot_config, "r") as f:
            hand_config_update = yaml.safe_load(f)
            if "handbase2robotbase" in hand_config_update:
                hand_config_update["handbase2robotbase"] = \
                    pt.transform_from_exponential_coordinates(
                        hand_config_update["handbase2robotbase"])
            hand_config.update(hand_config_update)

    if args.hand == "mia":
        mano_pose = np.array([
            0, 0, 0,
            -0.068, 0, 0.068,
            0, 0.068, 0.068,
            0, 0, 0.615,
            0, 0.137, 0.068,
            0, 0, 0.137,
            0, 0, 0.683,
            0, 0.205, -0.137,
            0, 0.068, 0.205,
            0, 0, 0.205,
            0, 0.137, -0.137,
            0, -0.068, 0.273,
            0, 0, 0.478,
            0.615, 0.068, 0.273,
            0, 0, 0,
            0, 0, 0
        ])
    else:
        mano_pose = np.zeros(48)

    hand_state = HandState(left=False)
    # TODO refactor
    _, shape_parameters = load_mano_config(args.mano_config)
    hand_state.betas[:] = shape_parameters
    hand_state.pose_parameters["J"], \
        hand_state.pose_parameters["v_template"] = \
        apply_shape_parameters(betas=shape_parameters,
                               **hand_state.shape_parameters)
    fig = Figure(args.hand, args.robot_config, 1920, 1080, ax_s=0.2)
    if not args.hide_mano:
        fig.add_hand_mesh(hand_state.hand_mesh, hand_state.material)
    emb = HandEmbodiment(
        hand_state, hand_config,
        use_fingers=hand_config["ee_frames"].keys())

    whitelist = [
        node for node in emb.transform_manager_.nodes
        if not (node.startswith("visual:") or
                node.startswith("collision:") or
                node.startswith("inertial_frame:"))]
    graph = pv.Graph(
        emb.transform_manager_, "world", show_frames=True,
        show_connections=False, show_visuals=True, show_collision_objects=True,
        show_name=False, s=0.02, whitelist=whitelist)
    graph.add_artist(fig)

    make_mia_widgets(fig, graph, emb.transform_manager_)
    make_mano_widgets(fig, hand_state, graph, emb.transform_manager_, emb,
                      not args.hide_mano, hand_config, mano_pose)

    fig.show()


if __name__ == "__main__":
    main()
