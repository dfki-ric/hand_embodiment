from functools import partial
import numpy as np
import open3d as o3d
from open3d.visualization import gui
from mocap.mano import HandState
from mocap import mano
import pytransform3d.transformations as pt
import pytransform3d.rotations as pr

from hand_embodiment.record_markers import MarkerBasedRecordMapping
from hand_embodiment.vis_utils import make_coordinate_system


class Figure:
    def __init__(self, window_name, width, height, ax_s=1.0):
        self.window_name = window_name
        self.width = width
        self.height = height

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
        width = 30 * theme.font_size
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

    def add_hand_mesh(self, mesh, material):
        self.main_scene.add_geometry("MANO", mesh, material)


def make_mano_widgets(fig, hand_state, initial_pose, initial_shape):
    em = fig.window.theme.font_size

    fig.tab1.add_child(gui.Label("MANO shape"))
    mano_change = OnManoChange(fig, hand_state, initial_pose, initial_shape)
    for i in range(hand_state.n_shape_parameters):
        pose_control_layout = gui.Horiz()
        pose_control_layout.add_child(gui.Label(f"{(i + 1):02d}"))
        slider = gui.Slider(gui.Slider.DOUBLE)
        slider.set_limits(-10, 10)
        slider.double_value = initial_shape[i]
        slider.set_on_value_changed(partial(mano_change.shape_changed, i=i))
        pose_control_layout.add_child(slider)
        fig.tab1.add_child(pose_control_layout)

    fig.tab2.add_child(gui.Label("MANO to Hand Markers / Exponential Coordinates"))
    names = ["o_1", "o_2", "o_3", "v_1", "v_2", "v_3"]
    ranges = [2, 2, 2, 0.1, 0.1, 0.1]
    for i, (name, r) in enumerate(zip(names, ranges)):
        pose_control_layout = gui.Horiz()
        pose_control_layout.add_child(gui.Label(name))
        slider = gui.Slider(gui.Slider.DOUBLE)
        slider.set_limits(initial_pose[i] - r, initial_pose[i] + r)
        slider.double_value = initial_pose[i]
        slider.set_on_value_changed(partial(mano_change.pos_changed, i=i))
        pose_control_layout.add_child(slider)
        fig.tab2.add_child(pose_control_layout)
        mano_change.shape_changed(initial_pose[i], i)
    mano_change.update_mesh()


class OnMano:
    def __init__(self, fig, hand_state):
        self.fig = fig
        self.hand_state = hand_state

    def redraw(self):
        self.fig.main_scene.remove_geometry("MANO")
        self.fig.add_hand_mesh(self.hand_state.hand_mesh, self.hand_state.material)


class OnManoChange(OnMano):
    def __init__(self, fig, hand_state, initial_pose, initial_shape):
        super(OnManoChange, self).__init__(fig, hand_state)
        self.pose = initial_pose.copy()
        self.shape = initial_shape.copy()

    def shape_changed(self, value, i):
        self.shape[i] = value
        self.hand_state.betas[i] = value
        self.update_mesh()
        self.redraw()

    def pos_changed(self, value, i):
        self.pose[i] = value
        self.update_mesh()
        self.redraw()

    def update_mesh(self):
        self.hand_state.pose_parameters["J"], self.hand_state.pose_parameters["v_template"] = \
            mano.apply_shape_parameters(betas=self.hand_state.betas, **self.hand_state.shape_parameters)
        mesh2world = pt.transform_from_exponential_coordinates(self.pose)
        self.hand_state.vertices[:, :] = mano.hand_vertices(
            pose=self.hand_state.pose, **self.hand_state.pose_parameters)
        self.hand_state.vertices[:, :] = pt.transform(
            mesh2world, pt.vectors_to_points(self.hand_state.vertices))[:, :3]
        self.hand_state._mesh.vertices = o3d.utility.Vector3dVector(self.hand_state.vertices)
        self.hand_state._mesh.compute_vertex_normals()
        self.hand_state._mesh.compute_triangle_normals()


fig = Figure("MANO shape", 1920, 1080, ax_s=0.2)

hand_state = HandState(left=False)
fig.add_hand_mesh(hand_state.hand_mesh, hand_state.material)

import glob
from mocap import qualisys, pandas_utils, cleaning, conversion
filename = "data/QualisysAprilTest/april_test_001.tsv"
trajectory = qualisys.read_qualisys_tsv(filename=filename)
hand_trajectory = pandas_utils.extract_markers(trajectory, ["hand_left", "hand_right", "hand_top", "ring_middle", "middle_middle", "index_middle", "ring_tip", "middle_tip", "index_tip", "thumb_tip"])
hand_trajectory = hand_trajectory.iloc[100:200]
hand_trajectory = cleaning.median_filter(cleaning.interpolate_nan(hand_trajectory), 3).iloc[2:]
hand_left = conversion.array_from_dataframe(hand_trajectory, ["hand_left X", "hand_left Y", "hand_left Z"])
hand_right = conversion.array_from_dataframe(hand_trajectory, ["hand_right X", "hand_right Y", "hand_right Z"])
hand_top = conversion.array_from_dataframe(hand_trajectory, ["hand_top X", "hand_top Y", "hand_top Z"])
ring_middle = conversion.array_from_dataframe(hand_trajectory, ["ring_middle X", "ring_middle Y", "ring_middle Z"])
middle_middle = conversion.array_from_dataframe(hand_trajectory, ["middle_middle X", "middle_middle Y", "middle_middle Z"])
index_middle = conversion.array_from_dataframe(hand_trajectory, ["index_middle X", "index_middle Y", "index_middle Z"])
ring = conversion.array_from_dataframe(hand_trajectory, ["ring_tip X", "ring_tip Y", "ring_tip Z"])
middle = conversion.array_from_dataframe(hand_trajectory, ["middle_tip X", "middle_tip Y", "middle_tip Z"])
index = conversion.array_from_dataframe(hand_trajectory, ["index_tip X", "index_tip Y", "index_tip Z"])
thumb = conversion.array_from_dataframe(hand_trajectory, ["thumb_tip X", "thumb_tip Y", "thumb_tip Z"])
t = 0
mbrm = MarkerBasedRecordMapping(left=False, mano2hand_markers=np.eye(4))  # TODO initialize transform
mbrm.estimate(
    [hand_top[t], hand_left[t], hand_right[t]],
    {"thumb": thumb[t], "index": index[t], "middle": middle[t], "ring": ring[t]})
world2mano = pt.invert_transform(mbrm.mano2world_)
markers_in_world = np.array([hand_top[t], hand_left[t], hand_right[t], thumb[t], index[t], middle[t], ring[t], index_middle[t], middle_middle[t], ring_middle[t]])
markers_in_mano = pt.transform(world2mano, pt.vectors_to_points(markers_in_world))[:, :3]

for p in markers_in_mano:
    marker = o3d.geometry.TriangleMesh.create_sphere(radius=0.005)
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

#initial_pose = pt.transform_from_exponential_coordinates(np.array([-0.041, 1.99, -0.133, -0.066, -0.03, 0.096]))
#initial_pose = pt.transform_from_exponential_coordinates(np.array([0.031, 1.98, -0.154, -0.066, -0.027, 0.094]))
#initial_pose = pt.transform_from_exponential_coordinates(np.array([0.103, 2.083, -0.123, -0.066, -0.034, 0.083]))
initial_pose = pt.transform_from_exponential_coordinates(np.array([-0.01, 1.97, -0.123, -0.066, -0.034, 0.083]))
initial_pose = pt.exponential_coordinates_from_transform(initial_pose)
initial_shape = np.array([-3.5, 0, 0, 0, 0, 0, 0, 0, 0, 0])
make_mano_widgets(fig, hand_state, initial_pose, initial_shape)

fig.show()
