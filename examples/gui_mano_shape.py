from functools import partial
import numpy as np
import open3d as o3d
from open3d.visualization import gui
from mocap.mano import HandState
import pytransform3d.transformations as pt

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
        self.layout.add_tab("MANO shape", self.tab1)

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


def make_mano_widgets(fig, hand_state):
    em = fig.window.theme.font_size

    fig.tab1.add_child(gui.Label("MANO shape"))
    mano_change = OnManoChange(fig, hand_state)
    for i in range(hand_state.n_shape_parameters):
        pose_control_layout = gui.Horiz()
        pose_control_layout.add_child(gui.Label(f"{(i + 1):02d}"))
        slider = gui.Slider(gui.Slider.DOUBLE)
        slider.set_limits(-10, 10)
        slider.double_value = 0
        slider.set_on_value_changed(partial(mano_change.shape_changed, i=i))
        pose_control_layout.add_child(slider)
        fig.tab1.add_child(pose_control_layout)


class OnMano:
    def __init__(self, fig, hand_state):
        self.fig = fig
        self.hand_state = hand_state

    def redraw(self):
        self.fig.main_scene.remove_geometry("MANO")
        self.fig.add_hand_mesh(self.hand_state.hand_mesh, self.hand_state.material)


class OnManoChange(OnMano):
    def __init__(self, fig, hand_state):
        super(OnManoChange, self).__init__(fig, hand_state)

    def shape_changed(self, value, i):
        self.hand_state.set_shape_parameter(i, value)
        self.redraw()


fig = Figure("MANO shape", 1920, 1080, ax_s=0.2)

hand_state = HandState(left=False)
fig.add_hand_mesh(hand_state.hand_mesh, hand_state.material)

import glob
from mocap import qualisys, pandas_utils, cleaning, conversion
pattern = "data/Qualisys_pnp/*.tsv"
demo_idx = 2
filename = list(sorted(glob.glob(pattern)))[demo_idx]
trajectory = qualisys.read_qualisys_tsv(filename=filename)
hand_trajectory = pandas_utils.extract_markers(trajectory, ["Hand left", "Hand right", "Hand top", "Middle", "Index", "Thumb"])
hand_trajectory = hand_trajectory.iloc[100:200]
hand_trajectory = cleaning.median_filter(cleaning.interpolate_nan(hand_trajectory), 3).iloc[2:]
hand_left = conversion.array_from_dataframe(hand_trajectory, ["Hand left X", "Hand left Y", "Hand left Z"])
hand_right = conversion.array_from_dataframe(hand_trajectory, ["Hand right X", "Hand right Y", "Hand right Z"])
hand_top = conversion.array_from_dataframe(hand_trajectory, ["Hand top X", "Hand top Y", "Hand top Z"])
middle = conversion.array_from_dataframe(hand_trajectory, ["Middle X", "Middle Y", "Middle Z"])
index = conversion.array_from_dataframe(hand_trajectory, ["Index X", "Index Y", "Index Z"])
thumb = conversion.array_from_dataframe(hand_trajectory, ["Thumb X", "Thumb Y", "Thumb Z"])
t = 0
mbrm = MarkerBasedRecordMapping(left=False)
mbrm.estimate(
    [hand_top[t], hand_left[t], hand_right[t]],
    {"thumb": thumb[t], "index": index[t], "middle": middle[t]})
world2mano = pt.invert_transform(mbrm.mano2world_)
markers_in_world = np.array([hand_top[t], hand_left[t], hand_right[t], thumb[t], index[t], middle[t]])
markers_in_mano = pt.transform(world2mano, pt.vectors_to_points(markers_in_world))[:, :3]

for p in markers_in_mano:
    marker = o3d.geometry.TriangleMesh.create_sphere(radius=0.005)
    marker.compute_vertex_normals()
    n_vertices = len(marker.vertices)
    colors = np.zeros((n_vertices, 3))
    colors[:] = (0.3, 0.3, 0.3)
    marker.vertex_colors = o3d.utility.Vector3dVector(colors)
    marker.translate(p)
    fig.add_geometry(marker)

coordinate_system = make_coordinate_system(s=0.2)
fig.add_geometry(coordinate_system)

make_mano_widgets(fig, hand_state)

fig.show()
