"""
Example call:

python bin/gui_mano.py --show-mesh --vertices 76 77 141 142 148 162 196 197 198 199 275 468 469 470 471 484 485 486 487 488 489 491 494 496 497 502 503 506 507 508 510 513 514 521 523 524 527 540 542 543 544 545 546 547 548 549 550 551 552 553 554 555 559 560 561 563 564 565 566 567 568 569 570 571 572 573 574 575 576 577 578 579
python bin/gui_mano.py --zero-pose --show-mesh --hide-frame --vertices 5 6 7 31 123 124 125 126 248 249 266 267 698 699 700 701 703 704 710 711 713 714 717 730 732 733 734 735 736 737 738 739 740 741 742 743 745 753 754 755 756 757 758 759 760 761 762 763 764 765 766 767 768 46 47 48 49 62 63 64 65 93 128 132 137 138 139 140 149 150 151 155 156 164 165 166 167 168 169 170 171 172 173 174 177 189 194 195 222 223 224 225 237 238 245 280 281 298 300 301 321 322 323 324 325 326 327 328 329 330 331 332 333 336 337 340 341 342 343 344 345 346 347 348 349 350 351 352 353 354 355 74 75 76 77 151 228 246 262 271 277 288 356 357 358 359 370 371 372 373 374 375 376 377 378 379 380 384 385 386 387 392 393 396 397 398 401 402 403 405 409 410 412 413 431 432 433 434 435 436 437 438 439 440 441 442 452 453 454 455 456 457 458 459 460 461 462 465 466 467 580 581 582 583 594 595 596 597 600 601 602 603 604 605 606 607 609 612 614 615 618 619 620 621 622 624 625 630 631 638 639 640 641 642 659 660 661 662 663 664 665 666 667 668 669 670 672 676 677 680 681 682 683 684 685 686 687 688 689 690 691 692 693 694 695 76 77 141 142 148 162 196 197 198 199 275 468 469 470 471 484 485 486 487 488 489 491 494 496 497 502 503 506 507 508 510 513 514 521 523 524 527 540 542 543 544 545 546 547 548 549 550 551 552 553 554 555 559 560 561 563 564 565 566 567 568 569 570 571 572 573 574 575 576 577 578 579
"""
import argparse
from functools import partial
import numpy as np
import open3d as o3d
from open3d.visualization import gui
import pytransform3d.rotations as pr
import pytransform3d.transformations as pt
import pytransform3d.visualizer as pv
from mocap.mano import HandState
from mocap.visualization import PointCollection

from hand_embodiment.vis_utils import make_coordinate_system
from hand_embodiment.config import load_mano_config
from hand_embodiment.record_markers import MANO_CONFIG, make_finger_kinematics


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--vertices", type=int, nargs="*", default=[],
        help="Highlight vertices in red color.")
    parser.add_argument(
        "--joints", type=int, nargs="*", default=[],
        help="Highlight joints (they will be bigger).")
    parser.add_argument(
        "--show-mesh", action="store_true", help="Show mesh.")
    parser.add_argument(
        "--show-reference", action="store_true",
        help="Show coordinate frame for size reference.")
    parser.add_argument(
        "--show-transforms", action="store_true",
        help="Show reference frames of markers and MANO.")
    parser.add_argument(
        "--show-tips", action="store_true",
        help="Show tip vertices of fingers in green color.")
    parser.add_argument(
        "--show-spheres", action="store_true",
        help="Show spheres at tip positions.")
    parser.add_argument(
        "--color-fingers", action="store_true",
        help="Show finger vertices in uniform color.")
    parser.add_argument(
        "--zero-pose", action="store_true",
        help="Set all pose parameters to 0.")
    parser.add_argument(
        "--hide-frame", action="store_true",
        help="Hide coordinate frame.")
    parser.add_argument(
        "--config-filename", type=str, default=None,
        help="MANO configuration that includes shape parameters.")

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
        self.layout = gui.TabControl()
        self.tab1 = gui.Vert(0, gui.Margins(0.5 * em, 0.5 * em, 0.5 * em, 0.5 * em))
        self.layout.add_tab("MANO", self.tab1)

        self.scene_widget = gui.SceneWidget()
        self.scene_widget.scene = o3d.visualization.rendering.Open3DScene(self.window.renderer)
        self.scene_widget.scene.set_background((1, 1, 1, 1))
        self.bounds = o3d.geometry.AxisAlignedBoundingBox(-ax_s * np.ones(3), ax_s * np.ones(3))
        self.scene_widget.setup_camera(60, self.bounds, self.bounds.get_center())
        self.scene_widget.scene.scene.set_sun_light([0, 1, 0], [1.0, 1.0, 1.0], 75000)

        self.window.set_on_layout(self._on_layout)
        self.window.add_child(self.scene_widget)
        self.window.add_child(self.layout)

        self.menu = gui.Menu()
        PRINT_ID = 1
        self.menu.add_item("Print vertices", PRINT_ID)
        QUIT_ID = 2
        self.menu.add_item("Quit", QUIT_ID)
        self.main_menu = gui.Menu()
        self.main_menu.add_menu("Menu", self.menu)
        gui.Application.instance.menubar = self.main_menu

        self.window.set_on_menu_item_activated(
            PRINT_ID, self.print_vertices)
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

        Returns
        ---------
        name of geometry
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

    def clear_all_geometries(self):
        for name in self.geometry_names:
            self.main_scene.remove_geometry(name)
        self.geometry_names = []

    def print_vertices(self):
        print(" ".join(map(str, np.nonzero(self.mano_change.vertex_mask)[0].tolist())))

    def make_mano_widgets(self, pc, vertex_mask, hand_state, show_mesh):
        self.tab1.add_child(gui.Label("Highlight vertices"))
        self.mano_change = OnManoChange(self, pc, vertex_mask, hand_state, show_mesh)
        for i in range(len(vertex_mask)):
            checkbox = gui.Checkbox(f"{i:03d}")
            checkbox.checked = vertex_mask[i]
            checkbox.set_on_checked(partial(self.mano_change.checked, i=i))
            self.tab1.add_child(checkbox)
        self.mano_change.update()


class OnManoChange:
    def __init__(self, fig, pc, vertex_mask, hand_state, show_mesh):
        self.fig = fig
        self.pc = pc
        self.vertex_mask = vertex_mask
        self.hand_state = hand_state
        self.show_mesh = show_mesh
        self._all_triangles = np.copy(hand_state.hand_mesh.triangles)
        self.update()
        self.draw()

    def checked(self, is_checked, i):
        self.remove()
        self.vertex_mask[i] = is_checked
        self.update()
        self.draw()

    def remove(self):
        self.fig.main_scene.remove_geometry("vertices")
        if self.show_mesh:
            self.fig.main_scene.remove_geometry("mesh")

    def update(self):
        colors = np.zeros((len(self.vertex_mask), 3))
        colors[self.vertex_mask] = (1, 0, 0)
        self.pc.colors = o3d.utility.Vector3dVector(colors)

        if self.show_mesh:
            vertex_colors = np.array(self.hand_state.hand_mesh.vertex_colors)
            vertex_colors[self.vertex_mask] = (1, 0, 0)
            vertex_colors[np.logical_not(self.vertex_mask)] = np.array([245, 214, 175]) / 255.0
            self.hand_state.hand_mesh.vertex_colors = o3d.utility.Vector3dVector(vertex_colors)

    def draw(self):
        try:  # Open3D <= 0.13
            material = o3d.visualization.rendering.Material()
            material.shader = "defaultLit"
        except AttributeError:  # Open3d >= 0.14
            material = o3d.visualization.rendering.MaterialRecord()
            material.shader = "defaultLit"
        self.fig.main_scene.add_geometry("vertices", self.pc, material)
        self.fig.main_scene.add_geometry("mesh", self.hand_state.hand_mesh, material)


def joint_poses(pose, J, kintree_table):
    """Computes global rotation and translation of the model.

    Parameters
    ----------
    pose : array, shape (n_parts * 3)
        Hand pose parameters

    J : array, shape (n_parts, 3)
        Joint positions

    kintree_table : array, shape (2, n_parts)
        Table that describes the kinematic tree of the hand.
        kintree_table[0, i] contains the index of the parent part of part i
        and kintree_table[1, :] does not matter for the MANO model.

    Returns
    -------
    J : list
        Poses of the joints as transformation matrices
    """
    id_to_col = {kintree_table[1, i]: i
                 for i in range(kintree_table.shape[1])}
    parent = {i: id_to_col[kintree_table[0, i]]
              for i in range(1, kintree_table.shape[1])}

    results = {0: pt.transform_from(
        pr.matrix_from_compact_axis_angle(pose[0, :]), J[0, :])}
    for i in range(1, kintree_table.shape[1]):
        T = pt.transform_from(pr.matrix_from_compact_axis_angle(
            pose[i, :]), J[i, :] - J[parent[i], :])
        results[i] = results[parent[i]].dot(T)

    results = [results[i] for i in sorted(results.keys())]
    return results


POSE = np.array([
    0, 0, 0,
    -0.068, 0, 0.068 + 1,
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


def main():
    args = parse_args()

    hand_state = HandState(left=False)

    if args.config_filename is None:
        mano2hand_markers, betas = np.eye(4), np.zeros(
            hand_state.n_shape_parameters)
    else:
        mano2hand_markers, betas = load_mano_config(args.config_filename)

    if args.zero_pose:
        pose = np.zeros_like(POSE)
    else:
        pose = POSE

    hand_state.betas[:] = betas
    hand_state.recompute_shape()
    hand_state.pose[:] = pose
    hand_state.recompute_mesh()

    J = joint_poses(pose.reshape(-1, 3), hand_state.pose_parameters["J"],
                    hand_state.pose_parameters["kintree_table"])

    pc = hand_state.hand_pointcloud

    vertex_mask = np.zeros(len(pc.points), dtype=bool)
    if args.vertices:
        vertex_mask[args.vertices] = True

        colors = np.zeros((len(vertex_mask), 3))
        colors[vertex_mask] = (1, 0, 0)
        pc.colors = o3d.utility.Vector3dVector(colors)

    if args.color_fingers:
        colors = [
            (1, 0, 0),
            (1, 1, 0),
            (0, 1, 1),
            (0, 0, 1),
            (1, 0, 1),
        ]
        for finger, c in zip(MANO_CONFIG["vertex_indices_per_finger"], colors):
            kin = make_finger_kinematics(hand_state, finger)
            for index in kin.all_finger_vertex_indices:
                pc.colors[index] = c

    spheres = None
    if args.show_tips:
        vipf = MANO_CONFIG["vertex_indices_per_finger"]
        all_positions = []
        for finger in vipf:
            indices = vipf[finger]
            kin = make_finger_kinematics(hand_state, finger)
            positions = kin.forward(pose[kin.finger_pose_param_indices])
            for i, index in enumerate(indices):
                pc.colors[index] = (0, 1, 0)
                for dist in range(1, 6):
                    if index - dist >= 0:
                        pc.colors[index - dist] = [dist / 5] * 3
                    if index + dist < len(pc.colors):
                        pc.colors[index + dist] = [dist / 5] * 3
                pc.points[index] = positions[i]
            all_positions.extend(positions.tolist())

        if args.show_spheres:
            spheres = PointCollection(all_positions, s=0.006, c=(0, 1, 0))

    fig = Figure("MANO", 1920, 1080, ax_s=0.2)
    fig.add_geometry(pc)
    if not args.hide_frame:
        for i in range(len(J)):
            if i in args.joints:
                s = 0.05
            else:
                s = 0.01
            frame = pv.Frame(J[i], s=s)
            frame.add_artist(fig)
    if args.show_reference:
        coordinate_system = make_coordinate_system(s=0.2)
        fig.add_geometry(coordinate_system)
    if args.show_transforms:
        frame = pv.Frame(np.eye(4), s=0.05)
        frame.add_artist(fig)
        frame = pv.Frame(pt.invert_transform(mano2hand_markers), s=0.05)
        frame.add_artist(fig)
    if spheres is not None:
        spheres.add_artist(fig)

    if not args.hide_frame:
        coordinate_system = make_coordinate_system(s=0.2)
        try:  # Open3D <= 0.13
            material = o3d.visualization.rendering.Material()
            material.shader = "defaultLit"
        except AttributeError:  # Open3d >= 0.14
            material = o3d.visualization.rendering.MaterialRecord()
            material.shader = "defaultLit"
        fig.main_scene.add_geometry(
            "COORDINATE_SYSTEM", coordinate_system, material)

    fig.make_mano_widgets(pc, vertex_mask, hand_state, args.show_mesh)
    fig.show()


if __name__ == "__main__":
    main()
