"""Visualize extended hand model for embodiment mapping."""
import argparse
import warnings
import numpy as np
import open3d as o3d
import pytransform3d.rotations as pr
import yaml
from pytransform3d import urdf, visualizer as pv, transformations as pt
from hand_embodiment.embodiment import load_kinematic_model
from hand_embodiment.target_configurations import TARGET_CONFIG
from hand_embodiment.command_line import add_hand_argument
from hand_embodiment.metrics import CONTACT_SURFACE_VERTICES


def plot_tm(
        tm, frame, hand, show_frames=False, show_name=False, whitelist=None,
        s=1.0, highlight_visuals=(), highlight_in_directions=np.zeros((1, 3)),
        return_highlighted_mesh=False, highlight_stored_vertices=False):
    if frame not in tm.nodes:
        raise KeyError("Unknown frame '%s'" % frame)

    geometries = []
    if show_frames:
        frames = _show_frames(tm, frame, s, show_name, whitelist)
        for f in frames.values():
            geometries += f.geometries

    visuals = {}
    if hasattr(tm, "visuals"):
        visuals.update(_objects_to_artists(tm.visuals))

    _place_visuals(tm, frame, visuals)

    if highlight_stored_vertices:
        highlight_vertex_indices, highlight_vertices = _highlight_stored_vertices(
            hand, visuals, return_highlighted_mesh)
    else:
        highlight_vertex_indices, highlight_vertices = _highlight_vertices(
            visuals, highlight_in_directions, highlight_visuals,
            return_highlighted_mesh)

    for obj in visuals.values():
        geometries += obj.geometries

    if return_highlighted_mesh:
        return (geometries,
                {k: np.array(list(v)) for k, v in highlight_vertices.items()},
                {k: v for k, v in highlight_vertex_indices.items()})
    else:
        return geometries


def _show_frames(tm, frame, s, show_name, whitelist):
    nodes = list(sorted(tm._whitelisted_nodes(whitelist)))
    frames = {}
    for node in nodes:
        try:
            node2frame = tm.get_transform(node, frame)
            name = node if show_name else None
            frames[node] = pv.Frame(node2frame, name, s)
        except KeyError:
            pass  # Frame is not connected to the reference frame
    return frames


def _highlight_stored_vertices(hand, visuals, return_highlighted_mesh):
    highlight_vertex_indices = dict()
    highlight_vertices = dict()
    contact_surface_vertices_per_finger = CONTACT_SURFACE_VERTICES[hand]
    contact_surface_vertices = dict()
    for finger_name in contact_surface_vertices_per_finger:
        contact_surface_vertices.update(
            contact_surface_vertices_per_finger[finger_name])
    for visual_frame, obj in visuals.items():
        if visual_frame in contact_surface_vertices:
            mesh = obj.geometries[0]
            mesh.compute_triangle_normals()
            vertex_colors = _get_vertex_colors(mesh)
            highlighted_vertices = contact_surface_vertices[visual_frame]
            vertex_colors[highlighted_vertices] = (1, 0, 0)
            mesh.vertex_colors = o3d.utility.Vector3dVector(vertex_colors)
            vertices = np.array(mesh.vertices)[highlighted_vertices]
            if return_highlighted_mesh:
                highlight_vertex_indices[visual_frame] = highlighted_vertices
                highlight_vertices[visual_frame] = vertices
    return highlight_vertex_indices, highlight_vertices


def _highlight_vertices(
        visuals, highlight_in_directions, highlight_visuals,
        return_highlighted_mesh):
    highlight_vertices = dict()
    highlight_vertex_indices = dict()
    for visual_frame, obj in visuals.items():
        if visual_frame in highlight_visuals:
            mesh = obj.geometries[0]
            mesh.compute_triangle_normals()
            vertex_colors = _get_vertex_colors(mesh)

            vertices = np.array(mesh.vertices)
            triangles = np.array(mesh.triangles)
            triangle_normals = np.array(mesh.triangle_normals)

            highlight_vertices[visual_frame] = set()
            highlight_vertex_indices[visual_frame] = list()
            for i in range(len(vertices)):
                triangles_containing_vertex = np.where(triangles == i)[0]
                if len(triangles_containing_vertex) == 0:
                    continue
                mean_triangle_normal = pr.norm_vector(triangle_normals[triangles_containing_vertex].mean(axis=0))
                highlight_vertex = all(
                    (highlight_in_directions - vertices[np.newaxis, i]).dot(mean_triangle_normal) > 0.0)
                if highlight_vertex:
                    vertex_colors[i] = (1, 0, 0)
                    if return_highlighted_mesh:
                        highlight_vertices[visual_frame].add(tuple(vertices[i]))
                        highlight_vertex_indices[visual_frame].append(i)
            mesh.vertex_colors = o3d.utility.Vector3dVector(vertex_colors)
    return highlight_vertex_indices, highlight_vertices


def _place_visuals(tm, frame, visuals):
    for visual_frame, obj in visuals.items():
        A2B = tm.get_transform(visual_frame, frame)
        obj.set_data(A2B)


def _get_vertex_colors(mesh):
    vertex_colors = np.array(mesh.vertex_colors)
    if len(vertex_colors) == 0:
        mesh.paint_uniform_color((0, 0, 0))
        vertex_colors = np.array(mesh.vertex_colors)
    return vertex_colors


def _objects_to_artists(objects):
    """Convert geometries from URDF to artists.

    Parameters
    ----------
    objects : list of Geometry
        Objects parsed from URDF.

    Returns
    -------
    artists : dict
        Mapping from frame names to artists.
    """
    artists = {}
    for obj in objects:
        if obj.color is None:
            color = None
        else:
            # we loose the alpha channel as it is not supported by Open3D
            color = (obj.color[0], obj.color[1], obj.color[2])
        try:
            if isinstance(obj, urdf.Sphere):
                artist = pv.Sphere(radius=obj.radius, c=color)
            elif isinstance(obj, urdf.Box):
                artist = pv.Box(obj.size, c=color)
            elif isinstance(obj, urdf.Cylinder):
                artist = pv.Cylinder(obj.length, obj.radius, c=color)
            else:
                assert isinstance(obj, urdf.Mesh)
                artist = pv.Mesh(obj.filename, s=obj.scale, c=color)
            artists[obj.frame] = artist
        except RuntimeError as e:
            warnings.warn(str(e))
    return artists


def main():
    parser = argparse.ArgumentParser()
    add_hand_argument(parser)
    parser.add_argument(
        "--show-frames", action="store_true", help="Show frames.")
    parser.add_argument(
        "--highlight-meshes", action="store_true",
        help="Highlight mesh surfaces.")
    parser.add_argument(
        "--hide-tips", action="store_true",
        help="Hide finger tips use for kinematics.")
    parser.add_argument(
        "--hide-visuals", action="store_true",
        help="Hide visuals of URDF.")
    parser.add_argument(
        "--show-focus", action="store_true",
        help="Show focus point for highlighted triangles.")
    parser.add_argument(
        "--show-contact-vertices", action="store_true",
        help="Show contact vertices.")
    parser.add_argument(
        "--show-frame", action="store_true",
        help="Show base frame.")
    parser.add_argument(
        "--write-vertex-indices", action="store_true",
        help="Write indices of highlighted vertices to files.")
    parser.add_argument(
        "--highlight-stored-vertices", action="store_true",
        help="Highlight vertices used for similarity metric computation.")
    parser.add_argument(
        "--robot-config", type=str, default=None,
        help="Target system configuration file.")
    args = parser.parse_args()

    highlight_in_directions, highlight_visuals = _configure_highlights(args)

    fig = pv.figure()

    hand_config = TARGET_CONFIG[args.hand]
    # TODO move to function
    if args.robot_config is not None:
        with open(args.robot_config, "r") as f:
            hand_config_update = yaml.safe_load(f)
            if "handbase2robotbase" in hand_config_update:
                hand_config_update["handbase2robotbase"] = \
                    pt.transform_from_exponential_coordinates(
                        hand_config_update["handbase2robotbase"])
            hand_config.update(hand_config_update)
    kin = load_kinematic_model(hand_config, unscaled_visual_model=False)[0]

    if not args.hide_tips:
        for finger_name in hand_config["ee_frames"].keys():
            finger2base = kin.tm.get_transform(
                hand_config["ee_frames"][finger_name], hand_config["base_frame"])
            fig.plot_sphere(radius=0.005, A2B=finger2base, c=(1, 0, 0))
        if "intermediate_frames" in hand_config:
            for finger_name in hand_config["intermediate_frames"].keys():
                finger2base = kin.tm.get_transform(
                    hand_config["intermediate_frames"][finger_name],
                    hand_config["base_frame"])
                fig.plot_sphere(radius=0.005, A2B=finger2base, c=(1, 0, 0))

    geometries, highlighted_vertices, highlighted_vertex_indices = plot_tm(
        kin.tm, hand_config["base_frame"], args.hand,
        show_frames=args.show_frames, show_name=False, s=0.02,
        highlight_visuals=highlight_visuals,
        highlight_in_directions=highlight_in_directions,
        return_highlighted_mesh=True,
        highlight_stored_vertices=args.highlight_stored_vertices)

    if not args.hide_visuals:
        for g in geometries:
            fig.add_geometry(g)

    if args.show_contact_vertices:
        for k, v in highlighted_vertices.items():
            pc = o3d.geometry.PointCloud()
            pc.points = o3d.utility.Vector3dVector(v)
            fig.add_geometry(pc)

    if args.write_vertex_indices:
        for k, v in highlighted_vertex_indices.items():
            with open(k.replace("/", "_") + ".txt", "w") as f:
                f.write(", ".join(map(str, v)))

    if args.show_frame:
        origin = pv.Frame(np.eye(4), s=0.1)
        origin.add_artist(fig)

    if args.show_focus:
        sphere = pv.PointCollection3D(highlight_in_directions, s=0.005)
        sphere.add_artist(fig)

    fig.show()


def _configure_highlights(args):
    if args.hand == "mia":
        highlight_in_directions = np.array([[-0.03, 0.125, 0.07],
                                            [-0.03, 0.15, 0.07],
                                            [-0.03, 0.175, 0.07],
                                            [0, 0.125, 0.07],
                                            [0, 0.15, 0.07],
                                            [0, 0.175, 0.07],
                                            [0.03, 0.125, 0.07],
                                            [0.03, 0.15, 0.07],
                                            [0.03, 0.175, 0.07]])
        highlight_visuals = [
            "visual:thumb_fle/0",
            # "visual:index_sensor/0",
            "visual:index_sensor/1",
            # "visual:middle_sensor/0",
            "visual:middle_sensor/1",
            # "visual:ring_fle/0",
            "visual:ring_fle/1",
            # "visual:little_fle/0"
            "visual:little_fle/1",
        ]
    elif args.hand in ["shadow", "shadow_hand"]:
        highlight_in_directions = np.array([[0, -0.08, 0.4],
                                            [0, -0.08, 0.45],
                                            [0, -0.08, 0.5],
                                            [0.05, -0.08, 0.4],
                                            [0.05, -0.08, 0.45],
                                            [0.05, -0.08, 0.5],
                                            [-0.05, -0.08, 0.4],
                                            [-0.05, -0.08, 0.45],
                                            [-0.05, -0.08, 0.5]])
        highlight_visuals = [
            # "visual:rh_thbase/0", "visual:rh_thproximal/0",
            "visual:rh_thhub/0", "visual:rh_thmiddle/0", "visual:rh_thdistal/0",
            # "visual:rh_ffknuckle/0", "visual:rh_ffproximal/0",
            "visual:rh_ffmiddle/0", "visual:rh_ffdistal/0",
            # "visual:rh_mfknuckle/0", "visual:rh_mfproximal/0",
            "visual:rh_mfmiddle/0", "visual:rh_mfdistal/0",
            # "visual:rh_rfknuckle/0", "visual:rh_rfproximal/0",
            "visual:rh_rfmiddle/0", "visual:rh_rfdistal/0",
            # "visual:rh_lfknuckle/0", "visual:rh_lfproximal/0",
            "visual:rh_lfmiddle/0", "visual:rh_lfdistal/0"]
    elif args.hand in ["robotiq", "robotiq_2f_140"]:
        highlight_in_directions = np.array([[0.0, 0.0, 0.175]])
        highlight_visuals = ["visual:left_inner_finger_pad/0",
                             "visual:right_inner_finger_pad/0"]
    elif args.hand == "barrett":
        highlight_in_directions = np.array([
            [0.0, 0.0, 0.4],
            [-0.1, 0.0, 0.4],
            [0.1, 0.0, 0.4]])
        highlight_visuals = [  # "visual:finger_1_med_liink/0",
            "visual:finger_1_dist_link/0",
            # "visual:finger_2_med_link/0",
            "visual:finger_2_dist_link/0",
            # "visual:finger_3_med_link/0",
            "visual:finger_3_dist_link/0"]
    else:
        highlight_in_directions = np.array([])
        highlight_visuals = []
    return highlight_in_directions, highlight_visuals


if __name__ == "__main__":
    main()
