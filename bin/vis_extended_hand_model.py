import argparse
import warnings
import numpy as np
import open3d as o3d
import pytransform3d.visualizer as pv
import pytransform3d.rotations as pr
import pytransform3d.transformations as pt
from pytransform3d import urdf
from hand_embodiment.embodiment import load_kinematic_model
from hand_embodiment.target_configurations import TARGET_CONFIG


def plot_tm(fig, tm, frame, show_frames=False, show_connections=False,
            show_visuals=False, show_collision_objects=False,
            show_name=False, whitelist=None, s=1.0,
            highlight_visuals=[], highlight_in_direction=np.zeros(3)):

    if frame not in tm.nodes:
        raise KeyError("Unknown frame '%s'" % frame)

    nodes = list(sorted(tm._whitelisted_nodes(whitelist)))

    frames = {}
    if show_frames:
        for node in nodes:
            try:
                node2frame = tm.get_transform(node, frame)
                name = node if show_name else None
                frames[node] = pv.Frame(node2frame, name, s)
            except KeyError:
                pass  # Frame is not connected to the reference frame

    connections = {}
    if show_connections:
        for frame_names in tm.transforms.keys():
            from_frame, to_frame = frame_names
            if (from_frame in tm.nodes and
                    to_frame in tm.nodes):
                try:
                    tm.get_transform(from_frame, frame)
                    tm.get_transform(to_frame, frame)
                    connections[frame_names] = o3d.geometry.LineSet()
                except KeyError:
                    pass  # Frame is not connected to reference frame

    visuals = {}
    if show_visuals and hasattr(tm, "visuals"):
        visuals.update(_objects_to_artists(tm.visuals))
    collision_objects = {}
    if show_collision_objects and hasattr(
            tm, "collision_objects"):
        collision_objects.update(_objects_to_artists(tm.collision_objects))

    if show_frames:
        for node in nodes:
            try:
                node2frame = tm.get_transform(node, frame)
                frames[node].set_data(node2frame)
            except KeyError:
                pass  # Frame is not connected to the reference frame

    if show_connections:
        for frame_names in connections:
            from_frame, to_frame = frame_names
            try:
                from2ref = tm.get_transform(
                    from_frame, frame)
                to2ref = tm.get_transform(to_frame, frame)

                points = np.vstack((from2ref[:3, 3], to2ref[:3, 3]))
                connections[frame_names].points = \
                    o3d.utility.Vector3dVector(points)
                connections[frame_names].lines = \
                    o3d.utility.Vector2iVector(np.array([[0, 1]]))
            except KeyError:
                pass  # Frame is not connected to the reference frame

    for visual_frame, obj in visuals.items():
        A2B = tm.get_transform(visual_frame, frame)
        obj.set_data(A2B)
        print(visual_frame)

    for visual_frame, obj in visuals.items():
        if visual_frame in highlight_visuals:
            try:
                obj.mesh.compute_triangle_normals()
            except AttributeError:
                continue
            vertex_colors = np.array(obj.mesh.vertex_colors)
            if len(vertex_colors) == 0:
                obj.mesh.paint_uniform_color((0, 0, 0))
                vertex_colors = np.array(obj.mesh.vertex_colors)
            vertices = np.array(obj.mesh.vertices)
            triangles = np.array(obj.mesh.triangles)
            triangle_normals = np.array(obj.mesh.triangle_normals)
            for i in range(len(vertices)):
                triangle_indices = np.where(triangles == i)[0]
                if len(triangle_indices) > 0:
                    mean_norm = pr.norm_vector(triangle_normals[triangle_indices].mean(axis=0))
                    point_on_plane = vertices[i]
                    if (highlight_in_direction - point_on_plane).dot(mean_norm) > 0:
                        vertex_colors[i] = (1, 0, 0)
            obj.mesh.vertex_colors = o3d.utility.Vector3dVector(vertex_colors)

    for collision_object_frame, obj in collision_objects.items():
        A2B = tm.get_transform(collision_object_frame, frame)
        obj.set_data(A2B)

    geometries = []
    if show_frames:
        for f in frames.values():
            geometries += f.geometries
    if show_connections:
        geometries += list(connections.values())
    for obj in visuals.values():
        geometries += obj.geometries
    for obj in collision_objects.values():
        geometries += obj.geometries

    for g in geometries:
        fig.add_geometry(g)


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
    parser.add_argument(
        "hand", type=str,
        help="Hand for which we show the extended model. Possible "
             "options: 'mia', 'shadow_hand'")
    parser.add_argument(
        "--show-frames", action="store_true", help="Show frames.")
    parser.add_argument(
        "--highlight-meshes", action="store_true",
        help="Highlight mesh surfaces.")
    args = parser.parse_args()

    if args.hand == "mia":
        highlight_in_direction = np.array([0, 0.2, 0.15])
        highlight_visuals = [
            "visual:thumb_fle/0",
            "visual:index_sensor/0", "visual:index_sensor/1",
            "visual:middle_sensor/0", "visual:middle_sensor/1",
            "visual:ring_fle/0", "visual:ring_fle/1",
            "visual:little_fle/0", "visual:little_fle/1"]
    elif args.hand == "shadow_hand":
        highlight_in_direction = np.array([0, -0.1, 0.3])
        highlight_visuals = [
            "visual:rh_thbase/0", "visual:rh_thproximal/0", "visual:rh_thhub/0", "visual:rh_thmiddle/0", "visual:rh_thdistal/0",
            "visual:rh_ffknuckle/0", "visual:rh_ffproximal/0", "visual:rh_ffmiddle/0", "visual:rh_ffdistal/0",
            "visual:rh_mfknuckle/0", "visual:rh_mfproximal/0", "visual:rh_mfmiddle/0", "visual:rh_mfdistal/0",
            "visual:rh_rfknuckle/0", "visual:rh_rfproximal/0", "visual:rh_rfmiddle/0", "visual:rh_rfdistal/0",
            "visual:rh_lfknuckle/0", "visual:rh_lfproximal/0", "visual:rh_lfmiddle/0", "visual:rh_lfdistal/0"]
    else:
        raise ValueError("Hand '%s'" % args.hand)

    fig = pv.figure()

    hand_config = TARGET_CONFIG[args.hand]
    kin = load_kinematic_model(hand_config)

    for jn in kin.tm._joints:
        kin.tm.set_joint(jn, 0.05)

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

    plot_tm(
        fig, kin.tm, hand_config["base_frame"], show_frames=args.show_frames,
        show_connections=False, show_visuals=True, show_collision_objects=False,
        show_name=False, s=0.02, highlight_visuals=highlight_visuals,
        highlight_in_direction=highlight_in_direction)

    origin = pv.Frame(np.eye(4), s=0.1)
    origin.add_artist(fig)

    sphere = pv.Sphere(
        radius=0.005,
        A2B=pt.transform_from(R=np.eye(3), p=highlight_in_direction))
    sphere.add_artist(fig)

    fig.show()


if __name__ == "__main__":
    main()
