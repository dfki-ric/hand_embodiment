"""Generate visualization of kinematic tree from URDF."""
import numpy as np
try:
    import pydot
    pydot_available = True
except ImportError:
    pydot_available = False


def write_png(self, filename, prog=None, show_visuals=False,
              show_collision_objects=False, show_inertial_frames=False,
              show_matrix=False):
    """Create PNG from dot graph of the transformations.

    .. warning::

        Note that this method requires the Python package pydot and an
        existing installation of graphviz on your system.

    Parameters
    ----------
    filename : str
        Name of the output file. Should end with '.png'.

    prog : str, optional (default: dot)
        Name of GraphViz executable that can be found in the `$PATH` or
        absolute path to GraphViz executable. Possible options are, for
        example, 'dot', 'twopi', 'neato', 'circo', 'fdp', 'sfdp'.

    show_visuals : bool, optional (default: False)
        Show visuals in graph

    show_collision_objects : bool, optional (default: False)
        Show collision objects in graph

    show_collision_objects : bool, optional (default: False)
        Show inertial frames in graph

    show_matrix : bool, optional (default: False)
        Show transformation matrix in connection
    """
    if not pydot_available:
        raise ImportError("pydot must be installed to use this feature.")

    graph = pydot.Dot(graph_type="graph")
    frame_color = "#dd3322"
    connection_color = "#d0d0ff"

    available_frames = []
    for frame in self.nodes:
        if not show_visuals and frame.startswith("visual:"):
            continue
        elif not show_collision_objects and frame.startswith("collision:"):
            continue
        elif not show_inertial_frames and frame.startswith("inertial_frame:"):
            continue

        available_frames.append(frame)
        node = pydot.Node(
            __display_name(frame), style="filled",
            fillcolor=frame_color, shape="egg")
        graph.add_node(node)
    for frames, A2B in self.transforms.items():
        a, b = frames
        if a not in available_frames or b not in available_frames:
            continue
        if show_matrix:
            connection_name = "%s to %s\n%s" % (
                __display_name(a), __display_name(b),
                str(np.round(A2B, 3)))
        else:
            connection_name = "%s to %s" % (
                __display_name(a), __display_name(b))
        node = pydot.Node(
            connection_name, style="filled", fillcolor=connection_color,
            shape="note")
        graph.add_node(node)
        a_name = __display_name(a)
        a_edge = pydot.Edge(connection_name, a_name, penwidth=3)
        graph.add_edge(a_edge)
        b_name = __display_name(b)
        b_edge = pydot.Edge(connection_name, b_name, penwidth=3)
        graph.add_edge(b_edge)

    graph.write_png(filename, prog=prog)


def __display_name(name):
    return name.replace("/", "").replace(":", "_")


