"""Manage complex chains of transformations."""
import numba
import numpy as np
import scipy.sparse as sp
from scipy.sparse import csgraph


class TransformManager(object):
    """Manage transformations between frames.

    This is a simplified version of `ROS tf <http://wiki.ros.org/tf>`_ that
    ignores the temporal aspect. A user can register transformations. The
    shortest path between all frames will be computed internally which enables
    us to provide transforms for any connected frames.

    Suppose we know the transformations A2B, D2C, and B2C. The transform
    manager can compute any transformation between the frames A, B, C and D.
    For example, you can request the transformation that represents frame D in
    frame A. The transformation manager will automatically concatenate the
    transformations D2C, C2B, and B2A, where C2B and B2A are obtained by
    inverting B2C and A2B respectively.
    """
    def __init__(self):
        self.transforms = {}
        self.nodes = []

        # A pair (self.i[n], self.j[n]) represents indices of connected nodes
        self.i = []
        self.j = []
        # We have to store the index n associated to a transformation to be
        # able to remove the transformation later
        self.transform_to_ij_index = {}
        # Same information as sparse matrix
        self.connections = sp.csr_matrix((0, 0))

        # Result of shortest path algorithm:
        # distance matrix (distance is the number of transformations)
        self.dist = np.empty(0)
        self.predecessors = np.empty(0)

        self._cached_shortest_paths = {}

    def add_transform(self, from_frame, to_frame, A2B):
        """Register a transformation.

        Parameters
        ----------
        from_frame : Hashable
            Name of the frame for which the transformation is added in the
            to_frame coordinate system

        to_frame : Hashable
            Name of the frame in which the transformation is defined

        A2B : array-like, shape (4, 4)
            Homogeneous matrix that represents the transformation from
            'from_frame' to 'to_frame'

        Returns
        -------
        self : TransformManager
            This object for chaining
        """
        if from_frame not in self.nodes:
            self.nodes.append(from_frame)
        if to_frame not in self.nodes:
            self.nodes.append(to_frame)

        transform_key = (from_frame, to_frame)

        recompute_shortest_path = False
        if transform_key not in self.transforms:
            ij_index = len(self.i)
            self.i.append(self.nodes.index(from_frame))
            self.j.append(self.nodes.index(to_frame))
            self.transform_to_ij_index[transform_key] = ij_index
            recompute_shortest_path = True

        self.transforms[transform_key] = A2B

        if recompute_shortest_path:
            self._recompute_shortest_path()

        return self

    def remove_transform(self, from_frame, to_frame):
        """Remove a transformation.

        Nothing happens if there is no such transformation.

        Parameters
        ----------
        from_frame : Hashable
            Name of the frame for which the transformation is added in the
            to_frame coordinate system

        to_frame : Hashable
            Name of the frame in which the transformation is defined

        Returns
        -------
        self : TransformManager
            This object for chaining
        """
        transform_key = (from_frame, to_frame)
        if transform_key in self.transforms:
            del self.transforms[transform_key]
            ij_index = self.transform_to_ij_index[transform_key]
            del self.transform_to_ij_index[transform_key]
            del self.i[ij_index]
            del self.j[ij_index]
            self._recompute_shortest_path()
        return self

    def _recompute_shortest_path(self):
        n_nodes = len(self.nodes)
        self.connections = sp.csr_matrix(
            (np.zeros(len(self.i)), (self.i, self.j)),
            shape=(n_nodes, n_nodes))
        self.dist, self.predecessors = csgraph.shortest_path(
            self.connections, unweighted=True, directed=False, method="D",
            return_predecessors=True)
        self._cached_shortest_paths.clear()

    def has_frame(self, frame):
        """Check if frame has been registered.

        Parameters
        ----------
        frame : Hashable
            Frame name

        Returns
        -------
        has_frame : bool
            Frame is registered
        """
        return frame in self.nodes

    def get_transform(self, from_frame, to_frame):
        """Request a transformation.

        Parameters
        ----------
        from_frame : Hashable
            Name of the frame for which the transformation is requested in the
            to_frame coordinate system

        to_frame : Hashable
            Name of the frame in which the transformation is defined

        Returns
        -------
        A2B : array-like, shape (4, 4)
            Homogeneous matrix that represents the transformation from
            'from_frame' to 'to_frame'

        Raises
        ------
        KeyError
            If one of the frames is unknown or there is no connection between
            them
        """
        if (from_frame, to_frame) in self.transforms:
            return self.transforms[(from_frame, to_frame)]

        if (to_frame, from_frame) in self.transforms:
            return invert_transform(
                self.transforms[(to_frame, from_frame)])

        i = self.nodes.index(from_frame)
        j = self.nodes.index(to_frame)
        if not np.isfinite(self.dist[i, j]):
            raise KeyError("Cannot compute path from frame '%s' to "
                           "frame '%s'." % (from_frame, to_frame))

        path = self._shortest_path(i, j)
        return self._path_transform(path)

    def _shortest_path(self, i, j):
        if (i, j) in self._cached_shortest_paths:
            return self._cached_shortest_paths[(i, j)]
        path = _shortest_path(self.nodes, self.predecessors, i, j)
        self._cached_shortest_paths[(i, j)] = path
        return path

    def _path_transform(self, path):
        A2B = np.eye(4)
        for from_f, to_f in zip(path[:-1], path[1:]):
            A2B = np.dot(self.get_transform(from_f, to_f), A2B)
        return A2B

    def _whitelisted_nodes(self, whitelist):
        """Get whitelisted nodes.

        Parameters
        ----------
        whitelist : list or None
            Whitelist of frames

        Returns
        -------
        nodes : set
            Existing whitelisted nodes

        Raises
        ------
        KeyError
            Will be raised if an unknown node is in the whitelist.
        """
        nodes = set(self.nodes)
        if whitelist is not None:
            whitelist = set(whitelist)
            nodes = nodes.intersection(whitelist)
            nonwhitlisted_nodes = whitelist.difference(nodes)
            if nonwhitlisted_nodes:
                raise KeyError("Whitelist contains unknown nodes: '%s'"
                               % nonwhitlisted_nodes)
        return nodes


@numba.njit(cache=True)
def _shortest_path(nodes, predecessors, i, j):
    path = []
    k = i
    while k != -9999:
        path.append(nodes[k])
        k = predecessors[j, k]
    return path


@numba.njit(numba.float64[:, :](numba.float64[:, :]), cache=True)
def invert_transform(A2B):
    """Invert transform.
    Parameters
    ----------
    A2B : array-like, shape (4, 4)
        Transform from frame A to frame B
    Returns
    -------
    B2A : array-like, shape (4, 4)
        Transform from frame B to frame A
    """
    B2A = np.empty((4, 4))
    RT = A2B[:3, :3].T
    B2A[:3, :3] = RT
    B2A[:3, 3] = -np.dot(RT, A2B[:3, 3])
    B2A[3, :3] = 0.0
    B2A[3, 3] = 1.0
    return B2A
