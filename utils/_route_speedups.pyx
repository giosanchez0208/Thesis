# cython: language_level=3
# cython: boundscheck=False, wraparound=False, cdivision=True, initializedcheck=False

from cpython.sequence cimport PySequence_Fast, PySequence_Fast_GET_ITEM, PySequence_Fast_GET_SIZE
from libc.math cimport fabs, sqrt
import heapq


cdef tuple _PERMUTATIONS = (
    (0, 1, 2, 3),
    (0, 1, 3, 2),
    (0, 2, 1, 3),
    (0, 2, 3, 1),
    (0, 3, 1, 2),
    (0, 3, 2, 1),
    (1, 0, 2, 3),
    (1, 0, 3, 2),
    (1, 2, 0, 3),
    (1, 2, 3, 0),
    (1, 3, 0, 2),
    (1, 3, 2, 0),
    (2, 0, 1, 3),
    (2, 0, 3, 1),
    (2, 1, 0, 3),
    (2, 1, 3, 0),
    (2, 3, 0, 1),
    (2, 3, 1, 0),
    (3, 0, 1, 2),
    (3, 0, 2, 1),
    (3, 1, 0, 2),
    (3, 1, 2, 0),
    (3, 2, 0, 1),
    (3, 2, 1, 0),
)


cdef inline double _orientation(double ax, double ay, double bx, double by, double cx, double cy):
    return (bx - ax) * (cy - ay) - (by - ay) * (cx - ax)


cdef inline bint _segments_intersect(
    double ax,
    double ay,
    double bx,
    double by,
    double cx,
    double cy,
    double dx,
    double dy,
):
    cdef double o1 = _orientation(ax, ay, bx, by, cx, cy)
    cdef double o2 = _orientation(ax, ay, bx, by, dx, dy)
    cdef double o3 = _orientation(cx, cy, dx, dy, ax, ay)
    cdef double o4 = _orientation(cx, cy, dx, dy, bx, by)
    return (o1 * o2 < 0.0) and (o3 * o4 < 0.0)


cdef inline double _shoelace_area(
    double x0,
    double y0,
    double x1,
    double y1,
    double x2,
    double y2,
    double x3,
    double y3,
    tuple order,
):
    cdef int i
    cdef int j
    cdef int idx_i
    cdef int idx_j
    cdef double area = 0.0
    cdef double xs[4]
    cdef double ys[4]
    xs[0] = x0
    xs[1] = x1
    xs[2] = x2
    xs[3] = x3
    ys[0] = y0
    ys[1] = y1
    ys[2] = y2
    ys[3] = y3
    for i in range(4):
        j = (i + 1) % 4
        idx_i = <int>order[i]
        idx_j = <int>order[j]
        area += xs[idx_i] * ys[idx_j] - ys[idx_i] * xs[idx_j]
    return fabs(area) * 0.5


cdef inline bint _is_simple_quad(
    double x0,
    double y0,
    double x1,
    double y1,
    double x2,
    double y2,
    double x3,
    double y3,
    tuple order,
):
    cdef int a = <int>order[0]
    cdef int b = <int>order[1]
    cdef int c = <int>order[2]
    cdef int d = <int>order[3]
    cdef double xs[4]
    cdef double ys[4]
    xs[0] = x0
    xs[1] = x1
    xs[2] = x2
    xs[3] = x3
    ys[0] = y0
    ys[1] = y1
    ys[2] = y2
    ys[3] = y3
    if _segments_intersect(xs[a], ys[a], xs[b], ys[b], xs[c], ys[c], xs[d], ys[d]):
        return False
    if _segments_intersect(xs[b], ys[b], xs[c], ys[c], xs[d], ys[d], xs[a], ys[a]):
        return False
    return True


def best_anchor_order(xs, ys):
    cdef object xs_seq = PySequence_Fast(xs, "best_anchor_order expects a 4-item x sequence.")
    cdef object ys_seq = PySequence_Fast(ys, "best_anchor_order expects a 4-item y sequence.")
    cdef Py_ssize_t xs_len = PySequence_Fast_GET_SIZE(xs_seq)
    cdef Py_ssize_t ys_len = PySequence_Fast_GET_SIZE(ys_seq)
    cdef double x0, x1, x2, x3, y0, y1, y2, y3
    cdef tuple best_order = ()
    cdef double best_area = -1.0
    cdef tuple order
    if xs_len != 4 or ys_len != 4:
        raise ValueError("best_anchor_order expects exactly four x/y coordinates.")
    x0 = float(<object>PySequence_Fast_GET_ITEM(xs_seq, 0))
    x1 = float(<object>PySequence_Fast_GET_ITEM(xs_seq, 1))
    x2 = float(<object>PySequence_Fast_GET_ITEM(xs_seq, 2))
    x3 = float(<object>PySequence_Fast_GET_ITEM(xs_seq, 3))
    y0 = float(<object>PySequence_Fast_GET_ITEM(ys_seq, 0))
    y1 = float(<object>PySequence_Fast_GET_ITEM(ys_seq, 1))
    y2 = float(<object>PySequence_Fast_GET_ITEM(ys_seq, 2))
    y3 = float(<object>PySequence_Fast_GET_ITEM(ys_seq, 3))
    for order in _PERMUTATIONS:
        if not _is_simple_quad(x0, y0, x1, y1, x2, y2, x3, y3, order):
            continue
        area = _shoelace_area(x0, y0, x1, y1, x2, y2, x3, y3, order)
        if area > best_area:
            best_order = order
            best_area = area
    if not best_order:
        raise ValueError("The sampled anchors did not form a valid simple quadrilateral.")
    return list(best_order), float(best_area)


def summarize_costs(costs):
    cdef object seq = PySequence_Fast(costs, "summarize_costs expects a sequence of numbers.")
    cdef Py_ssize_t i, count = PySequence_Fast_GET_SIZE(seq)
    cdef double total = 0.0
    cdef double average
    cdef double variance = 0.0
    cdef double diff
    if count == 0:
        return 0.0, 0.0, 0.0
    for i in range(count):
        total += float(<object>PySequence_Fast_GET_ITEM(seq, i))
    average = total / count
    if count > 1:
        for i in range(count):
            diff = float(<object>PySequence_Fast_GET_ITEM(seq, i)) - average
            variance += diff * diff
        variance /= count
    return float(average), float(sqrt(variance) if variance > 0.0 else 0.0), float(total)


def nearest_node_id(node_ids, lats, lons, double target_lat, double target_lon):
    cdef object nodes_seq = PySequence_Fast(node_ids, "node_ids must be a sequence")
    cdef object lat_seq = PySequence_Fast(lats, "lats must be a sequence")
    cdef object lon_seq = PySequence_Fast(lons, "lons must be a sequence")
    cdef Py_ssize_t n = PySequence_Fast_GET_SIZE(nodes_seq)
    cdef Py_ssize_t i
    cdef double best_dist = 1e300
    cdef double dlat, dlon, dist
    cdef Py_ssize_t best_idx = -1

    if PySequence_Fast_GET_SIZE(lat_seq) != n or PySequence_Fast_GET_SIZE(lon_seq) != n:
        raise ValueError("node_ids, lats, and lons must have the same length.")
    if n == 0:
        return None

    for i in range(n):
        dlat = float(<object>PySequence_Fast_GET_ITEM(lat_seq, i)) - target_lat
        dlon = float(<object>PySequence_Fast_GET_ITEM(lon_seq, i)) - target_lon
        dist = dlat * dlat + dlon * dlon
        if dist < best_dist:
            best_dist = dist
            best_idx = i

    if best_idx < 0:
        return None
    return <object>PySequence_Fast_GET_ITEM(nodes_seq, best_idx)


def shortest_path_edge_ids(adjacency_nodes, adjacency_weights, adjacency_edge_ids, start_idx, end_idx):
    cdef Py_ssize_t n = PySequence_Fast_GET_SIZE(PySequence_Fast(adjacency_nodes, "adjacency_nodes must be a sequence"))
    cdef object nodes_seq
    cdef object weights_seq
    cdef object edges_seq
    cdef Py_ssize_t node_idx, neighbor_count, i
    cdef int current_idx, neighbor_idx
    cdef double current_dist, new_dist
    cdef list distances
    cdef list prev_node
    cdef list prev_edge
    cdef list visited
    cdef list heap
    cdef object nbrs
    cdef object wts
    cdef object eids

    nodes_seq = PySequence_Fast(adjacency_nodes, "adjacency_nodes must be a sequence")
    weights_seq = PySequence_Fast(adjacency_weights, "adjacency_weights must be a sequence")
    edges_seq = PySequence_Fast(adjacency_edge_ids, "adjacency_edge_ids must be a sequence")
    if PySequence_Fast_GET_SIZE(nodes_seq) != PySequence_Fast_GET_SIZE(weights_seq) or PySequence_Fast_GET_SIZE(nodes_seq) != PySequence_Fast_GET_SIZE(edges_seq):
        raise ValueError("Adjacency sequences must have the same length.")
    if n == 0:
        return []

    start_idx = int(start_idx)
    end_idx = int(end_idx)
    if start_idx < 0 or end_idx < 0 or start_idx >= n or end_idx >= n:
        raise ValueError("Start or end node index is out of bounds.")
    if start_idx == end_idx:
        return []

    distances = [float("inf")] * n
    prev_node = [-1] * n
    prev_edge = [None] * n
    visited = [False] * n
    heap = [(0.0, start_idx)]
    distances[start_idx] = 0.0

    while heap:
        current_dist, current_idx = heapq.heappop(heap)
        if visited[current_idx]:
            continue
        visited[current_idx] = True
        if current_idx == end_idx:
            break

        nbrs = <object>PySequence_Fast_GET_ITEM(nodes_seq, current_idx)
        wts = <object>PySequence_Fast_GET_ITEM(weights_seq, current_idx)
        eids = <object>PySequence_Fast_GET_ITEM(edges_seq, current_idx)
        neighbor_count = len(nbrs)
        for i in range(neighbor_count):
            neighbor_idx = int(nbrs[i])
            new_dist = current_dist + float(wts[i])
            if new_dist < distances[neighbor_idx]:
                distances[neighbor_idx] = new_dist
                prev_node[neighbor_idx] = current_idx
                prev_edge[neighbor_idx] = eids[i]
                heapq.heappush(heap, (new_dist, neighbor_idx))

    if not visited[end_idx]:
        raise ValueError("No path found between the requested nodes.")

    path_edges = []
    current_idx = end_idx
    while current_idx != start_idx:
        edge_id = prev_edge[current_idx]
        if edge_id is None:
            raise ValueError("No path found between the requested nodes.")
        path_edges.append(edge_id)
        current_idx = prev_node[current_idx]
    path_edges.reverse()
    return path_edges
