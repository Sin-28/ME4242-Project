"""
Obstacle-avoiding paths for the tip, in the robot's horizontal plane (x, z), in mm.

Each obstacle is its footprint: a convex polygon ("polygon", from its 3D points, see
Vision.detect_obstacles) or a circle ("centre", "radius"), grown by an offset per obstacle
("inflate"). Letters get a clearance so they aren't dragged; with the contact route, rigid objects
get only the arm's radius, so the shortest route wraps tightly round them: the arm lies against the
object and bends around it (with tendons only, contact is what gives the arm its second bend).
The path is the shortest route through waypoints placed around the grown footprints (a visibility
graph searched with Dijkstra).
"""

import heapq
import math

import numpy as np

CIRCLE_SIDES = 16


def convex_hull(points):
    """Convex hull of 2D points, counter-clockwise, as an N x 2 array (Andrew's monotone chain)."""
    pts = sorted(set(map(tuple, np.round(np.asarray(points, float), 3))))
    if len(pts) <= 2:
        return np.array(pts, float)

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower, upper = [], []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    for p in reversed(pts):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    return np.array(lower[:-1] + upper[:-1], float)


def footprint(o, offset=0.0):
    """An obstacle's footprint grown by `offset` mm, as a convex polygon that contains the grown shape."""
    offset = max(0.0, float(offset))
    if o.get("polygon") is not None and len(o["polygon"]) >= 3:
        base = np.asarray(o["polygon"], float)
        if offset <= 0:
            return convex_hull(base)
        k = 12                                       # Minkowski sum with a 12-gon around the offset circle
        ring = offset / math.cos(math.pi / k) * np.array(
            [[math.cos(2 * math.pi * i / k), math.sin(2 * math.pi * i / k)] for i in range(k)])
        return convex_hull((base[:, None, :] + ring[None, :, :]).reshape(-1, 2))
    centre, r = np.asarray(o["centre"], float), float(o["radius"]) + offset
    r /= math.cos(math.pi / CIRCLE_SIDES)            # polygon around the circle, not inside it
    angles = 2 * math.pi * np.arange(CIRCLE_SIDES) / CIRCLE_SIDES
    return centre + r * np.stack([np.cos(angles), np.sin(angles)], axis=1)


def inside(point, poly, eps=1e-6):
    """Is the point strictly inside the convex polygon (counter-clockwise)?"""
    if len(poly) < 3:
        return False
    p = np.asarray(point, float)
    e = np.roll(poly, -1, axis=0) - poly
    w = p - poly
    return bool(np.all(e[:, 0] * w[:, 1] - e[:, 1] * w[:, 0] > eps))


def segment_hits(a, b, poly, eps=1e-6):
    """Does the segment a-b pass through the convex polygon's interior? (separating axis test;
    touching the boundary doesn't count)"""
    if len(poly) < 3:
        return False
    a, b = np.asarray(a, float), np.asarray(b, float)
    edges = np.roll(poly, -1, axis=0) - poly
    axes = np.vstack([np.stack([-edges[:, 1], edges[:, 0]], axis=1), [[-(b - a)[1], (b - a)[0]]]])
    for n in axes:
        if not n.any():
            continue
        proj = poly @ n
        s0, s1 = a @ n, b @ n
        tol = eps * (np.abs(n).sum() + 1)
        if max(s0, s1) <= proj.min() + tol or min(s0, s1) >= proj.max() - tol:
            return False
    return True


def _point_segment(p, a, b):
    ab = b - a
    t = 0.0 if not ab.any() else float(np.clip(np.dot(p - a, ab) / np.dot(ab, ab), 0.0, 1.0))
    return float(np.linalg.norm(a + t * ab - p))


def segment_distance(a, b, poly):
    """Shortest distance from the segment a-b to the convex polygon (0 if it touches or crosses it)."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    if segment_hits(a, b, poly) or inside(a, poly) or inside(b, poly):
        return 0.0
    nxt = np.roll(poly, -1, axis=0)
    d = min(_point_segment(p, a, b) for p in poly)
    for p in (a, b):
        d = min(d, min(_point_segment(p, c, e) for c, e in zip(poly, nxt)))
    return d


def plan_path(start, goal, obstacles, clearance_mm=15.0, reachable=None):
    """Shortest path start -> goal (both (x, z)) that keeps each obstacle's offset ("inflate",
    default clearance_mm) from its footprint. If the start or goal is within an obstacle's grown
    footprint (the target sits next to something), the offset is reduced until it is outside, so
    the obstacle is still avoided as far as possible; an obstacle the start or goal lies on (the
    letter the tip is leaving, the block a letter sits on) is ignored.
    reachable(point) -> bool limits the waypoints to where the cables can take the tip.
    Returns a list of (x, z) points from start to goal, or None if no free path exists."""
    start, goal = np.asarray(start, float), np.asarray(goal, float)
    zones, corners = [], []                        # grown footprints, and waypoints just outside them
    for o in obstacles:
        offset = float(o.get("inflate", clearance_mm))
        for scale in (1.0, 0.75, 0.5, 0.25, 0.0):
            zone = footprint(o, offset * scale)
            if not (inside(start, zone) or inside(goal, zone)):
                zones.append(zone)
                corners.append(footprint(o, offset * scale + 1.0))
                break

    def free(a, b):
        return not any(segment_hits(a, b, z) for z in zones)

    if free(start, goal):
        return [start, goal]

    nodes = [start, goal]
    for ring in corners:
        for p in ring:
            if not any(inside(p, z) for z in zones) and (reachable is None or reachable(p)):
                nodes.append(p)

    # Dijkstra over the visibility graph (edges = obstacle-free straight segments)
    best = {0: 0.0}
    previous = {}
    queue = [(0.0, 0)]
    done = set()
    while queue:
        cost, i = heapq.heappop(queue)
        if i in done:
            continue
        done.add(i)
        if i == 1:
            break
        for j in range(len(nodes)):
            if j in done or not free(nodes[i], nodes[j]):
                continue
            new = cost + float(np.linalg.norm(nodes[j] - nodes[i]))
            if new < best.get(j, math.inf):
                best[j], previous[j] = new, i
                heapq.heappush(queue, (new, j))
    if 1 not in done:
        return None
    path, i = [nodes[1]], 1
    while i != 0:
        i = previous[i]
        path.append(nodes[i])
    return path[::-1]


def touched(path, obstacles, max_gap_mm):
    """Obstacles the path passes within max_gap_mm of (from their edge): where the arm lies on them."""
    return [o for o in obstacles if min_clearance(path, [o]) <= max_gap_mm]


def path_length(path):
    return float(sum(np.linalg.norm(b - a) for a, b in zip(path, path[1:])))


def densify(path, step_mm=12.0):
    """Points every <= step_mm along the path, without the start, ending exactly at the goal,
    so the closed-loop controller follows the path instead of cutting across."""
    points = []
    for a, b in zip(path, path[1:]):
        n = max(1, int(math.ceil(np.linalg.norm(b - a) / step_mm)))
        points += [a + (b - a) * (k / n) for k in range(1, n + 1)]
    return points


def min_clearance(points, obstacles):
    """Closest approach (mm) of a polyline to any obstacle's footprint (0 = touches or crosses it)."""
    if not obstacles or len(points) < 2:
        return math.inf
    return min(segment_distance(a, b, footprint(o)) for o in obstacles for a, b in zip(points, points[1:]))
