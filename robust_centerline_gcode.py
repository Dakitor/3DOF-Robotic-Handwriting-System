#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Centerline reconstruction + continuous junction tracking + stroke semantic merge + stroke-order planning
for single Chinese character PNG -> CNC/GRBL style single-line G-code.
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import networkx as nx
import numpy as np
from scipy import interpolate
from scipy.ndimage import gaussian_filter
from skimage.morphology import medial_axis

Point = Tuple[float, float]  # (x, y)
Node = Tuple[int, int]  # (x, y)


@dataclass
class Params:
    min_branch_length: float = 5.0
    min_component_area: int = 20
    merge_angle_threshold: float = 38.0
    simplify_epsilon: float = 1.5
    smooth_factor: float = 2.0

    # G-code params
    scale: float = 1.0
    z_up: float = 5.0
    z_down: float = 0.0
    feed_rate: float = 1800.0
    rapid_rate: float = 4000.0


@dataclass
class Segment:
    seg_id: int
    points: List[Point]
    start: Node
    end: Node
    start_degree: int
    end_degree: int
    component_id: int


@dataclass
class StrokeInfo:
    points: List[Point]
    bbox: Tuple[float, float, float, float]  # xmin, ymin, xmax, ymax
    center: Tuple[float, float]
    length: float
    stroke_type: str


def imread_unicode(path: Path, flags: int = cv2.IMREAD_GRAYSCALE) -> np.ndarray:
    data = np.fromfile(str(path), dtype=np.uint8)
    img = cv2.imdecode(data, flags)
    if img is None:
        raise ValueError(f"Failed to read image: {path}")
    return img


def imwrite_unicode(path: Path, img: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ext = path.suffix.lower() if path.suffix else ".png"
    ok, buf = cv2.imencode(ext, img)
    if not ok:
        raise ValueError(f"Failed to encode image for writing: {path}")
    buf.tofile(str(path))


def ensure_fg_white(gray: np.ndarray) -> np.ndarray:
    _, b1 = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    b2 = cv2.bitwise_not(b1)

    def score(bin_img: np.ndarray) -> float:
        fg_ratio = float(np.mean(bin_img > 0))
        if fg_ratio <= 1e-6 or fg_ratio >= 0.95:
            return 1e9
        n, _, stats, _ = cv2.connectedComponentsWithStats(bin_img, connectivity=8)
        tiny = 0
        for i in range(1, n):
            if stats[i, cv2.CC_STAT_AREA] < 8:
                tiny += 1
        return abs(fg_ratio - 0.22) * 1000.0 + tiny

    return b1 if score(b1) <= score(b2) else b2


def remove_small_components(bin_fg: np.ndarray, min_component_area: int) -> np.ndarray:
    n, labels, stats, _ = cv2.connectedComponentsWithStats(bin_fg, connectivity=8)
    out = np.zeros_like(bin_fg)
    for i in range(1, n):
        if int(stats[i, cv2.CC_STAT_AREA]) >= min_component_area:
            out[labels == i] = 255
    return out


def preprocess_image(image_path: Path, min_component_area: int) -> np.ndarray:
    gray = imread_unicode(image_path, cv2.IMREAD_GRAYSCALE)
    denoise = cv2.GaussianBlur(gray, (3, 3), 0.8)
    bin_fg = ensure_fg_white(denoise)

    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    bin_fg = cv2.morphologyEx(bin_fg, cv2.MORPH_CLOSE, k, iterations=1)
    bin_fg = cv2.morphologyEx(bin_fg, cv2.MORPH_OPEN, k, iterations=1)

    bin_fg = remove_small_components(bin_fg, min_component_area)
    return (bin_fg > 0).astype(np.uint8)


def build_width_map(bin_fg01: np.ndarray) -> np.ndarray:
    # Stroke half-width estimate from foreground distance transform.
    src = (bin_fg01.astype(np.uint8) * 255)
    dist = cv2.distanceTransform(src, cv2.DIST_L2, 3)
    return dist.astype(np.float32)


def zhang_suen_thinning(bin01: np.ndarray) -> np.ndarray:
    img = (bin01 > 0).astype(np.uint8).copy()
    changed = True
    while changed:
        changed = False
        for step in (0, 1):
            to_remove = []
            h, w = img.shape
            for y in range(1, h - 1):
                for x in range(1, w - 1):
                    if img[y, x] == 0:
                        continue
                    p2, p3, p4 = img[y - 1, x], img[y - 1, x + 1], img[y, x + 1]
                    p5, p6, p7 = img[y + 1, x + 1], img[y + 1, x], img[y + 1, x - 1]
                    p8, p9 = img[y, x - 1], img[y - 1, x - 1]
                    ns = [p2, p3, p4, p5, p6, p7, p8, p9]
                    n = sum(ns)
                    if n < 2 or n > 6:
                        continue
                    a = sum((ns[i] == 0 and ns[(i + 1) % 8] == 1) for i in range(8))
                    if a != 1:
                        continue
                    if step == 0:
                        c1 = p2 * p4 * p6 == 0
                        c2 = p4 * p6 * p8 == 0
                    else:
                        c1 = p2 * p4 * p8 == 0
                        c2 = p2 * p6 * p8 == 0
                    if c1 and c2:
                        to_remove.append((y, x))
            if to_remove:
                changed = True
                for y, x in to_remove:
                    img[y, x] = 0
    return img


def extract_centerline(bin_fg01: np.ndarray) -> np.ndarray:
    # Main strategy: medial-axis ridge (distance-transform aware), more stable than pure thinning.
    med, dist = medial_axis(bin_fg01.astype(bool), return_distance=True)

    # Keep stronger ridge points to suppress burr-like pseudo-branches.
    dist_s = gaussian_filter(dist.astype(np.float32), sigma=0.8)
    if np.any(dist_s > 0):
        th = np.percentile(dist_s[dist_s > 0], 35)
        ridge_conf = (dist_s >= 0.9) | (dist_s >= th)
    else:
        ridge_conf = dist_s > 0
    center = (med & ridge_conf).astype(np.uint8)

    # Safety fallback.
    if np.count_nonzero(center) < max(8, int(np.count_nonzero(bin_fg01) * 0.002)):
        center = zhang_suen_thinning(bin_fg01)

    center = zhang_suen_thinning(center)
    return center


def neighbors8(x: int, y: int, w: int, h: int) -> Iterable[Tuple[int, int]]:
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dx == 0 and dy == 0:
                continue
            nx_, ny_ = x + dx, y + dy
            if 0 <= nx_ < w and 0 <= ny_ < h:
                yield nx_, ny_


def skeleton_to_graph(skel01: np.ndarray) -> nx.Graph:
    h, w = skel01.shape
    g = nx.Graph()
    ys, xs = np.where(skel01 > 0)
    for y, x in zip(ys, xs):
        g.add_node((int(x), int(y)))
    for y, x in zip(ys, xs):
        p = (int(x), int(y))
        for nx_, ny_ in neighbors8(x, y, w, h):
            if skel01[ny_, nx_] > 0:
                q = (int(nx_), int(ny_))
                if p < q:
                    g.add_edge(p, q, weight=math.hypot(q[0] - p[0], q[1] - p[1]))
    return g


def graph_to_skeleton(g: nx.Graph, shape: Tuple[int, int]) -> np.ndarray:
    out = np.zeros(shape, dtype=np.uint8)
    for x, y in g.nodes:
        out[y, x] = 1
    return out


def walk_branch_from_endpoint(g: nx.Graph, endpoint: Node, max_len: float) -> Tuple[List[Node], float, Node]:
    path = [endpoint]
    prev: Optional[Node] = None
    cur = endpoint
    total = 0.0

    while True:
        nbrs = list(g.neighbors(cur))
        if prev is not None:
            nbrs = [n for n in nbrs if n != prev]
        if len(nbrs) != 1:
            return path, total, cur
        nxt = nbrs[0]
        total += g[cur][nxt].get("weight", 1.0)
        path.append(nxt)
        prev, cur = cur, nxt
        if g.degree(cur) != 2 or total > max_len:
            return path, total, cur


def prune_spurs(centerline01: np.ndarray, min_branch_length: float) -> np.ndarray:
    g = skeleton_to_graph(centerline01)
    changed = True

    while changed:
        changed = False
        for ep in [n for n in g.nodes if g.degree(n) == 1]:
            if ep not in g:
                continue
            path, length, stop = walk_branch_from_endpoint(g, ep, min_branch_length + 1.0)
            if length < min_branch_length:
                removable = path[:-1] if g.degree(stop) >= 3 else path
                if removable:
                    g.remove_nodes_from(removable)
                    changed = True

        for comp in list(nx.connected_components(g)):
            sub = g.subgraph(comp)
            clen = sum(d.get("weight", 1.0) for _, _, d in sub.edges(data=True))
            if clen < min_branch_length:
                g.remove_nodes_from(list(comp))
                changed = True

    return graph_to_skeleton(g, centerline01.shape)


def rdp(points: List[Point], epsilon: float) -> List[Point]:
    if len(points) < 3:
        return points
    p0 = np.array(points[0], dtype=np.float32)
    p1 = np.array(points[-1], dtype=np.float32)
    line = p1 - p0
    n = np.linalg.norm(line)

    dists: List[float] = []
    if n < 1e-8:
        for p in points:
            dists.append(float(np.linalg.norm(np.array(p, dtype=np.float32) - p0)))
    else:
        denom = n * n
        for p in points:
            pv = np.array(p, dtype=np.float32) - p0
            t = np.dot(pv, line) / denom
            foot = p0 + t * line
            dists.append(float(np.linalg.norm(np.array(p, dtype=np.float32) - foot)))

    idx = int(np.argmax(dists))
    if dists[idx] > epsilon:
        left = rdp(points[: idx + 1], epsilon)
        right = rdp(points[idx:], epsilon)
        return left[:-1] + right
    return [points[0], points[-1]]


def extract_path_between_keys(g: nx.Graph, start: Node, next_node: Node, key_nodes: set, used: set) -> List[Node]:
    path = [start, next_node]
    used.add(frozenset((start, next_node)))
    prev, cur = start, next_node
    while cur not in key_nodes:
        nxts = [n for n in g.neighbors(cur) if n != prev]
        if not nxts:
            break
        nxt = nxts[0]
        e = frozenset((cur, nxt))
        if e in used:
            break
        used.add(e)
        path.append(nxt)
        prev, cur = cur, nxt
    return path


def cycle_from_component(g_sub: nx.Graph) -> List[Node]:
    s = next(iter(g_sub.nodes))
    path = [s]
    prev = None
    cur = s
    for _ in range(len(g_sub.nodes) + 3):
        nxts = [n for n in g_sub.neighbors(cur) if n != prev]
        if not nxts:
            break
        nxt = nxts[0]
        if nxt == s:
            path.append(nxt)
            break
        path.append(nxt)
        prev, cur = cur, nxt
    return path


def simplify_graph(centerline01: np.ndarray, simplify_epsilon: float) -> Tuple[List[Segment], nx.MultiGraph]:
    # Pixel skeleton -> polyline segment graph (junction-aware compressed graph)
    g = skeleton_to_graph(centerline01)
    segments: List[Segment] = []
    seg_graph = nx.MultiGraph()

    sid = 0
    for comp_id, comp_nodes in enumerate(nx.connected_components(g)):
        g_sub = g.subgraph(comp_nodes).copy()
        deg = dict(g_sub.degree())
        key_nodes = {n for n, d in deg.items() if d != 2}
        used = set()

        if len(key_nodes) == 0:
            cyc = cycle_from_component(g_sub)
            pts = rdp([(float(x), float(y)) for x, y in cyc], simplify_epsilon)
            if len(pts) >= 2:
                s = (int(round(pts[0][0])), int(round(pts[0][1])))
                e = (int(round(pts[-1][0])), int(round(pts[-1][1])))
                segments.append(Segment(sid, pts, s, e, 2, 2, comp_id))
                sid += 1
            continue

        for k in key_nodes:
            for nbr in g_sub.neighbors(k):
                edge = frozenset((k, nbr))
                if edge in used:
                    continue
                npth = extract_path_between_keys(g_sub, k, nbr, key_nodes, used)
                pts = rdp([(float(x), float(y)) for x, y in npth], simplify_epsilon)
                if len(pts) < 2:
                    continue
                s = (int(round(pts[0][0])), int(round(pts[0][1])))
                e = (int(round(pts[-1][0])), int(round(pts[-1][1])))
                segments.append(Segment(sid, pts, s, e, deg.get(npth[0], 1), deg.get(npth[-1], 1), comp_id))
                sid += 1

    for seg in segments:
        seg_graph.add_node(seg.start)
        seg_graph.add_node(seg.end)
        seg_graph.add_edge(seg.start, seg.end, key=seg.seg_id, seg_id=seg.seg_id, weight=polyline_length(seg.points))

    return segments, seg_graph


def polyline_length(points: Sequence[Point]) -> float:
    if len(points) < 2:
        return 0.0
    return float(sum(math.hypot(points[i][0] - points[i - 1][0], points[i][1] - points[i - 1][1]) for i in range(1, len(points))))


def direction_at_end(points: Sequence[Point], at_start: bool) -> np.ndarray:
    if len(points) < 2:
        return np.array([0.0, 0.0], dtype=np.float32)
    if at_start:
        v = np.array(points[1], dtype=np.float32) - np.array(points[0], dtype=np.float32)
    else:
        v = np.array(points[-1], dtype=np.float32) - np.array(points[-2], dtype=np.float32)
    n = np.linalg.norm(v)
    if n < 1e-6:
        return np.array([0.0, 0.0], dtype=np.float32)
    return v / n


def angle_deg(v1: np.ndarray, v2: np.ndarray) -> float:
    n1 = np.linalg.norm(v1)
    n2 = np.linalg.norm(v2)
    if n1 < 1e-6 or n2 < 1e-6:
        return 180.0
    c = float(np.clip(np.dot(v1 / n1, v2 / n2), -1.0, 1.0))
    return float(math.degrees(math.acos(c)))


def orient_segment_for_node(seg: Segment, node: Node) -> List[Point]:
    if seg.start == node:
        return seg.points
    if seg.end == node:
        return list(reversed(seg.points))
    return seg.points


def orient_segment_to_node(seg: Segment, node: Node) -> List[Point]:
    # Return polyline ending at node.
    if seg.end == node:
        return seg.points
    if seg.start == node:
        return list(reversed(seg.points))
    return seg.points


def branch_features(seg: Segment, node: Node, width_map: np.ndarray) -> Dict[str, float]:
    pts = orient_segment_for_node(seg, node)
    if len(pts) < 2:
        return {
            "length": 0.0,
            "curv": 180.0,
            "w": 0.0,
            "jx": float(node[0]),
            "jy": float(node[1]),
            "d1x": 0.0,
            "d1y": 0.0,
            "d2x": 0.0,
            "d2y": 0.0,
        }

    k1 = min(3, len(pts) - 1)
    k2 = min(8, len(pts) - 1)
    p0 = np.array(pts[0], dtype=np.float32)
    p1 = np.array(pts[k1], dtype=np.float32)
    p2 = np.array(pts[k2], dtype=np.float32)
    d1 = p1 - p0
    d2 = p2 - p0
    n1 = float(np.linalg.norm(d1))
    n2 = float(np.linalg.norm(d2))
    d1 = d1 / max(n1, 1e-6)
    d2 = d2 / max(n2, 1e-6)
    curv = angle_deg(d1, d2)

    ws = []
    for p in pts[: min(5, len(pts))]:
        x = int(np.clip(round(p[0]), 0, width_map.shape[1] - 1))
        y = int(np.clip(round(p[1]), 0, width_map.shape[0] - 1))
        ws.append(float(width_map[y, x] * 2.0))

    return {
        "length": polyline_length(pts),
        "curv": curv,
        "w": float(np.mean(ws)) if ws else 0.0,
        "jx": float(node[0]),
        "jy": float(node[1]),
        "d1x": float(d1[0]),
        "d1y": float(d1[1]),
        "d2x": float(d2[0]),
        "d2y": float(d2[1]),
    }


def continuity_score(
    feat_a: Dict[str, float],
    feat_b: Dict[str, float],
    angle_threshold: float,
    min_pair_length: float,
) -> Tuple[float, Dict[str, float]]:
    d1a = np.array([feat_a["d1x"], feat_a["d1y"]], dtype=np.float32)
    d1b = np.array([feat_b["d1x"], feat_b["d1y"]], dtype=np.float32)
    d2a = np.array([feat_a["d2x"], feat_a["d2y"]], dtype=np.float32)
    d2b = np.array([feat_b["d2x"], feat_b["d2y"]], dtype=np.float32)

    # Straight continuation across junction means two outgoing directions are opposite.
    turn_local = abs(180.0 - angle_deg(d1a, d1b))
    turn_far = abs(180.0 - angle_deg(d2a, d2b))

    direction_cont = max(0.0, 1.0 - turn_local / max(angle_threshold, 1.0))
    collinear = max(0.0, 1.0 - turn_far / max(angle_threshold * 1.25, 1.0))

    curv_diff = abs(feat_a["curv"] - feat_b["curv"])
    curvature_smooth = max(0.0, 1.0 - curv_diff / 60.0)

    min_len = min(feat_a["length"], feat_b["length"])
    length_score = min(1.0, min_len / max(min_pair_length, 1e-6))

    w1 = max(feat_a["w"], 1e-6)
    w2 = max(feat_b["w"], 1e-6)
    width_cons = max(0.0, 1.0 - abs(w1 - w2) / max(w1, w2))

    total = (
        0.36 * direction_cont
        + 0.24 * collinear
        + 0.16 * curvature_smooth
        + 0.12 * length_score
        + 0.12 * width_cons
    )
    sub = {
        "direction": direction_cont,
        "collinear": collinear,
        "curvature": curvature_smooth,
        "length": length_score,
        "width": width_cons,
        "turn_local": turn_local,
        "turn_far": turn_far,
        "total": total,
    }
    return total, sub


def build_junction_pairings(
    segments: List[Segment],
    seg_graph: nx.MultiGraph,
    width_map: np.ndarray,
    merge_angle_threshold: float,
    min_pair_length: float,
) -> Tuple[Dict[Tuple[int, Node], Tuple[int, Node]], List[Dict[str, object]], set]:
    by_id = {s.seg_id: s for s in segments}
    node_to_seg: Dict[Node, List[int]] = {}
    for s in segments:
        node_to_seg.setdefault(s.start, []).append(s.seg_id)
        node_to_seg.setdefault(s.end, []).append(s.seg_id)

    pair_map: Dict[Tuple[int, Node], Tuple[int, Node]] = {}
    pairing_debug: List[Dict[str, object]] = []
    junction_nodes = {n for n in seg_graph.nodes if seg_graph.degree(n) >= 3}

    for node in sorted(junction_nodes):
        cands = node_to_seg.get(node, [])
        if len(cands) < 2:
            continue

        feats = {sid: branch_features(by_id[sid], node, width_map) for sid in cands}
        g_match = nx.Graph()
        g_match.add_nodes_from(cands)
        details: Dict[Tuple[int, int], Dict[str, float]] = {}

        for i in range(len(cands)):
            for j in range(i + 1, len(cands)):
                sid_a = cands[i]
                sid_b = cands[j]
                score, sub = continuity_score(feats[sid_a], feats[sid_b], merge_angle_threshold, min_pair_length)
                # Hard gate: orientation and minimum confidence
                if sub["turn_local"] > merge_angle_threshold:
                    continue
                if score < 0.35:
                    continue
                g_match.add_edge(sid_a, sid_b, weight=score)
                details[(sid_a, sid_b)] = sub

        matched = nx.algorithms.matching.max_weight_matching(g_match, maxcardinality=False)
        for a, b in matched:
            sid_a, sid_b = (a, b) if a < b else (b, a)
            pair_map[(sid_a, node)] = (sid_b, node)
            pair_map[(sid_b, node)] = (sid_a, node)
            sub = details.get((sid_a, sid_b), {})
            pairing_debug.append(
                {
                    "node": node,
                    "sid_a": sid_a,
                    "sid_b": sid_b,
                    "score": float(sub.get("total", 0.0)),
                    "turn_local": float(sub.get("turn_local", 180.0)),
                    "turn_far": float(sub.get("turn_far", 180.0)),
                    "direction": float(sub.get("direction", 0.0)),
                    "collinear": float(sub.get("collinear", 0.0)),
                    "curvature": float(sub.get("curvature", 0.0)),
                    "length": float(sub.get("length", 0.0)),
                    "width": float(sub.get("width", 0.0)),
                }
            )

    return pair_map, pairing_debug, junction_nodes


def merge_segments_into_strokes(
    segments: List[Segment],
    seg_graph: nx.MultiGraph,
    width_map: np.ndarray,
    merge_angle_threshold: float,
    min_pair_length: float,
) -> Tuple[List[List[Point]], List[Dict[str, object]], set]:
    # Junction continuity reconstruction: local optimal pairings, then follow continuity through junctions.
    if not segments:
        return [], [], set()

    by_id = {s.seg_id: s for s in segments}
    pair_map, pairing_debug, junction_nodes = build_junction_pairings(
        segments,
        seg_graph,
        width_map,
        merge_angle_threshold,
        min_pair_length,
    )

    used: set = set()
    strokes: List[List[Point]] = []

    for seg in sorted(segments, key=lambda s: polyline_length(s.points), reverse=True):
        if seg.seg_id in used:
            continue
        used.add(seg.seg_id)
        chain = list(seg.points)

        # Extend forward from seed end.
        cur_sid = seg.seg_id
        cur_node = seg.end
        while True:
            key = (cur_sid, cur_node)
            if key not in pair_map:
                break
            nxt_sid, _ = pair_map[key]
            if nxt_sid in used:
                break
            nxt = by_id[nxt_sid]
            nxt_pts = orient_segment_for_node(nxt, cur_node)
            chain.extend(nxt_pts[1:])
            used.add(nxt_sid)
            cur_sid = nxt_sid
            cur_node = nxt.end if nxt.start == cur_node else nxt.start

        # Extend backward from seed start.
        cur_sid = seg.seg_id
        cur_node = seg.start
        while True:
            key = (cur_sid, cur_node)
            if key not in pair_map:
                break
            nxt_sid, _ = pair_map[key]
            if nxt_sid in used:
                break
            nxt = by_id[nxt_sid]
            nxt_pts = orient_segment_for_node(nxt, cur_node)  # node -> outward
            chain = list(reversed(nxt_pts[1:])) + chain
            used.add(nxt_sid)
            cur_sid = nxt_sid
            cur_node = nxt.end if nxt.start == cur_node else nxt.start

        if len(chain) >= 2:
            strokes.append(chain)

    return strokes, pairing_debug, junction_nodes


def polyline_linearity(points: List[Point]) -> Tuple[float, float]:
    # linearity_ratio close to 1 means near-straight; max_dev measures bending in pixel units.
    if len(points) < 3:
        return 1.0, 0.0
    arr = np.array(points, dtype=np.float32)
    p0 = arr[0]
    p1 = arr[-1]
    chord = float(np.linalg.norm(p1 - p0))
    length = max(polyline_length(points), 1e-6)
    ratio = chord / length
    if chord < 1e-6:
        dev = float(np.max(np.linalg.norm(arr - p0, axis=1)))
        return ratio, dev

    line = p1 - p0
    denom = float(np.dot(line, line))
    max_dev = 0.0
    for p in arr:
        t = float(np.dot(p - p0, line) / denom)
        foot = p0 + t * line
        d = float(np.linalg.norm(p - foot))
        if d > max_dev:
            max_dev = d
    return ratio, max_dev


def simplify_spacing(points: List[Point], min_step: float = 0.9) -> List[Point]:
    if not points:
        return points
    out = [points[0]]
    for p in points[1:]:
        if math.hypot(p[0] - out[-1][0], p[1] - out[-1][1]) >= min_step:
            out.append(p)
    if len(out) == 1 and len(points) > 1:
        out.append(points[-1])
    return out


def smooth_piece(points: List[Point], smooth_factor: float, simplify_epsilon: float) -> List[Point]:
    if len(points) < 3:
        return points

    ratio, max_dev = polyline_linearity(points)
    # High-linearity segments are preserved as line/polyline: no spline smoothing.
    if ratio >= 0.985 and max_dev <= 1.2:
        return rdp(points, simplify_epsilon)

    arr = np.array(points, dtype=np.float32)
    if len(arr) < 5:
        return rdp(points, simplify_epsilon)

    # Light smoothing only for clearly curved pieces.
    try:
        tck, _ = interpolate.splprep([arr[:, 0], arr[:, 1]], s=max(0.4, smooth_factor * 0.28) * len(arr), k=min(3, len(arr) - 1))
        u = np.linspace(0.0, 1.0, max(8, int(len(arr) * 0.7)))
        x_new, y_new = interpolate.splev(u, tck)
        sm = list(zip(map(float, x_new), map(float, y_new)))
    except Exception:
        sm = points
    return rdp(sm, simplify_epsilon)


def smooth_strokes(strokes: List[List[Point]], smooth_factor: float, simplify_epsilon: float, junction_nodes: Optional[set] = None) -> List[List[Point]]:
    out: List[List[Point]] = []
    junction_nodes = junction_nodes or set()

    for pts in strokes:
        if len(pts) < 2:
            continue
        pts = simplify_spacing(rdp(pts, simplify_epsilon * 0.7), min_step=0.6)
        if len(pts) < 2:
            continue

        # Hard breakpoint at junction: never smooth across a junction interior point.
        break_idxs = []
        for i in range(1, len(pts) - 1):
            n = (int(round(pts[i][0])), int(round(pts[i][1])))
            if n in junction_nodes:
                break_idxs.append(i)

        split_points: List[List[Point]] = []
        last = 0
        for bi in break_idxs:
            piece = pts[last : bi + 1]
            if len(piece) >= 2:
                split_points.append(piece)
            last = bi
        tail = pts[last:]
        if len(tail) >= 2:
            split_points.append(tail)
        if not split_points:
            split_points = [pts]

        merged_piece: List[Point] = []
        for j, piece in enumerate(split_points):
            sp = smooth_piece(piece, smooth_factor, simplify_epsilon)
            sp = simplify_spacing(sp, min_step=0.9)
            if len(sp) < 2:
                continue
            if j == 0:
                merged_piece.extend(sp)
            else:
                merged_piece.extend(sp[1:])

        if len(merged_piece) >= 2:
            out.append(merged_piece)
    return out


def stroke_bbox(points: List[Point]) -> Tuple[float, float, float, float]:
    arr = np.array(points, dtype=np.float32)
    return float(np.min(arr[:, 0])), float(np.min(arr[:, 1])), float(np.max(arr[:, 0])), float(np.max(arr[:, 1]))


def stroke_type(points: List[Point]) -> str:
    if len(points) < 2:
        return "other"
    length = polyline_length(points)
    p0 = np.array(points[0], dtype=np.float32)
    p1 = np.array(points[-1], dtype=np.float32)
    dx, dy = float(p1[0] - p0[0]), float(p1[1] - p0[1])
    if length < 8.0:
        return "dot"
    if abs(dx) >= abs(dy) * 1.45:
        return "heng"
    if abs(dy) >= abs(dx) * 1.45:
        return "shu"
    if dx < 0 and dy > 0:
        return "pie"
    if dx > 0 and dy > 0:
        return "na"
    if abs(dx) < 0.35 * abs(dy):
        return "shu"
    return "other"


def build_stroke_info(strokes: List[List[Point]]) -> List[StrokeInfo]:
    infos: List[StrokeInfo] = []
    for s in strokes:
        bbox = stroke_bbox(s)
        cx = (bbox[0] + bbox[2]) * 0.5
        cy = (bbox[1] + bbox[3]) * 0.5
        infos.append(StrokeInfo(s, bbox, (cx, cy), polyline_length(s), stroke_type(s)))
    return infos


def gap_partition(values: List[float], total_span: float, min_ratio: float = 0.13) -> List[int]:
    if len(values) <= 1:
        return [0] * len(values)
    sorted_idx = np.argsort(values)
    sorted_vals = [values[i] for i in sorted_idx]
    gaps = [sorted_vals[i + 1] - sorted_vals[i] for i in range(len(sorted_vals) - 1)]
    if not gaps:
        return [0] * len(values)

    gap_idx = np.argsort(gaps)[::-1]
    cuts = []
    for gi in gap_idx[:2]:
        if gaps[gi] > total_span * min_ratio:
            cuts.append(gi)
    cuts = sorted(cuts)

    groups = [0] * len(values)
    gid = 0
    cut_ptr = 0
    for i in range(len(sorted_vals)):
        if cut_ptr < len(cuts) and i > cuts[cut_ptr]:
            gid += 1
            cut_ptr += 1
        groups[sorted_idx[i]] = gid
    return groups


def classify_structure(strokes: List[List[Point]]) -> str:
    if len(strokes) <= 1:
        return "single"

    infos = build_stroke_info(strokes)
    xs = [i.center[0] for i in infos]
    ys = [i.center[1] for i in infos]
    xmins = [i.bbox[0] for i in infos]
    ymins = [i.bbox[1] for i in infos]
    xmaxs = [i.bbox[2] for i in infos]
    ymaxs = [i.bbox[3] for i in infos]
    W = max(xmaxs) - min(xmins) + 1e-6
    H = max(ymaxs) - min(ymins) + 1e-6

    for i, info in enumerate(infos):
        x0, y0, x1, y1 = info.bbox
        inside = 0
        for j, other in enumerate(infos):
            if i == j:
                continue
            ox0, oy0, ox1, oy1 = other.bbox
            if ox0 >= x0 - 2 and ox1 <= x1 + 2 and oy0 >= y0 - 2 and oy1 <= y1 + 2:
                inside += 1
        if inside >= max(1, int(0.5 * (len(infos) - 1))) and (x1 - x0) > 0.55 * W and (y1 - y0) > 0.55 * H:
            return "enclosure"

    gx = gap_partition(xs, W)
    gy = gap_partition(ys, H)
    nx = len(set(gx))
    ny = len(set(gy))

    if nx >= 3 and nx >= ny:
        return "left-mid-right"
    if ny >= 3 and ny > nx:
        return "top-mid-bottom"
    if nx == 2 and W >= H * 0.75:
        return "left-right"
    if ny == 2 and H > W * 0.7:
        return "top-bottom"
    return "single"


def order_components(strokes: List[List[Point]], structure: str) -> List[List[int]]:
    infos = build_stroke_info(strokes)
    n = len(infos)
    if n == 0:
        return []

    xmins = [i.bbox[0] for i in infos]
    ymins = [i.bbox[1] for i in infos]
    xmaxs = [i.bbox[2] for i in infos]
    ymaxs = [i.bbox[3] for i in infos]
    W = max(xmaxs) - min(xmins) + 1e-6
    H = max(ymaxs) - min(ymins) + 1e-6
    xs = [i.center[0] for i in infos]
    ys = [i.center[1] for i in infos]

    if structure == "single":
        return [list(range(n))]

    if structure == "enclosure":
        areas = [max(1e-6, (i.bbox[2] - i.bbox[0]) * (i.bbox[3] - i.bbox[1])) for i in infos]
        outer_idx = int(np.argmax(areas))
        outer = [outer_idx]
        inner = [i for i in range(n) if i != outer_idx]

        closing = []
        not_closing = []
        for i in outer:
            st = infos[i]
            dx = st.points[-1][0] - st.points[0][0]
            dy = st.points[-1][1] - st.points[0][1]
            ymean = (st.bbox[1] + st.bbox[3]) * 0.5
            if abs(dx) > abs(dy) * 1.4 and ymean > min(ymins) + 0.62 * H:
                closing.append(i)
            else:
                not_closing.append(i)
        return [not_closing, inner, closing] if closing else [outer, inner]

    if structure in ("left-right", "left-mid-right"):
        gx = gap_partition(xs, W)
        groups: Dict[int, List[int]] = {}
        for i, g in enumerate(gx):
            groups.setdefault(g, []).append(i)
        return [groups[g] for g in sorted(groups.keys())]

    if structure in ("top-bottom", "top-mid-bottom"):
        gy = gap_partition(ys, H)
        groups: Dict[int, List[int]] = {}
        for i, g in enumerate(gy):
            groups.setdefault(g, []).append(i)
        return [groups[g] for g in sorted(groups.keys())]

    return [list(range(n))]


def orient_stroke(points: List[Point], stype: str) -> List[Point]:
    if len(points) < 2:
        return points
    p0, p1 = points[0], points[-1]

    if stype == "heng":
        return points if p0[0] <= p1[0] else list(reversed(points))
    if stype == "shu":
        return points if p0[1] <= p1[1] else list(reversed(points))
    if stype == "pie":
        score0 = p0[1] - 0.25 * p0[0]
        score1 = p1[1] - 0.25 * p1[0]
        return points if score0 <= score1 else list(reversed(points))
    if stype == "na":
        score0 = p0[1] + 0.25 * p0[0]
        score1 = p1[1] + 0.25 * p1[0]
        return points if score0 <= score1 else list(reversed(points))

    s0 = p0[1] * 1.25 + p0[0]
    s1 = p1[1] * 1.25 + p1[0]
    return points if s0 <= s1 else list(reversed(points))


def order_strokes_within_component(strokes: List[List[Point]], idxs: List[int], image_shape: Tuple[int, int]) -> List[int]:
    infos = build_stroke_info(strokes)
    h, w = image_shape

    type_priority = {
        "heng": 0,
        "shu": 1,
        "pie": 2,
        "na": 3,
        "dot": 4,
        "other": 5,
    }

    scored = []
    for i in idxs:
        inf = infos[i]
        x0, y0, _, _ = inf.bbox
        band_y = int(y0 / max(6.0, h * 0.08))
        band_x = int(x0 / max(6.0, w * 0.08))
        key = (band_y, band_x, type_priority.get(inf.stroke_type, 5), y0, x0)
        scored.append((key, i))

    scored.sort(key=lambda t: t[0])
    return [i for _, i in scored]


def plan_stroke_order(strokes: List[List[Point]], image_shape: Tuple[int, int]) -> List[List[Point]]:
    if not strokes:
        return []
    structure = classify_structure(strokes)
    comp_groups = order_components(strokes, structure)

    infos = build_stroke_info(strokes)
    oriented = [orient_stroke(info.points, info.stroke_type) for info in infos]

    ordered_idxs: List[int] = []
    for g in comp_groups:
        if not g:
            continue
        ordered_idxs.extend(order_strokes_within_component(oriented, g, image_shape))

    return [oriented[i] for i in ordered_idxs]


def order_strokes_by_rules(strokes: List[List[Point]], image_shape: Tuple[int, int]) -> List[List[Point]]:
    return plan_stroke_order(strokes, image_shape)


def strokes_to_gcode(
    strokes: List[List[Point]],
    image_shape: Tuple[int, int],
    output_gcode_path: Path,
    scale: float,
    z_up: float,
    z_down: float,
    feed_rate: float,
    rapid_rate: float,
) -> None:
    h, _ = image_shape

    def map_xy(p: Point) -> Tuple[float, float]:
        return p[0] * scale, (h - 1 - p[1]) * scale

    lines = [
        "; Generated by robust centerline stroke extractor",
        "G21 ; mm",
        "G90 ; absolute",
        f"G0 Z{z_up:.3f}",
        f"G0 F{rapid_rate:.1f}",
        f"G1 F{feed_rate:.1f}",
    ]

    for s in strokes:
        if len(s) < 2:
            continue
        x0, y0 = map_xy(s[0])
        lines.append(f"G0 X{x0:.3f} Y{y0:.3f}")
        lines.append(f"G1 Z{z_down:.3f}")
        for p in s[1:]:
            x, y = map_xy(p)
            lines.append(f"G1 X{x:.3f} Y{y:.3f}")
        lines.append(f"G0 Z{z_up:.3f}")

    lines.append("M2")
    output_gcode_path.parent.mkdir(parents=True, exist_ok=True)
    output_gcode_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def to_u8(bin01: np.ndarray) -> np.ndarray:
    return (bin01.astype(np.uint8) * 255)


def draw_segment_graph(segments: List[Segment], seg_graph: nx.MultiGraph, shape: Tuple[int, int]) -> np.ndarray:
    canvas = np.zeros((*shape, 3), dtype=np.uint8)
    for seg in segments:
        pts = np.array(seg.points, dtype=np.int32)
        if len(pts) >= 2:
            cv2.polylines(canvas, [pts.reshape(-1, 1, 2)], False, (120, 180, 255), 1, cv2.LINE_AA)

    for n in seg_graph.nodes:
        d = seg_graph.degree(n)
        color = (0, 255, 255) if d >= 3 else (100, 100, 100)
        r = 2 if d >= 3 else 1
        cv2.circle(canvas, n, r, color, -1, cv2.LINE_AA)
    return canvas


def draw_detected_junctions(seg_graph: nx.MultiGraph, shape: Tuple[int, int]) -> np.ndarray:
    canvas = np.zeros((*shape, 3), dtype=np.uint8)
    for n in seg_graph.nodes:
        d = seg_graph.degree(n)
        if d >= 3:
            cv2.circle(canvas, n, 3, (0, 255, 255), -1, cv2.LINE_AA)
            cv2.putText(canvas, str(d), (n[0] + 3, n[1] - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1, cv2.LINE_AA)
    return canvas


def draw_junction_pairing_result(
    segments: List[Segment],
    seg_graph: nx.MultiGraph,
    pairing_debug: List[Dict[str, object]],
    shape: Tuple[int, int],
) -> np.ndarray:
    canvas = np.zeros((*shape, 3), dtype=np.uint8)
    by_id = {s.seg_id: s for s in segments}

    for seg in segments:
        pts = np.array(seg.points, dtype=np.int32)
        if len(pts) >= 2:
            cv2.polylines(canvas, [pts.reshape(-1, 1, 2)], False, (70, 70, 70), 1, cv2.LINE_AA)

    rng = np.random.default_rng(7)
    for item in pairing_debug:
        node = item["node"]
        sid_a = int(item["sid_a"])
        sid_b = int(item["sid_b"])
        score = float(item.get("score", 0.0))
        turn = float(item.get("turn_local", 180.0))
        col = tuple(int(v) for v in rng.integers(80, 255, size=3))
        for sid in (sid_a, sid_b):
            pts = np.array(by_id[sid].points, dtype=np.int32)
            cv2.polylines(canvas, [pts.reshape(-1, 1, 2)], False, col, 2, cv2.LINE_AA)
        cv2.circle(canvas, node, 3, (255, 255, 255), -1, cv2.LINE_AA)
        cv2.putText(
            canvas,
            f"a{turn:.0f}/s{score:.2f}",
            (node[0] + 3, node[1] - 3),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.3,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

    for n in seg_graph.nodes:
        if seg_graph.degree(n) >= 3:
            cv2.circle(canvas, n, 2, (0, 255, 255), -1, cv2.LINE_AA)
    return canvas


def draw_strokes(strokes: List[List[Point]], shape: Tuple[int, int], with_index: bool) -> np.ndarray:
    canvas = np.zeros((*shape, 3), dtype=np.uint8)
    n = max(1, len(strokes))
    for i, s in enumerate(strokes):
        c = cv2.applyColorMap(np.array([[int(255 * i / n)]], dtype=np.uint8), cv2.COLORMAP_TURBO)[0, 0]
        color = (int(c[0]), int(c[1]), int(c[2]))
        pts = np.array(s, dtype=np.int32)
        if len(pts) >= 2:
            cv2.polylines(canvas, [pts.reshape(-1, 1, 2)], False, color, 1, cv2.LINE_AA)
        if len(pts) > 0:
            sp = tuple(pts[0])
            cv2.circle(canvas, sp, 2, (255, 255, 255), -1, cv2.LINE_AA)
            if with_index:
                cv2.putText(canvas, str(i + 1), (sp[0] + 2, sp[1] - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1, cv2.LINE_AA)
    return canvas


def process_one_image(image_path: Path, output_dir: Path, params: Params, debug_dir: Path) -> None:
    base = image_path.stem

    bin01 = preprocess_image(image_path, params.min_component_area)
    width_map = build_width_map(bin01)
    center_init = extract_centerline(bin01)
    center_pruned = prune_spurs(center_init, params.min_branch_length)

    segments, seg_graph = simplify_graph(center_pruned, params.simplify_epsilon)
    merged, pairing_debug, junction_nodes = merge_segments_into_strokes(
        segments,
        seg_graph,
        width_map,
        params.merge_angle_threshold,
        min_pair_length=max(3.0, params.min_branch_length * 0.8),
    )
    merged = [s for s in merged if polyline_length(s) >= max(3.0, params.min_branch_length * 0.6)]

    smoothed = smooth_strokes(merged, params.smooth_factor, params.simplify_epsilon, junction_nodes=junction_nodes)
    smoothed = [s for s in smoothed if polyline_length(s) >= max(5.0, params.min_branch_length * 1.0)]

    ordered = order_strokes_by_rules(smoothed, bin01.shape)

    strokes_to_gcode(
        ordered,
        bin01.shape,
        output_dir / f"{base}.gcode",
        scale=params.scale,
        z_up=params.z_up,
        z_down=params.z_down,
        feed_rate=params.feed_rate,
        rapid_rate=params.rapid_rate,
    )

    # Requested debug images for junction continuity reconstruction.
    dbg0 = to_u8(center_init)
    dbg1 = draw_segment_graph(segments, seg_graph, bin01.shape)
    dbg2 = draw_detected_junctions(seg_graph, bin01.shape)
    dbg3 = draw_junction_pairing_result(segments, seg_graph, pairing_debug, bin01.shape)
    dbg4 = draw_strokes(smoothed, bin01.shape, with_index=False)
    dbg5 = draw_strokes(ordered, bin01.shape, with_index=True)

    imwrite_unicode(debug_dir / f"{base}_01_original_skeleton.png", dbg0)
    imwrite_unicode(debug_dir / f"{base}_02_segment_graph.png", dbg1)
    imwrite_unicode(debug_dir / f"{base}_03_detected_junctions.png", dbg2)
    imwrite_unicode(debug_dir / f"{base}_04_branch_pairing.png", dbg3)
    imwrite_unicode(debug_dir / f"{base}_05_reconstructed_strokes.png", dbg4)
    # Keep ordered debug for previous requirements (stroke order / structure rules).
    imwrite_unicode(debug_dir / f"{base}_06_ordered_strokes.png", dbg5)


def list_pngs(input_dir: Path) -> List[Path]:
    return sorted([p for p in input_dir.rglob("*.png") if p.is_file()])


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Robust centerline + junction continuation + stroke-order planning")
    p.add_argument("--input_dir", type=str, default="png_library")
    p.add_argument("--output_dir", type=str, default="gcode_library")
    p.add_argument("--debug_dir", type=str, default=None)
    p.add_argument("--samples", type=str, default="", help="comma-separated stems, e.g. 木,永,国")

    p.add_argument("--min_branch_length", type=float, default=5.0)
    p.add_argument("--min_component_area", type=int, default=20)
    p.add_argument("--merge_angle_threshold", type=float, default=38.0)
    p.add_argument("--simplify_epsilon", type=float, default=1.5)
    p.add_argument("--smooth_factor", type=float, default=2.0)

    p.add_argument("--scale", type=float, default=1.0)
    p.add_argument("--z_up", type=float, default=5.0)
    p.add_argument("--z_down", type=float, default=0.0)
    p.add_argument("--feed_rate", type=float, default=1800.0)
    p.add_argument("--rapid_rate", type=float, default=4000.0)

    p.add_argument("--limit", type=int, default=0)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    debug_dir = Path(args.debug_dir) if args.debug_dir else output_dir / "debug"

    params = Params(
        min_branch_length=args.min_branch_length,
        min_component_area=args.min_component_area,
        merge_angle_threshold=args.merge_angle_threshold,
        simplify_epsilon=args.simplify_epsilon,
        smooth_factor=args.smooth_factor,
        scale=args.scale,
        z_up=args.z_up,
        z_down=args.z_down,
        feed_rate=args.feed_rate,
        rapid_rate=args.rapid_rate,
    )

    files = list_pngs(input_dir)
    if args.samples.strip():
        wants = {s.strip() for s in args.samples.split(",") if s.strip()}
        files = [p for p in files if p.stem in wants]
    if args.limit > 0:
        files = files[: args.limit]

    output_dir.mkdir(parents=True, exist_ok=True)
    debug_dir.mkdir(parents=True, exist_ok=True)

    if not files:
        print(f"No PNG files found in: {input_dir}")
        return

    print(f"Processing {len(files)} PNG files...")
    ok = 0
    for i, f in enumerate(files, 1):
        try:
            process_one_image(f, output_dir, params, debug_dir)
            ok += 1
        except Exception as e:
            print(f"[FAIL] {f.name}: {e}")
        if i % 20 == 0 or i == len(files):
            print(f"Progress: {i}/{len(files)}")

    print(f"Done. Success: {ok}/{len(files)}")
    print(f"G-code output: {output_dir}")
    print(f"Debug images: {debug_dir}")


if __name__ == "__main__":
    main()
