"""Reusable, side-effect-free geometric distance metrics for reward computation.

This module extracts the point-cloud distance math used by the evaluation
harness (``eval_metrics/icp_original.py``) into clean, importable functions so
the live reward loop can score generated geometry against a reference shape.

All heavy imports (open3d / numpy) are performed lazily inside the functions so
that importing :class:`~core.reward_engine.RewardEngine` never forces an open3d
dependency unless distance-based scoring is actually requested.
"""

from __future__ import annotations

from typing import List, Optional, Sequence


def stl_to_point_cloud(stl_path: str, num_points: int = 1000) -> "object":
    """Poisson-disk sample a triangle mesh STL into an (N, 3) numpy array."""
    import numpy as np
    import open3d as o3d

    mesh = o3d.io.read_triangle_mesh(stl_path)
    pcd = mesh.sample_points_poisson_disk(num_points)
    return np.asarray(pcd.points)


def normalize_point_cloud(points: "object") -> "object":
    """Center a point cloud at the origin and scale it to the unit sphere.

    Mirrors ``pc_normalize`` in ``eval_metrics/icp_original.py`` so that scores
    are translation- and scale-invariant (matching the benchmark methodology).
    """
    import numpy as np

    pts = np.asarray(points, dtype=float)
    centroid = np.mean(pts, axis=0)
    pts = pts - centroid
    scale = np.max(np.sqrt(np.sum(pts ** 2, axis=1)))
    if scale == 0:
        return pts
    return pts / scale


def _directed_min_distances(src: "object", dst: "object") -> "object":
    """Minimum euclidean distance from each point in ``src`` to ``dst``."""
    import numpy as np

    src_e = np.expand_dims(src, axis=1)  # (N, 1, 3)
    dst_e = np.expand_dims(dst, axis=0)  # (1, M, 3)
    dists = np.sqrt(np.sum((src_e - dst_e) ** 2, axis=2))
    return np.min(dists, axis=1)


def chamfer_distance(pc1: "object", pc2: "object") -> float:
    """Symmetric average closest-point distance between two point clouds.

    This is the mean of both directed average minimum distances, i.e. a
    symmetric Chamfer distance. Lower is better; 0.0 means identical clouds.
    """
    import numpy as np

    d12 = _directed_min_distances(pc1, pc2)
    d21 = _directed_min_distances(pc2, pc1)
    return float((np.mean(d12) + np.mean(d21)) / 2.0)


def hausdorff_distance(pc1: "object", pc2: "object") -> float:
    """Symmetric Hausdorff distance (max of both directed worst-case distances)."""
    import numpy as np

    d12 = _directed_min_distances(pc1, pc2)
    d21 = _directed_min_distances(pc2, pc1)
    return float(max(np.max(d12), np.max(d21)))


def load_reference_points(reference: object, num_points: int = 1000) -> Optional["object"]:
    """Resolve a reference into a normalized (N, 3) point array.

    ``reference`` may be:
      * a path to an ``.stl`` file (string), or
      * an already-materialized list/array of (x, y, z) points.

    Returns ``None`` if the reference cannot be resolved.
    """
    import numpy as np

    if reference is None:
        return None
    if isinstance(reference, str):
        try:
            pts = stl_to_point_cloud(reference, num_points)
        except Exception:
            return None
    else:
        try:
            pts = np.asarray(reference, dtype=float)
        except Exception:
            return None
    if pts.ndim != 2 or pts.shape[1] != 3 or pts.shape[0] == 0:
        return None
    return normalize_point_cloud(pts)


def distance_score(chamfer: float, threshold: float) -> float:
    """Map a chamfer distance to a graded score in [0, 1].

    Returns 1.0 at distance 0 and decays linearly to 0.0 at ``2 * threshold``,
    so that a model right at the pass threshold still scores ~0.5 and gives the
    agent a smooth gradient to optimize against (instead of a binary cliff).
    """
    if threshold <= 0:
        return 1.0 if chamfer == 0 else 0.0
    score = 1.0 - (chamfer / (2.0 * threshold))
    return max(0.0, min(1.0, score))
