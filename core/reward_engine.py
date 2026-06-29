from typing import Dict, Any, Optional, Tuple, List
import math


class RewardEngine:
    """Computes a gated reward validating compilation and geometric correctness.

    The reward keeps the multiplicative gate ``R = R_exec * R_geom`` so that the
    agent loop can rely on ``reward == 1.0`` as its terminal success signal. On
    top of the binary gate it also exposes a continuous ``geom_score`` in [0, 1]
    so the agent receives partial-credit feedback (e.g. "you are at 0.82, faces
    off by one") instead of an all-or-nothing zero.

    Two geometric scoring modes are supported:

    * **Topology mode** (default): compares B-Rep metrics (volume, face / edge /
      vertex counts, center of mass) against target constraints. Integer counts
      accept an optional ``topology_tolerance`` slack so that a single extra
      fillet edge no longer zeroes the whole reward.
    * **Distance mode**: active when ``constraints`` provides a ``reference``
      (an STL path or an explicit point list) together with a ``generated_stl``
      to score. Uses symmetric Chamfer + Hausdorff distance on normalized point
      clouds, matching the evaluation harness methodology, and passes the gate
      when the Chamfer distance falls under ``distance_threshold``.
    """

    @staticmethod
    def calculate_reward(
        exec_success: bool,
        metrics: Optional[Dict[str, Any]],
        constraints: Dict[str, Any],
        generated_stl: Optional[str] = None,
    ) -> Tuple[float, Dict[str, Any]]:
        """Calculate the gated reward ``R = R_exec * R_geom``.

        Returns a tuple of ``(reward, breakdown)``. ``breakdown`` always
        contains ``R_exec``, ``R_geom`` (binary gate), ``geom_score``
        (continuous), ``failed_constraints`` and, in distance mode, ``distances``.
        """
        breakdown: Dict[str, Any] = {
            "R_exec": 0.0,
            "R_geom": 1.0,
            "geom_score": 0.0,
            "failed_constraints": [],
        }

        # 1. Execution / compile gate.
        if not exec_success or not metrics:
            return 0.0, breakdown

        breakdown["R_exec"] = 1.0

        # 2. Distance-based geometric scoring (only when a reference is given).
        reference = constraints.get("reference")
        if reference is not None:
            RewardEngine._score_distance(breakdown, constraints, generated_stl)
        else:
            RewardEngine._score_topology(breakdown, metrics, constraints)

        reward = breakdown["R_exec"] * breakdown["R_geom"]
        return reward, breakdown

    # ------------------------------------------------------------------
    # Topology (B-Rep metric) scoring
    # ------------------------------------------------------------------
    @staticmethod
    def _score_topology(
        breakdown: Dict[str, Any],
        metrics: Dict[str, Any],
        constraints: Dict[str, Any],
    ) -> None:
        tolerance = constraints.get("tolerance", 0.01)
        int_slack = int(constraints.get("topology_tolerance", 0))

        checks: List[bool] = []

        # Volume (relative tolerance).
        if "volume" in constraints:
            target = constraints["volume"]
            actual = metrics.get("volume", 0.0)
            ok = math.isclose(actual, target, rel_tol=tolerance)
            checks.append(ok)
            if not ok:
                breakdown["R_geom"] = 0.0
                breakdown["failed_constraints"].append(
                    f"volume (expected: {target}, got: {actual})"
                )

        # Integer count constraints (allow optional slack).
        for key in ("num_faces", "num_edges", "num_vertices"):
            if key in constraints:
                target = constraints[key]
                actual = metrics.get(key, 0)
                ok = abs(actual - target) <= int_slack
                checks.append(ok)
                if not ok:
                    breakdown["R_geom"] = 0.0
                    breakdown["failed_constraints"].append(
                        f"{key} (expected: {target}, got: {actual})"
                    )

        # Center of mass (absolute tolerance per axis).
        if "center_of_mass" in constraints:
            target = constraints["center_of_mass"]
            actual = metrics.get("center_of_mass", [0.0, 0.0, 0.0])
            com_ok = all(
                math.isclose(actual[i], target[i], abs_tol=tolerance)
                for i in range(3)
            )
            checks.append(com_ok)
            if not com_ok:
                breakdown["R_geom"] = 0.0
                breakdown["failed_constraints"].append(
                    f"center_of_mass (expected: {target}, got: {actual})"
                )

        # Continuous score = fraction of satisfied constraints (1.0 if none set).
        breakdown["geom_score"] = (sum(checks) / len(checks)) if checks else 1.0

    # ------------------------------------------------------------------
    # Distance-based scoring against a reference shape
    # ------------------------------------------------------------------
    @staticmethod
    def _score_distance(
        breakdown: Dict[str, Any],
        constraints: Dict[str, Any],
        generated_stl: Optional[str],
    ) -> None:
        threshold = float(constraints.get("distance_threshold", 0.1))

        if not generated_stl:
            breakdown["R_geom"] = 0.0
            breakdown["failed_constraints"].append(
                "distance scoring requested but no generated STL was provided"
            )
            return

        try:
            from core.geometry_metrics import (
                stl_to_point_cloud,
                normalize_point_cloud,
                load_reference_points,
                chamfer_distance,
                hausdorff_distance,
                distance_score,
            )

            ref_pts = load_reference_points(constraints["reference"])
            if ref_pts is None:
                raise ValueError("could not resolve reference geometry")

            gen_pts = normalize_point_cloud(stl_to_point_cloud(generated_stl))

            chamfer = chamfer_distance(gen_pts, ref_pts)
            hausdorff = hausdorff_distance(gen_pts, ref_pts)
        except Exception as e:  # pragma: no cover - depends on open3d/runtime files
            breakdown["R_geom"] = 0.0
            breakdown["failed_constraints"].append(
                f"distance scoring failed: {e}"
            )
            return

        breakdown["distances"] = {
            "chamfer": chamfer,
            "hausdorff": hausdorff,
            "threshold": threshold,
        }
        breakdown["geom_score"] = distance_score(chamfer, threshold)

        if chamfer > threshold:
            breakdown["R_geom"] = 0.0
            breakdown["failed_constraints"].append(
                f"chamfer distance {chamfer:.4f} exceeds threshold {threshold:.4f} "
                f"(hausdorff {hausdorff:.4f})"
            )
