from typing import Dict, Any, Optional, Tuple
import math

class RewardEngine:
    """Computes the multiplicative gating reward to strictly validate compilation and topological geometry correctness."""
    @staticmethod
    def calculate_reward(
        exec_success: bool,
        metrics: Optional[Dict[str, Any]],
        constraints: Dict[str, Any]
    ) -> Tuple[float, Dict[str, Any]]:
        """
        Calculates the gated reward R = R_exec * R_geom.
        Returns a tuple of (reward, breakdown_info).
        """
        breakdown = {
            "R_exec": 0.0,
            "R_geom": 1.0,
            "failed_constraints": []
        }

        # 1. Check Execution compile reward
        if not exec_success or not metrics:
            return 0.0, breakdown

        breakdown["R_exec"] = 1.0

        # 2. Check Geometric constraints
        tolerance = constraints.get("tolerance", 0.01)

        # Volume constraint
        if "volume" in constraints:
            target = constraints["volume"]
            actual = metrics.get("volume", 0.0)
            if not math.isclose(actual, target, rel_tol=tolerance):
                breakdown["R_geom"] = 0.0
                breakdown["failed_constraints"].append(f"volume (expected: {target}, got: {actual})")

        # Number of faces constraint
        if "num_faces" in constraints:
            target = constraints["num_faces"]
            actual = metrics.get("num_faces", 0)
            if actual != target:
                breakdown["R_geom"] = 0.0
                breakdown["failed_constraints"].append(f"num_faces (expected: {target}, got: {actual})")

        # Number of edges constraint
        if "num_edges" in constraints:
            target = constraints["num_edges"]
            actual = metrics.get("num_edges", 0)
            if actual != target:
                breakdown["R_geom"] = 0.0
                breakdown["failed_constraints"].append(f"num_edges (expected: {target}, got: {actual})")

        # Number of vertices constraint
        if "num_vertices" in constraints:
            target = constraints["num_vertices"]
            actual = metrics.get("num_vertices", 0)
            if actual != target:
                breakdown["R_geom"] = 0.0
                breakdown["failed_constraints"].append(f"num_vertices (expected: {target}, got: {actual})")

        # Center of Mass constraint
        if "center_of_mass" in constraints:
            target = constraints["center_of_mass"]
            actual = metrics.get("center_of_mass", [0.0, 0.0, 0.0])
            for i in range(3):
                if not math.isclose(actual[i], target[i], abs_tol=tolerance):
                    breakdown["R_geom"] = 0.0
                    breakdown["failed_constraints"].append(
                        f"center_of_mass[{i}] (expected: {target[i]}, got: {actual[i]})"
                    )
                    break

        reward = breakdown["R_exec"] * breakdown["R_geom"]
        return reward, breakdown
