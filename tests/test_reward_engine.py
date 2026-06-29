"""Unit tests for RewardEngine graded topology scoring.

These tests exercise the topology path only, so they do not require open3d /
the distance-based scoring runtime. Run with: ``python -m pytest tests/test_reward_engine.py``
or directly: ``python tests/test_reward_engine.py``.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.reward_engine import RewardEngine


def test_compile_failure_zeroes_reward():
    reward, breakdown = RewardEngine.calculate_reward(False, None, {})
    assert reward == 0.0
    assert breakdown["R_exec"] == 0.0


def test_no_constraints_passes():
    metrics = {"volume": 100.0, "num_faces": 6}
    reward, breakdown = RewardEngine.calculate_reward(True, metrics, {})
    assert reward == 1.0
    assert breakdown["geom_score"] == 1.0


def test_volume_within_tolerance_passes():
    metrics = {"volume": 100.4}
    reward, breakdown = RewardEngine.calculate_reward(
        True, metrics, {"volume": 100.0, "tolerance": 0.01}
    )
    assert reward == 1.0


def test_single_failed_constraint_gives_partial_score():
    # Volume ok, face count off -> gate fails but geom_score reflects 1/2 satisfied.
    metrics = {"volume": 100.0, "num_faces": 7}
    reward, breakdown = RewardEngine.calculate_reward(
        True, metrics, {"volume": 100.0, "num_faces": 6}
    )
    assert reward == 0.0
    assert breakdown["geom_score"] == 0.5
    assert any("num_faces" in f for f in breakdown["failed_constraints"])


def test_topology_tolerance_allows_slack():
    # One extra edge should pass when topology_tolerance allows +/-1.
    metrics = {"num_edges": 13}
    reward, _ = RewardEngine.calculate_reward(
        True, metrics, {"num_edges": 12, "topology_tolerance": 1}
    )
    assert reward == 1.0


def _run():
    fns = [v for k, v in globals().items() if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"PASS {fn.__name__}")
    print(f"\n{len(fns)} tests passed.")


if __name__ == "__main__":
    _run()
