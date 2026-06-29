"""Tests for the parallel candidate search scoring/ranking logic."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core import candidate_search
from core.candidate_search import Candidate, build_candidate_plan, score_candidate, select_best


def test_plan_size_and_diversity():
    plan = build_candidate_plan(4)
    assert len(plan) == 4
    names = [p[0] for p in plan]
    assert len(set(names)) == 4  # four distinct strategies


def test_plan_wraps_with_version_suffix():
    plan = build_candidate_plan(7)
    assert len(plan) == 7
    # The 6th entry reuses the first strategy with a version suffix.
    assert plan[5][0].endswith("_v2")


def test_failed_candidate_scores_zero():
    c = Candidate(strategy="x", code="model = 1", success=False, geom_score=0.0)
    assert score_candidate(c) == 0.0


def test_higher_geom_score_wins():
    good = Candidate(strategy="a", code="model = cq.Workplane('XY').box(1,1,1)",
                     success=True, geom_score=1.0, reward=1.0, metrics={"volume": 1.0})
    weak = Candidate(strategy="b", code="model = cq.Workplane('XY').box(1,1,1)",
                     success=True, geom_score=0.4, reward=0.0, metrics={"volume": 1.0})
    assert score_candidate(good) > score_candidate(weak)


def test_simplicity_breaks_ties():
    simple = Candidate(strategy="s", code="model = cq.Workplane('XY').box(1,1,1)",
                       success=True, geom_score=1.0, metrics={"volume": 1.0})
    complex_ = Candidate(
        strategy="c",
        code="\n".join([f"model = model.faces('>Z').workplane().hole({i})" for i in range(20)]),
        success=True, geom_score=1.0, metrics={"volume": 1.0},
    )
    assert score_candidate(simple) > score_candidate(complex_)


def test_select_best_prefers_compiling():
    cands = [
        Candidate(strategy="fail", code="boom", success=False),
        Candidate(strategy="ok", code="model = cq.Workplane('XY').box(1,1,1)",
                  success=True, geom_score=0.8, metrics={"volume": 1.0}),
    ]
    best = select_best(cands)
    assert best.strategy == "ok"


def test_visual_match_nudges_score():
    base = Candidate(strategy="v", code="model = cq.Workplane('XY').box(1,1,1)",
                     success=True, geom_score=0.7, metrics={"volume": 1.0})
    matched = Candidate(strategy="v", code=base.code, success=True, geom_score=0.7,
                        metrics={"volume": 1.0}, extra={"visual_match": True})
    assert score_candidate(matched) > score_candidate(base)


def test_run_candidate_search_orchestration():
    # Fake generate/execute callbacks: strategy "minimal" yields the best score.
    def generate_fn(name, hint, temp):
        return f"# {name}\nmodel = cq.Workplane('XY').box(1,1,1)"

    def execute_fn(code):
        gs = 1.0 if "minimal" in code else 0.5
        return Candidate(strategy="", code=code, success=True, geom_score=gs,
                         metrics={"volume": 1.0})

    best, cands = candidate_search.run_candidate_search(5, generate_fn, execute_fn)
    assert len(cands) == 5
    assert best is not None and best.success
    assert "minimal" in best.code


def _run():
    fns = [v for k, v in globals().items() if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"PASS {fn.__name__}")
    print(f"\n{len(fns)} tests passed.")


if __name__ == "__main__":
    _run()
