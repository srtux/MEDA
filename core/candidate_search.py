"""Parallel candidate generation + geometry-aware judging.

The roadmap (``docs/text_to_cad_landscape_2026.md`` §8.6) calls for **parallel
candidates + a judge panel / evolutionary search**, the technique EvoCAD
(arXiv:2510.11631) uses to beat single-attempt agentic loops on hard prompts:
generate several designs with *different decomposition strategies*, score them
with a geometry-aware ranker, and keep the best (optionally refining it).

A single-shot LLM commits to one modeling approach and lives or dies by it.
Sampling diverse strategies (primitive-first vs. sketch-extrude vs. boolean
composition vs. revolve/loft) and ranking by compile success + geometric score +
simplicity recovers many prompts the single attempt would fail.

This module holds the **pure, dependency-free** strategy/scoring logic so it is
unit-testable without cadquery or an LLM. ``core.reasoning_core`` supplies the
``generate_fn``/``execute_fn`` callbacks that actually call the model and the
sandbox, and drives :func:`run_candidate_search`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

# Diverse decomposition strategies. Each becomes an extra system hint so the N
# candidates explore genuinely different modeling routes instead of resampling
# the same one. Ordered so that with N=1 you get the most general approach.
STRATEGY_HINTS: List[Tuple[str, str]] = [
    (
        "primitive_first",
        "Strategy: build from primitive solids (box, cylinder, sphere) and combine "
        "them with boolean union/cut. Prefer the simplest primitive that fits each part.",
    ),
    (
        "sketch_extrude",
        "Strategy: sketch 2D profiles on workplanes (rect, circle, polygon, lineTo/close) "
        "and extrude/cut them. Anchor secondary features to faces with selectors.",
    ),
    (
        "boolean_composition",
        "Strategy: model each distinct feature as its own positioned solid (using "
        "translate/rotate) and assemble them with union/cut. Ensure solids overlap before union.",
    ),
    (
        "revolve_loft",
        "Strategy: for round or tapering parts, use revolve on a closed profile or loft "
        "between profiles on offset workplanes rather than stacking primitives.",
    ),
    (
        "minimal",
        "Strategy: use the fewest possible features and parameters; favor one base solid "
        "plus at most a couple of surgical operations.",
    ),
]

# Temperatures paired to candidates: first deterministic, rest progressively
# more exploratory so diversity rises without going fully random.
_TEMPS = [0.0, 0.2, 0.4, 0.3, 0.5]


@dataclass
class Candidate:
    """One generated-and-executed design attempt."""

    strategy: str
    code: str
    success: bool = False
    metrics: Optional[Dict[str, Any]] = None
    geom_score: float = 0.0
    reward: float = 0.0
    error: str = ""
    score: float = 0.0
    extra: Dict[str, Any] = field(default_factory=dict)


def build_candidate_plan(n: int) -> List[Tuple[str, str, float]]:
    """Return ``n`` (strategy_name, strategy_hint, temperature) tuples."""
    n = max(1, n)
    plan = []
    for i in range(n):
        name, hint = STRATEGY_HINTS[i % len(STRATEGY_HINTS)]
        temp = _TEMPS[i % len(_TEMPS)]
        # Disambiguate repeated strategies on a second cycle by nudging temp.
        if i >= len(STRATEGY_HINTS):
            temp = min(0.8, temp + 0.2)
            name = f"{name}_v{i // len(STRATEGY_HINTS) + 1}"
        plan.append((name, hint, temp))
    return plan


def _num_feature_lines(code: str) -> int:
    """Count substantive (non-comment, non-blank) lines as a simplicity proxy."""
    count = 0
    for line in (code or "").splitlines():
        s = line.strip()
        if not s or s.startswith("#") or s.startswith("import "):
            continue
        count += 1
    return count


def score_candidate(cand: Candidate) -> float:
    """Geometry-aware score in roughly [0, 1.2]; higher is better.

    Compile failures score 0. Among compiling candidates the geometric score
    dominates, with a small simplicity bonus to break ties toward cleaner models
    and a small validity bonus for a positive-volume solid. A visual-match
    verdict (``extra['visual_match']``), when present, nudges the score.
    """
    if not cand.success:
        return 0.0

    # Base: graded geometric correctness (1.0 when there are no constraints).
    base = cand.geom_score if cand.geom_score else (1.0 if cand.reward == 1.0 else 0.5)
    score = base

    # Validity: a real positive-volume solid.
    vol = (cand.metrics or {}).get("volume")
    if vol is not None and vol > 0:
        score += 0.05

    # Simplicity: reward fewer feature lines (diminishing, capped small).
    nlines = _num_feature_lines(cand.code)
    if nlines > 0:
        score += min(0.1, 2.0 / (nlines + 5))

    # Optional visual judge verdict.
    vm = cand.extra.get("visual_match")
    if vm is True:
        score += 0.1
    elif vm is False:
        score -= 0.15

    return round(score, 5)


def rank_candidates(cands: List[Candidate]) -> List[Candidate]:
    """Score and return candidates sorted best-first (stable on ties)."""
    for c in cands:
        c.score = score_candidate(c)
    # Prefer higher score, then a built solid, then fewer lines.
    return sorted(
        cands,
        key=lambda c: (-c.score, not c.success, _num_feature_lines(c.code)),
    )


def select_best(cands: List[Candidate]) -> Optional[Candidate]:
    ranked = rank_candidates(cands)
    return ranked[0] if ranked else None


def run_candidate_search(
    n: int,
    generate_fn: Callable[[str, str, float], str],
    execute_fn: Callable[[str], Candidate],
    log_fn: Optional[Callable[[str], None]] = None,
) -> Tuple[Optional[Candidate], List[Candidate]]:
    """Generate, execute and rank ``n`` candidates.

    ``generate_fn(strategy_name, strategy_hint, temperature) -> code`` produces a
    candidate script; ``execute_fn(code) -> Candidate`` compiles/scores the
    script. The returned ``Candidate.strategy`` is ignored and overwritten here
    with the plan's strategy name, so ``execute_fn`` need not set it.
    Returns ``(best, all_candidates)``.
    """
    plan = build_candidate_plan(n)
    cands: List[Candidate] = []
    for name, hint, temp in plan:
        if log_fn:
            log_fn(f"[LOG] Generating candidate '{name}' (temp={temp})...")
        try:
            code = generate_fn(name, hint, temp)
            cand = execute_fn(code)
            cand.strategy = name
        except Exception as e:  # pragma: no cover - depends on LLM/sandbox runtime
            cand = Candidate(strategy=name, code="", success=False, error=str(e))
        cands.append(cand)
        if log_fn:
            status = "ok" if cand.success else f"fail ({cand.error[:60]})"
            log_fn(f"[LOG]   candidate '{name}': {status}, geom_score={cand.geom_score:.3f}")

    best = select_best(cands)
    if best and log_fn:
        log_fn(
            f"[LOG] Best candidate: '{best.strategy}' "
            f"(score={best.score:.3f}, success={best.success})"
        )
    return best, cands
