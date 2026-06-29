# MEDA Agent Improvements (June 2026)

*Last updated: 2026-06-29*

This change set implements the highest-leverage items from the prioritized
roadmap in [`text_to_cad_landscape_2026.md`](./text_to_cad_landscape_2026.md) §8,
**borrowing concrete techniques from the strongest open-source and commercial
peers** rather than chasing the (training-heavy) RL frontier. It also repairs
three live regressions that the last round of merges left in the multi-agent
loop.

## 0. Regression repairs (the loop was broken before this)

Auditing the last few PRs (#1–#5) against the actual code surfaced three breaks
introduced by the merges, all on the **primary multi-agent path**:

| ID | Symptom | Root cause | Fix |
|----|---------|------------|-----|
| **R1** | Every `reward==1.0` threw, was caught, and forced `reward=0.0` → **success unreachable** | `core/reasoning_core` imported `capture_orthographic_collage`, but that function was **never actually added** to `utils/capture_screenshot.py` (PR #3 added only the import + call site) → `ImportError` on every render | Added a loop-safe `capture_orthographic_collage()` that writes the collage and returns PNG bytes, never `sys.exit`s |
| **R2** | Non-fast loop **crashed immediately** with `NameError: memory_guidance` | PR #5's merge referenced an undefined `memory_guidance` in the opening user message | Removed the dangling reference; retrieved memory is already injected into the agent instructions |
| **R3** | Trajectory memory silently never recorded | `self.memory_store.record_run(...)` referenced an attribute that was never created (`CADMemoryStore` was imported-by-design in PR #5 but never instantiated) | Instantiate `self.trajectory_memory = CADMemoryStore()` and record/retrieve through it |

## 1. CadQuery API RAG grounding — *from CADSmith* (roadmap §8.1)

**`core/cad_knowledge.py`** ships a curated, kernel-verified CadQuery API
reference (signatures + the *gotchas* that actually cause failures + worked
examples). Borrowed from the open-source peer **CADSmith** (arXiv:2603.26512),
whose reliability comes largely from a Coder agent that retrieves over CadQuery
API docs before writing code.

- Retrieval is keyword-based by default (offline, zero-dependency) and upgrades
  to embedding cosine similarity when the `LearningStore` embedder is available.
- Always-on injection into the modeler preamble (`build_memory_preamble`) — the
  cheapest defense against hallucinated methods like `.roundEdges()`.
- Exposed as an on-demand **`lookup_cadquery_api`** tool the modeler can call
  before writing an unfamiliar operation.

## 2. B-Rep selector grounding — *from BRepGround / CADSmith* (roadmap §8.3)

**`core/geometry_introspection.py`** lets the modeler **see real faces and
edges** instead of guessing selectors. This is the field's explicit frontier
(Text2CAD-Bench, the BRepGround paper arXiv:2603.11831): resolving a
natural-language geometric query to specific B-Rep entities before applying
fillet/chamfer/shell.

Rather than a trained BERT+UV-Net resolver, MEDA exposes the **actual
OpenCASCADE measurements** (à la CADSmith's exact-kernel validation):

- An introspection snippet runs the current timeline in the sandbox and reports
  each face's normal/center/area and each edge's length/orientation.
- The host formats this into a selector guide ("4x vertical (|Z)", "top (>Z)")
  **plus a safe fillet/chamfer ceiling** (45% of the shortest edge) so radii
  stop overflowing the kernel.
- Exposed as the **`inspect_current_model`** tool, which the modeler is
  instructed to call *before* any fillet/chamfer/shell/face-hole.

## 3. Parallel candidates + geometry-aware judge — *from EvoCAD* (roadmap §8.6)

**`core/candidate_search.py`** generates several candidates with **different
decomposition strategies** (primitive-first, sketch-extrude, boolean
composition, revolve/loft, minimal) and **keeps the best**, the technique
EvoCAD (arXiv:2510.11631) uses to beat single-attempt loops on hard prompts.

- Pure, dependency-free strategy/scoring logic (unit-tested without cadquery or
  an LLM); `reasoning_core` supplies the generate/execute callbacks.
- Score = graded geometric correctness + validity (positive-volume solid) +
  a small simplicity bonus + an optional visual-match nudge.
- Surfaced as a **"Parallel candidates" slider (1–5)** in the Streamlit UI and a
  `num_candidates` argument on `run_design_loop` (1 = off).

## 4. Sandbox hardening + compiler-as-verifier — *roadmap §8.5 & §8.8*

**`core/sandbox.py`** closes the long-tracked **M5** RCE surface with portable
hardening (full containerization remains the deployment-time goal):

- **AST allow-list** (default-deny): generated code may only import a curated
  set of modeling modules and may not call `eval`/`exec`/`open`/`__import__`…
  or touch escape dunders (`__subclasses__`/`__globals__`…). Rejected code never
  runs; the reason is fed back to the agent.
- **POSIX resource limits** (CPU-time + output-file-size; opt-in memory cap via
  `MEDA_SANDBOX_MEMLIMIT_MB`) as a runaway backstop.
- **Secret scrubbing**: API keys are stripped from the child environment so a
  prompt-injected script cannot exfiltrate them.
- **CAD-Judge-style validity signals** (`is_valid`, `num_solids`) added to the
  topology metrics — a cheap compiler-as-verifier check (CAD-Judge
  arXiv:2508.04002) that needs no VLM call.

## Tests

All new logic is covered by offline, dependency-free unit tests:

- `tests/test_cad_knowledge.py` — API retrieval relevance + formatting (7)
- `tests/test_candidate_search.py` — plan diversity, scoring, ranking (8)
- `tests/test_sandbox_validation.py` — allow/deny matrix incl. escape attempts (10)
- `tests/test_geometry_introspection.py` — report parsing/classification (9)

plus the existing `test_reward_engine.py` (5) and `test_learning_store.py` (5).
Full-loop validation still requires the synced `uv` env (cadquery / open3d /
google-adk) and a Gemini key; the kernel-dependent paths are exercised
structurally here and import cleanly.

## What we deliberately did *not* do

- **RL / rejection-sampling fine-tune** (roadmap §8.2) — the highest ceiling but
  requires a training pipeline + GPU; out of scope for an in-repo code change.
  The graded reward MEDA already computes is the data source when that lands.
- **Full container isolation** — needs a runtime (Docker/nsjail) not guaranteed
  in every deployment; the AST allow-list + rlimits are the portable subset.
