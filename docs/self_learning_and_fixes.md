# MEDA: Self-Learning Loop, Per-Iteration Rendering & Correctness Fixes

*Last updated: 2026-06-29*

This change set does three things, informed by a parallel multi-agent audit of
the codebase:

1. **Fixes serious correctness bugs** (two critical, four high, several medium).
2. **Adds a durable, cross-run self-learning loop** (lessons + skills library).
3. **Renders the model on every iteration** and feeds it back to the agent.

---

## 1. Correctness fixes

| ID | Severity | File | Problem | Fix |
|----|----------|------|---------|-----|
| **C1** | Critical | `core/reasoning_core.py`, `utils/capture_screenshot.py` | The visual-critique step imported `capture_orthographic_collage`, which **did not exist** (the function was `capture_stl_screenshot`). Every time the B-Rep reward hit 1.0 the import threw, was caught, and **forced `reward = 0.0`** — so a *correct* model was always downgraded to failure and the loop could never confirm success. | Added a loop-safe `capture_orthographic_collage` that returns PNG bytes and never calls `sys.exit`; the critique now reuses the per-iteration render. |
| **C2** | Critical | `core/reasoning_core.py` | `max_iterations` was accepted but **never enforced** — termination relied entirely on the orchestrator LLM. `iteration` counted raw ADK events, not design turns. A stuck run could spin with unbounded API spend. | Added a real `exec_turns` counter (compile/verify cycles), an explicit `break` on success / `max_iterations` / an absolute event cap, and report both `design_iterations` and `events`. |
| **H1** | High | `core/sandbox.py`, `core/reasoning_core.py` | Stale `001.stl` / `001.png` from a previous iteration could be read when the current compile failed — corrupting distance-reward scoring and the screenshot. | The sandbox deletes `001.stl/step/png` before every run; reward/render only use the STL when the current run actually produced it; renders use per-iteration `iter_{n}.png` filenames. |
| **H2** | High | `core/sandbox.py` | `success` required `[COMPILE_SUCCESS]`, which only printed on *export* success, so a valid solid that failed only at STL/STEP export was discarded along with its metrics. | Emit `[MODEL_BUILT]` right after the solid is constructed; success is based on the solid being built + metrics extracted, independent of export. Export failure is surfaced as a soft warning. |
| **H3** | High | `core/reasoning_core.py`, `core/reward_engine.py` | An LLM was asked to predict *exact* `num_faces`/`num_edges` from text, merged as zero-slack hard constraints — making most designs (fillets, holes, patterns) permanently unsatisfiable. | Constraint inference no longer guesses integer counts; it infers only tolerance-checked metrics (volume, center of mass). Shape correctness is enforced by the (now-working) visual critic. (Graded scoring + `topology_tolerance` already landed in PR #2.) |
| **H4** | High | `core/reasoning_core.py` | All tools mutate a single module global `_active_core`; two concurrent Streamlit sessions would clobber each other. | `run_design_loop` now serializes on a module lock, and per-run state is reset at the start of each run. |
| **M1** | Medium | `core/canvas.py` | Every generated script did `import ocp_vscode` (never used); if that optional package was absent the **entire pipeline hard-failed** with an unrelated `ModuleNotFoundError`. | Dropped the unused import. |
| **M3** | Medium | `core/sandbox.py` | Topology counts came from the `Workplane` (selection-dependent) while volume came from the solid — counts could describe a different thing than the measured volume. | All metrics now derive from the resolved solid `Shape` (`val.Faces()/Edges()/Vertices()`). |

Remaining known item (partially mitigated, tracked for follow-up): **M5** —
generated CAD still runs in a `subprocess`, so hosted/multi-user deployments
should add OS-level isolation such as a locked-down container, nsjail, or
firejail. The current local sandbox is no longer a bare subprocess: it deletes
stale artifacts, applies AST allow-list validation, blocks common file/process
I/O escape helpers, scrubs secret-bearing environment variables, and applies
POSIX resource limits where available.

---

## 2. Self-learning loop (lessons + skills)

**Why.** ADK's memory/session services were wired into the `Runner` but
**never written to or read from** — nothing persisted across runs, so the agent
repeated the same mistakes every time.

**What.** A local, cloud-free SQLite store (`memory/meda_memory.db`) with two
tables, exposed via `core/learning_store.py`:

- **`lessons`** — `error signature → root cause → corrective fix`, born when a
  failure that was present on one turn disappears on a later turn with improved
  reward (Reflexion / ExpeL-style verbal self-reflection).
- **`skills`** — parameterized CadQuery snippets distilled from successful runs,
  retrievable by sub-goal similarity (Voyager / Agent-Workflow-Memory-style
  reusable skill library).

**How it plugs in** (`core/reasoning_core.py`, all additive — the reward gate
and `reward == 1.0` terminal semantics are unchanged):

- **Run start** — retrieve the most relevant skills + lessons for the prompt and
  inject them into the modeler's instructions (`build_memory_preamble`).
- **On each failing turn** — normalize each `failed_constraints` entry to a
  stable signature (`core/lesson_signature.py`), recall matching lessons, and
  attach them to the tool's JSON feedback (`recall_lessons`).
- **Across turns** — when a prior failure is resolved, record a new lesson and
  give positive feedback to the lesson that helped; give negative feedback when
  a surfaced lesson didn't help (`learn_from_transition`).
- **On success** — distill one reusable skill from the timeline via an LLM
  abstraction pass (`harvest_skill`).

**Robustness.** Embeddings use `gemini-embedding-001` (768-dim, L2-normalized)
when a genai client + API key are available, and **degrade gracefully to
keyword + signature matching** otherwise — so the committed seed DB (NULL
embeddings) works offline and back-fills lazily. Growth is bounded by
deduplication (merge near-duplicates), Laplace-smoothed confidence, exponential
time decay, a confidence floor on retrieval, and per-table eviction.

**Seed knowledge.** `memory/seed_memory.py` ships 6 starter lessons (common
CadQuery failure modes) and **17 starter skills** spanning primitives (box,
cylinder, sphere, polygon prism, hollow tube), holes (through, counterbore,
countersink, grid pattern), edge treatments (fillet/chamfer all or top edges),
and advanced features (shell/container, revolve, loft, union, mirror). Every
skill snippet is executed against CadQuery 2.8.0 by
`tests/test_seed_skills.py`, so the seeds are **known-valid, not hallucinated**.
The seeded `memory/meda_memory.db` is committed so a fresh clone starts
informed. Regenerate with `python memory/seed_memory.py`.

### Research grounding
- **Voyager** — skill library of executable code + self-verification. [arXiv:2305.16291](https://arxiv.org/abs/2305.16291)
- **Reflexion** — verbal self-reflection in episodic memory. [arXiv:2303.11366](https://arxiv.org/abs/2303.11366)
- **Generative Agents** — memory stream + reflection + retrieval (recency/importance/relevance). [arXiv:2304.03442](https://arxiv.org/abs/2304.03442)
- **ExpeL** — cross-task insight extraction without fine-tuning. [arXiv:2308.10144](https://arxiv.org/abs/2308.10144)
- **Agent Workflow Memory** — induced reusable workflows from past trajectories. [arXiv:2409.07429](https://arxiv.org/abs/2409.07429)

### ADK references
- Memory: <https://google.github.io/adk-docs/sessions/memory/> · Session: <https://google.github.io/adk-docs/sessions/session/> · State: <https://google.github.io/adk-docs/sessions/state/>
- Vertex AI Memory Bank (managed alternative we deliberately avoided to stay cloud-free): <https://cloud.google.com/blog/topics/developers-practitioners/remember-this-agent-state-and-memory-with-adk>
- Gemini embeddings: <https://ai.google.dev/gemini-api/docs/embeddings>

*Future option:* swap `InMemorySessionService` → `DatabaseSessionService`
(`sqlite+aiosqlite://`) to also persist ADK sessions; the SQLite lessons/skills
store above is the load-bearing learning mechanism and does not depend on it.

---

## 3. Per-iteration rendering & visual feedback

**Before:** a render happened only when the B-Rep reward was already 1.0 (and
even then it crashed — see C1). Intermediate iterations got **no visual
feedback**, so the agent was flying blind between turns.

**Now** (`render_current_model` in `core/reasoning_core.py`):

- Render an orthographic collage on **every successful compile**, to a
  per-iteration `iter_{n}.png` (no stale reads).
- **Skip re-rendering** when the compiled code is unchanged (SHA-1 geometry
  hash) to control Open3D cost.
- Attach the **latest** render to the next agent turn via a
  `before_model_callback` (best-effort; wrapped so an unsupported ADK version
  simply skips attachment) — so the modeler/critic can visually course-correct.
- The heavyweight LLM visual critique still runs only at `reward == 1.0` (cost
  control) but now reuses the captured render bytes.
- **Streamlit** shows each iteration's render live via a render callback.

---

## Tests

- `tests/test_reward_engine.py` — reward gating + graded scoring (from PR #2).
- `tests/test_learning_store.py` — signature mapping, lesson record/retrieve,
  dedup-merge, confidence feedback, keyword-fallback skill retrieval (all run
  offline, no API key).

The full ADK/Streamlit loop and Open3D rendering require the synced `uv`
environment (open3d, cadquery, google-adk, google-genai) and a Gemini API key;
those paths are validated structurally here, not executed in CI.
