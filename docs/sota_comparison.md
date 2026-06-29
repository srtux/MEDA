# MEDA in the LLM-to-CAD Landscape: A State-of-the-Art Comparison

*Last updated: 2026-06-29*

This document situates MEDA's technique within the current (2025–2026) research
landscape for converting natural language into CAD models, compares it against
academic systems and commercial products (e.g. **adam.new**, **CADAM**), and
records the rationale for the distance-based reward upgrade that accompanies it.

---

## 1. What MEDA actually does (current code)

> **Note:** the published IDETC 2025 paper describes an AutoGen + GPT-4o,
> 7-agent group-chat system. The current code has been refactored to a
> **Google ADK** orchestration with a *Unified Reasoning Core*. This document
> describes the current code.

MEDA is a **training-free, agentic, code-generation loop with B-Rep + visual
feedback**:

1. **Constraint inference** — an LLM call parses the prompt into target B-Rep
   metrics (`volume`, `num_faces`, `num_edges`, `center_of_mass`).
2. **Multi-agent orchestration (Google ADK)** — three `LlmAgent`s with
   delegation/handoff:
   - `MEDAOrchestrator` (coordinator, no tools)
   - `CADModelerAgent` (`add_parameter`, `set_parameter`, `add_feature`,
     `modify_feature`, `remove_feature`)
   - `VisualCriticAgent` (`run_cad_execution`)
3. **Intermediate representation = CadQuery Python**, built incrementally on a
   `Canvas` — a parametric *feature timeline* plus named parameters, edited
   surgically via tool calls (`core/canvas.py`).
4. **Sandbox execution** (`core/sandbox.py`) — runs the script in a subprocess,
   extracts B-Rep topology metrics (volume, area, face/edge/vertex counts,
   center of mass).
5. **Reward gating** (`core/reward_engine.py`) — multiplicative
   `R = R_exec × R_geom`.
6. **Multi-view visual critique** — orthographic collage (iso/top/front/right)
   evaluated by a multimodal critic returning `{match, critique}`.
7. **Loop** until `reward == 1.0`.

**Models:** Gemini (`gemini-3.5-flash` default; GPT-4o in the paper). **Kernel:**
OpenCASCADE via CadQuery. **Eval:** Open3D point-cloud metrics (GICP, Hausdorff,
IoGT) in `eval_metrics/`.

---

## 2. The current research landscape (2025–2026)

The field has split into three families. MEDA lives in the third.

### A. Fine-tuned domain models (predict CAD sequence / CadQuery directly)
Models trained specifically for text→CAD, emitting parametric command
sequences or CadQuery code.

- **Text2CAD** — VLM captions DeepCAD shapes, then trains a transformer to
  predict CAD command sequences from text.
  ([arXiv:2505.19490](https://arxiv.org/pdf/2505.19490))
- **Text-to-CadQuery** — fine-tunes LLMs to emit CadQuery Python directly.
  ([arXiv:2505.06507](https://arxiv.org/pdf/2505.06507))
- **CADFusion** — strong primitive classification via pretrained LMs; weaker on
  precise parameter control. (See Text2CAD-Bench below.)
- **CAD-MLLM** — multimodal (text / image / point cloud) parametric CAD
  generation.
- **CAD-Coder** — text→CadQuery with chain-of-thought and a geometric reward.

### B. Reinforcement-learning post-training (the current frontier)
Post-trains a model on executable/geometric reward signals — the direction
currently topping benchmarks.

- **cadrille** — multimodal CAD reconstruction; SFT followed by RL. Published at
  **ICLR 2026**. ([arXiv:2505.22914](https://arxiv.org/pdf/2505.22914))
- **ReCAD** — RL-enhanced parametric CAD generation with VLMs.
  ([arXiv:2512.06328](https://arxiv.org/pdf/2512.06328))
- **TOOLCAD** — tool-using LLMs for text→CAD with RL.
  ([arXiv:2604.07960](https://arxiv.org/pdf/2604.07960))
- **CME-CAD** — heterogeneous collaborative multi-expert RL for CAD code.
  ([arXiv:2512.23333](https://arxiv.org/pdf/2512.23333))
- **From Intent to Execution** — multimodal chain-of-thought RL for precise CAD
  code. ([arXiv:2508.10118](https://arxiv.org/pdf/2508.10118))

### C. Training-free agentic loops with visual feedback — **MEDA is here**
Places a (multimodal) LLM inside a verify-and-refine loop, no fine-tuning.

- **CADCodeVerify** — VLM generates validation questions and uses visual
  feedback to refine CAD code. (MEDA's headline baseline.)
- **Seek-CAD** — self-refined CAD generation using visual + CoT feedback from
  DeepSeek-R1 / Gemini; a near-twin of MEDA's loop.
  ([arXiv:2505.17702](https://arxiv.org/pdf/2505.17702))
- **CADDesigner** — conceptual CAD design via a general-purpose agent.
  ([arXiv:2508.01031](https://arxiv.org/html/2508.01031v1))
- **EvoCAD** — evolutionary CAD code generation guided by VLMs.
  ([arXiv:2510.11631](https://arxiv.org/pdf/2510.11631))
- **CAD-Assistant** — tool-augmented MLLM agent in a feedback loop.

### Supporting infrastructure (benchmarks, verifiers, grounding)
- **Text2CAD-Bench** — benchmark for LLM-based text-to-parametric CAD; executes
  generated CadQuery in an isolated env and converts STEP→point cloud for
  scoring. ([arXiv:2605.18430](https://arxiv.org/html/2605.18430))
- **CAD-Judge** — efficient morphological grading/verification for text→CAD.
  ([arXiv:2508.04002](https://arxiv.org/html/2508.04002v1))
- **B-Rep primitive grounding** — high-fidelity CAD via LLM program generation
  grounded on text-based B-Rep primitives.
  ([arXiv:2603.11831](https://arxiv.org/abs/2603.11831))
- **Survey: LLMs for Computer-Aided Design.**
  ([arXiv:2505.08137](https://arxiv.org/pdf/2505.08137))

### Commercial products
- **adam.new** — browser-based AI CAD chat copilot; YC alum, raised $4.1M
  (Oct 2025), pivoting toward an Onshape copilot for parametric mechanical
  design. ([thesis](https://adam.new/thesis),
  [TechCrunch](https://techcrunch.com/2025/10/31/yc-alum-adam-raises-4-1m-to-turn-viral-text-to-3d-tool-into-ai-copilot/))
- **CADAM** — free text/image→3D model generator, generation-focused.
  ([overview](https://www.scriptbyai.com/text-image-3d-model-cadam/))

MEDA is essentially the **open, transparent, verifiable research analog** of the
commercial tools: it exposes executable parametric CadQuery + B-Rep metric
checking, which adam.new / CADAM largely do not.

---

## 3. Is MEDA "latest and greatest"?

**Modern and well-built, but one generation behind the frontier.** Honest
assessment:

### Genuinely current / strong
- **Parametric feature-timeline canvas** with *surgical* edits (modify/remove a
  single step) rather than regenerating the whole script — more advanced than
  most agentic baselines.
- **B-Rep topology gating** as a hard reward — closer to CAD-Judge / B-Rep
  grounding than pure visual critique.
- **Multi-view orthographic critique** (vs a single render) — aligns with the
  "automated visual feedback wins" finding across the literature.
- **ADK multi-agent handoffs** — a current orchestration style.

### Where the frontier has moved past it
1. **Training-free is now being beaten on benchmarks by RL-fine-tuned models**
   (cadrille, CAD-Coder, ReCAD). MEDA's reward only steers an in-context loop;
   nothing is learned.
2. **Brittle exact-match constraints** *(addressed by this PR — see §4).*
   Requiring an LLM to predict the *exact* `num_faces`/`num_edges` of an
   unbuilt model and then demanding strict integer equality is unreliable — a
   fillet or chamfer shifts counts unpredictably.
3. **No retrieval/grounding of the CadQuery API** — newer systems ground on
   B-Rep primitives or API docs to cut hallucinated methods.
4. **Sandbox isn't truly isolated** — a `subprocess`, not a container.
5. **Single-attempt loop** — no judge-panel or evolutionary search (cf. EvoCAD).

### Recommended upgrade path (highest leverage first)
1. ✅ **Distance-based / graded geometric reward** (this PR).
2. **RAG over the CadQuery API** into the modeler agent's context.
3. **RL or rejection-sampling fine-tune** of the modeler on the computed reward
   (the cadrille / CAD-Coder leap).
4. **Containerize the sandbox.**
5. **Refresh the README** to match the current ADK/Gemini architecture.

---

## 4. The reward-engine upgrade in this PR

**Problem.** The previous `RewardEngine` used a binary multiplicative gate where
integer topology constraints (`num_faces`, `num_edges`, `num_vertices`) required
**exact equality**. A model off by a single edge scored **0.0** — identical to a
compile failure — so the agent got no gradient and the LLM-inferred exact counts
were an unreliable target in the first place.

**Change.** `core/reward_engine.py` now provides:

- **Graded `geom_score` ∈ [0, 1]** alongside the binary gate, so the agent gets
  partial-credit feedback ("0.82, faces off by one") instead of all-or-nothing.
  The `reward == 1.0` terminal semantics the loop relies on are preserved.
- **`topology_tolerance`** — optional integer slack on count constraints, so a
  stray fillet edge no longer zeroes the reward.
- **Distance mode** — when `constraints` provides a `reference` (an STL path or
  an explicit point list) plus the generated STL, the reward scores symmetric
  **Chamfer + Hausdorff** distance on normalized point clouds (matching the
  `eval_metrics/` benchmark methodology) and passes the gate when Chamfer falls
  under `distance_threshold`. This reuses clean, side-effect-free helpers in the
  new `core/geometry_metrics.py`.

The richer signal (`geom_score`, `distances`) is now surfaced to the agent in
`run_cad_execution`'s JSON response.

**Why distance mode is conditional.** In pure text→CAD there is no ground-truth
mesh at generation time, so distance scoring only activates when a reference is
supplied (image/reference-mesh runs or the eval harness). For pure-text runs the
brittleness fix is the *graded* topology score and the `topology_tolerance`
slack.

**Tests.** `tests/test_reward_engine.py` covers the topology path (no open3d
dependency): compile-failure gating, no-constraint pass, volume tolerance,
partial-credit scoring, and integer slack.

---

## Sources

- Survey — Large Language Models for Computer-Aided Design — https://arxiv.org/pdf/2505.08137
- Text2CAD-Bench — https://arxiv.org/html/2605.18430
- Text-to-CadQuery — https://arxiv.org/pdf/2505.06507
- Automated CAD Modeling Sequence Generation (Text2CAD) — https://arxiv.org/pdf/2505.19490
- cadrille (ICLR 2026) — https://arxiv.org/pdf/2505.22914
- ReCAD — https://arxiv.org/pdf/2512.06328
- TOOLCAD — https://arxiv.org/pdf/2604.07960
- CME-CAD — https://arxiv.org/pdf/2512.23333
- From Intent to Execution (multimodal CoT RL) — https://arxiv.org/pdf/2508.10118
- Seek-CAD — https://arxiv.org/pdf/2505.17702
- CADDesigner — https://arxiv.org/html/2508.01031v1
- EvoCAD — https://arxiv.org/pdf/2510.11631
- CAD-Judge — https://arxiv.org/html/2508.04002v1
- B-Rep primitive grounding / high-fidelity CAD — https://arxiv.org/abs/2603.11831
- adam.new — https://adam.new/thesis · https://techcrunch.com/2025/10/31/yc-alum-adam-raises-4-1m-to-turn-viral-text-to-3d-tool-into-ai-copilot/
- CADAM — https://www.scriptbyai.com/text-image-3d-model-cadam/
