# MEDA Architecture Vision

MEDA is evolving from a prompt-to-script demo into a verifiable, research-grade CAD synthesis platform. The north star is simple: every generated model should be parametric, executable, measurable, inspectable, and easy to iterate.

## Current generation modes

MEDA exposes three runtime paths in `ReasoningCADCore.run_design_loop`:

1. **Default ADK feature-timeline loop.** A user supplies a design goal plus optional image/constraints. `ReasoningCADCore` resets or keeps a shared `Canvas`, retrieves CadQuery API references plus prior lessons/skills, and builds a concise user message. `MEDAOrchestrator` delegates CAD edits to `CADModelerAgent`; the modeler mutates the feature timeline with parameter and feature tools. `VisualCriticAgent` compiles that timeline with `run_cad_execution`, which executes in the sandbox, extracts B-Rep metrics, saves iteration artifacts, renders orthographic views, and runs visual critique once geometric reward reaches 1.0. The loop stops on reward 1.0, `max_iterations`, or an event cap.
2. **Fast Mode.** For low-latency exploration, MEDA asks the model for a complete CadQuery script and retries up to three times with compile-error feedback. Fast Mode still uses the sandbox and now promotes successful runs into the same iteration/render artifacts, but it does not run the full modeler/critic handoff loop.
3. **Multi-candidate search.** When `num_candidates > 1`, MEDA generates up to five complete scripts using diverse strategy hints (primitive-first, sketch-extrude, boolean composition, revolve/loft, minimal), executes them serially in the shared sandbox, ranks them by compile success, continuous geometry score, positive-volume validity, and simplicity, then re-executes and promotes the winning script. This is currently a geometry-first selector; future work can convert candidates into editable feature timelines and add top-candidate VLM critique/refinement.

All modes write the winning/final Python script and canonical CAD artifacts (`001.py`, `001.stl`, `001.step` when export succeeds). The default loop also writes a diagnostic JSON snapshot; candidate search returns candidate diagnostics in the result object.

## Current reliability layers

- **CadQuery API grounding:** retrieved API snippets and prior reusable skills are injected before modeling.
- **B-Rep selector grounding:** `inspect_current_model` reports measured faces/edges and selector hints before fillets, chamfers, shells, and face-targeted holes.
- **Sandbox hardening:** generated code is validated with an AST allow-list, stale artifacts are deleted before execution, secrets are scrubbed from the child environment, and POSIX resource limits are applied where available. This is a substantial mitigation but not a substitute for container/nsjail isolation in hosted multi-user deployment.
- **Reward scoring:** compile/model-build success gates reward; configured constraints can score volume, face/edge/vertex counts, center of mass, or reference-shape distances. Surface area, solid count, and validity are emitted as metrics for downstream inspection, but not all are default reward gates.

## Near-term research upgrades

### 1. CAD intermediate representation

Introduce an explicit JSON IR between natural language and CadQuery. The IR should model parameters, sketches, features, constraints, coordinate frames, and semantic intent. Agents would propose and revise IR first, then compile IR to CadQuery. This reduces prompt drift and enables deterministic validation.

### 2. Constraint solver pass

Before code generation, infer dimensions and relations into a constraint graph. Use symbolic equations where possible and numeric solving where needed. The modeler should receive resolved, named parameters rather than rediscovering arithmetic during feature coding.

### 3. Retrieval-augmented CAD memory

Partially implemented today: MEDA retrieves curated CadQuery API entries, reusable skills, lessons, and compact trajectory tips at run start. Future work is full prompt → IR → CadQuery → metrics trace retrieval once the structured IR exists.

### 4. Candidate refinement after ranking

Partially implemented today: MEDA can generate and rank multiple complete CadQuery scripts. Future work is to represent those candidates as editable feature timelines, run isolated true-parallel sandboxes, apply VLM critique to the top candidates, and refine the best candidates rather than simply promoting the highest-scoring script.

### 5. Learned visual reward model

Replace brittle binary visual critique with a calibrated reward model trained on rendered views, prompt labels, and human/metric feedback. Keep the LLM critic as an explanation layer, not the only scoring mechanism.

### 6. Geometry regression suite

Promote benchmark prompts and ground-truth meshes into automated tests. Track compile rate, Hausdorff distance, Chamfer distance, IoGT, latency, token cost, and number of repair turns for every model/provider.

## Product upgrades

- Real-time run timeline with agent handoffs, tool calls, compile status, and score cards.
- Editable parameter table with units, descriptions, min/max bounds, and regeneration history.
- Artifact browser for Python, STEP, STL, screenshots, and diagnostics.
- Gallery of canonical prompt templates and benchmark examples.
- Model/provider comparison dashboard for speed, cost, and quality.

## Engineering principles

- Keep execution deterministic wherever possible.
- Prefer structured CAD state over unstructured generated text.
- Fail fast on invalid code, missing models, and missing artifacts.
- Make every acceptance decision reproducible from saved diagnostics.
- Treat the UI as a professional CAD copilot, not a chat transcript.
