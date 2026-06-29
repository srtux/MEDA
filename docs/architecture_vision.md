# MEDA Architecture Vision

MEDA is evolving from a prompt-to-script demo into a verifiable, research-grade CAD synthesis platform. The north star is simple: every generated model should be parametric, executable, measurable, inspectable, and easy to iterate.

## Current execution loop

1. The user supplies a natural-language design goal and optionally an engineering drawing.
2. `ReasoningCADCore` builds a shared parametric canvas and asks the orchestrator agent to delegate work.
3. `CADModelerAgent` adds named dimensions and CadQuery feature steps to the canvas.
4. `VisualCriticAgent` compiles the current script in the sandbox and returns metrics, errors, and visual critique feedback.
5. MEDA saves the final Python, STL, STEP, and diagnostic JSON artifacts for the run.

## What makes MEDA stronger than direct code generation

- **Feature timeline instead of one-shot scripts:** geometry is built through editable steps, so agents can modify a single feature without rewriting the full model.
- **Sandboxed compilation:** generated code is executed in a subprocess with a timeout, artifact directory, and topology extraction.
- **Metric-gated reward:** compilation success is necessary but not sufficient; volume, face count, edge count, vertices, and center of mass can gate acceptance.
- **Multiview visual critique:** successful B-Rep compilation can still fail if rendered orthographic views do not match the design intent.
- **Diagnostics by default:** each run records prompt, constraints, metrics, failed checks, final code, and suggested color.

## Near-term research upgrades

### 1. CAD intermediate representation

Introduce an explicit JSON IR between natural language and CadQuery. The IR should model parameters, sketches, features, constraints, coordinate frames, and semantic intent. Agents would propose and revise IR first, then compile IR to CadQuery. This reduces prompt drift and enables deterministic validation.

### 2. Constraint solver pass

Before code generation, infer dimensions and relations into a constraint graph. Use symbolic equations where possible and numeric solving where needed. The modeler should receive resolved, named parameters rather than rediscovering arithmetic during feature coding.

### 3. Retrieval-augmented CAD memory

Store successful prompt → IR → CadQuery → metrics traces. At generation time, retrieve similar designs and reuse proven feature templates. This should improve speed, consistency, and complex-feature reliability.

### 4. Parallel candidate generation

Generate multiple candidate feature timelines with different decomposition strategies. Execute them in parallel, rank by compilation, topology, visual match, simplicity, and parametric editability, then refine only the best candidates.

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
