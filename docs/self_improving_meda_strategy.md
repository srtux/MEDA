# Self-Improving MEDA Strategy

This note captures how MEDA can compete with, and potentially surpass, current text-to-CAD systems by making self-improvement a first-class capability rather than a post-hoc evaluation report.

## Recent PR trajectory

The recent codebase direction is coherent:

- `26d2b5c` introduced the unified reasoning core, sandbox, multi-view visual critique, backoff, isolated workspaces, and UI log streaming.
- `eda528f` documented architecture and code-review recommendations.
- `ef611ec` migrated orchestration from AutoGen/Conda to Google ADK and UV.
- `5b7ff29` aligned Streamlit model/key handling with `ReasoningCADCore`.
- `791c2a9` polished the UI and added the architecture vision.

The missing bridge was persistent learning across runs. This PR starts that bridge with lightweight trajectory memory.

## Competitive landscape

Current text-to-CAD systems usually improve an individual sample by re-prompting after execution or visual feedback. CADCodeVerify is the clearest competitor in this family: it uses VLM-generated validation questions and visual feedback to correct CAD code. Newer benchmarks such as Text2CAD-Bench make the field harder by evaluating geometric complexity, freeform topology, and non-mechanical domains.

The gap: most systems are **episodic**. They can repair the current design but do not systematically convert every run into reusable procedural knowledge.

## MEDA's self-improvement thesis

MEDA should become a closed-loop CAD research engine:

1. **Generate** multiple parametric candidates from text and image input.
2. **Execute** every candidate in a sandbox.
3. **Score** with B-Rep metrics, visual critique, editability, latency, and cost.
4. **Reflect** on why each trajectory succeeded or failed.
5. **Store** compact memories and reusable skills with provenance.
6. **Retrieve** relevant lessons for future prompts.
7. **Distill** recurring successful trajectories into templates, constraints, and eventually fine-tuning/evaluation data.

## Memory layers

### Episodic trajectory memory

Store prompt, generated code, metrics, failure reasons, screenshots, repair turns, and final outcome for each run. Use it to retrieve similar prior attempts.

### Semantic CAD memory

Convert trajectories into generalized design rules: e.g. "for centered through-holes, build the base solid first, select the top face, and call `.hole(diameter)` instead of subtracting an ad hoc cylinder."

### Procedural skill memory

Promote repeated successful patterns into named skills: sketch-extrude, revolve, patterned holes, shelling, fillets/chamfers, ribs, threads, and assemblies. Skills should include preconditions, CadQuery templates, validation metrics, and common failure repairs.

### Benchmark memory

Track performance per model/provider/prompt family: compile rate, repair turns, point-cloud distance, Hausdorff distance, IoGT, latency, and token cost. Route future tasks to the best model/strategy for that family.

## How MEDA can beat competitors

- **From verification to learning:** competitors verify and correct; MEDA should verify, correct, remember, retrieve, and distill.
- **From one candidate to search:** run diverse CAD strategies in parallel and use geometry-aware ranking.
- **From opaque prompts to CAD IR:** represent sketches, features, constraints, and coordinate frames explicitly before codegen.
- **From visual critique only to multi-signal reward:** combine B-Rep metrics, rendered views, code simplicity, parametric editability, and benchmark distance metrics.
- **From static agents to evolving agents:** update agent prompts with retrieved memories and promote high-confidence memories into durable skills.

## Implementation path

1. Use the new `CADMemoryStore` as a dependency-free baseline.
2. Add embeddings/vector search when the memory file grows beyond simple keyword retrieval.
3. Add a reflection agent that converts diagnostics into structured success, recovery, and optimization tips.
4. Add a skill registry for reusable feature generators with tests.
5. Add a benchmark runner over CADPrompt, Text2CAD-Bench-style tasks, and MEDA ground-truth STL assets.
6. Report ablations: no-memory vs memory retrieval vs skills vs memory+skills+parallel search.

The strongest publishable claim would be: **a multi-agent, multimodal, self-improving CAD system improves compile rate, geometric similarity, and repair efficiency over visual-feedback-only baselines by reusing trajectory-derived memory and procedural skills.**
