# Text-to-CAD Competitive Landscape & MEDA Positioning (through June 2026)

*Last updated: 2026-06-29*

This is the comprehensive landscape survey requested for MEDA: a full competitive
map of Text-to-CAD across **open source**, **startups / proprietary products**,
and **academic / industry research**, with verified citations, a frank assessment
of where MEDA sits, and a prioritized improvement roadmap.

It supersedes and corrects the narrower [`sota_comparison.md`](./sota_comparison.md)
(see [§9 Citation corrections](#9-citation-corrections-to-the-prior-doc)). It was
compiled from primary sources (arXiv, GitHub, HuggingFace, company press, and the
[Xometry](https://xometry.pro/en/articles/text-to-cad-tools-test/) and
[Leo AI](https://www.getleo.ai/blog/text-to-cad-tools-comparison-guide) hands-on
tests). Every arXiv ID below was resolved against arXiv; disputed claims are
flagged inline.

---

## 1. Executive summary

- **The field's central fault line is representation: mesh vs. parametric/B-Rep.**
  Most "AI 3D" tools that go viral (Meshy, Tripo, Rodin, Luma Genie, Spline,
  Sloyd) output **polygon meshes** — not editable CAD. They are *not* MEDA's
  competitors. MEDA competes in the much smaller **parametric / B-Rep** arena
  (editable feature history, STEP output), where the real players are Zoo,
  Adam, Spectral Labs SGS-1, the CAD incumbents, and the research community.

- **MEDA's research family (training-free agentic loops with visual feedback) is
  real and active**, but it is **no longer where the benchmark frontier lives.**
  As of mid-2026 the SOTA on the standard benchmarks (DeepCAD, Fusion360, CC3D,
  Text2CAD) is held by **RL-fine-tuned models** — cadrille (ICLR 2026), ReCAD
  (AAAI 2026), CAD-Coder, CME-CAD — all using GRPO with geometric/execution
  rewards on a Qwen-VL backbone. MEDA *learns nothing*; its reward only steers an
  in-context loop.

- **Independent hands-on tests agree no tool is production-ready.** Both Xometry
  ("We Tested 7 Text-to-CAD Tools") and Leo AI (Zoo vs Adam vs SGS-1) conclude
  that current text-to-CAD is good for *concept/ideation and simple mechanical
  parts* but fails on complex multi-feature geometry. This is the gap MEDA can
  credibly target — verifiable, editable, correct simple-to-mid parts — rather
  than chasing freeform generation.

- **MEDA's genuine differentiators** are worth protecting: an editable
  **parametric feature-timeline canvas** with *surgical* per-step edits, **hard
  B-Rep topology gating** as reward, **multi-view orthographic** critique, and a
  fully **open, transparent, verifiable** pipeline. On its own benchmark
  (CADPrompt) MEDA reports best-in-class point-cloud distance (0.0555) and 99%
  compile rate vs its baseline CADCodeVerify.

- **Highest-leverage improvements:** (1) retrieval-grounding of the CadQuery API,
  (2) rejection-sampling / RL fine-tune on MEDA's own computed reward (the
  cadrille/CAD-Coder leap), (3) advanced-feature grounding (fillet/chamfer/shell
  via face/edge selection — the frontier per Text2CAD-Bench and the B-Rep
  grounding paper), (4) a real containerized sandbox, (5) parallel candidates +
  judge panel (EvoCAD-style). See [§8](#8-how-meda-can-improve--prioritized-roadmap).

---

## 2. The fundamental divide: mesh vs. parametric / B-Rep CAD

| | **Mesh / "AI 3D"** | **Parametric / B-Rep CAD** (MEDA's arena) |
|---|---|---|
| Output | Triangle/quad mesh + PBR textures | Feature history → B-Rep solid → STEP |
| Editable? | Push/pull polygons only | Dimensions, constraints, feature tree |
| Manufacturable? | 3D-print / render / games | Engineering / CNC / assemblies |
| Examples | Meshy, Tripo, Rodin, Luma Genie, Spline, Sloyd | MEDA, Zoo, Adam, SGS-1, CADFusion, cadrille |
| Typical kernel | Diffusion / Gaussian splat / LRM | OpenCASCADE, Parasolid, ACIS, custom |

**Why it matters:** a mesh of a bracket cannot be re-dimensioned, constrained,
or dropped into an assembly; a parametric CAD model can. Mesh tools are far
better funded and more viral, but they solve a different problem. Throughout this
doc, "competitor" means a **parametric** system unless explicitly noted.

---

## 3. Taxonomy used in this document

**Commercial / proprietary** (§4): CAD-native startups (Zoo, Adam, SGS-1),
incumbents (Autodesk, Siemens, PTC, Dassault, Onshape), and mesh generators
(non-competitors, catalogued for completeness).

**Open source** (§5): the underlying libraries (CadQuery, build123d, OpenSCAD,
FreeCAD-MCP) and the research code/weights/datasets that are actually downloadable.

**Academic / industry research** (§6), in four families:
- **A — Fine-tuned domain models** (predict CAD command sequence or CadQuery directly)
- **B — RL post-training** (the current benchmark frontier)
- **C — Training-free agentic loops with visual/execution feedback** ← **MEDA is here**
- **D — Benchmarks, verifiers, datasets, grounding** (the infrastructure)

---

## 4. Commercial / startup / proprietary

### 4A. CAD-native startups (MEDA's real commercial competitors)

#### Zoo (formerly KittyCAD) — `zoo.dev`
- **What:** Full hardware-design platform: the **KittyCAD Design API**, the
  **ML-ephant** ML API, **KCL** (KittyCAD Language — a domain-specific,
  *parametric* CAD programming language), and **Zoo Design Studio** (browser +
  desktop). Text-to-CAD shipped Dec 2023; in **Jan 2026** they shipped a
  conversational agent **"Zookeeper"** with Design Studio v1.1.
- **Representation:** Genuinely **parametric** — generates KCL and exports STEP
  (plus FBX/GLB/OBJ/PLY/STL/KCL).
- **Funding/backers:** incubated at Embedded Ventures; investors include Madrona,
  Liquid 2, Nat Friedman, Tom Preston-Werner's family office.
- **Hands-on verdict (Xometry):** accurate, editable models for *simple
  functional parts* (e.g. a dimensioned cylinder); **incoherent on organic/
  freeform** (cat, tree, volcano island). Leo AI noted some outputs are
  effectively mesh STL rather than clean parametric solids and that it struggles
  with multi-feature complexity.
- **vs MEDA:** Zoo is the closest *productized* analog — but it is a closed
  platform built on a custom DSL (KCL). MEDA is open, uses standard CadQuery, and
  adds explicit B-Rep metric + multi-view verification that Zoo does not expose.

#### Adam (AdamCAD) — `adam.new`
- **What:** "CAD copilot for hardware teams." Web text-to-CAD that emits
  **parametric** geometry; **Onshape** and **Fusion** plugins inject generated
  models *into the native feature tree* (avoiding export/reimport). YC W25.
- **Funding:** **$4.1M seed** (announced 2025-10-31, led by TQ Ventures) after a
  viral text-to-3D launch (10M+ impressions).
- **Output:** STL, OBJ, **SCAD** (OpenSCAD), STEP coming. Open-source **CADAM**
  repo (~3.5k★). SolidWorks/Fusion native integrations + enterprise copilot
  planned 1H2026.
- **Founders:** Zach Dive, Aaron Li (UC Berkeley).
- **Hands-on verdict (Leo AI):** generates *inspectable, modifiable* parametric
  code (CadQuery/OpenSCAD) — a plus for engineers — but complex geometries fail
  and debugging generated code has a learning curve.
- **vs MEDA:** Adam is the startup whose *approach* most resembles MEDA's
  (LLM → parametric code, code is the artifact). MEDA's edge is the closed-loop
  **verification** (compile + topology + visual) and the surgical feature
  timeline; Adam's edge is **distribution** (native plugins in Onshape/Fusion).

#### Spectral Labs — SGS-1 — `spectrallabs.ai`
- **What:** Billed as **"the first generative model for structured CAD."** Given
  an **image or 3D mesh** (and assembly/sketch context), SGS-1 outputs **B-Rep
  parts directly in STEP** that open cleanly in SolidWorks/Fusion. Strong on
  reverse-engineering scans/STLs to parametric STEP, sketch→3D, and
  in-assembly-context generation.
- **Limits (self-stated + Leo AI):** struggles with organic/complex curvature,
  thin structures, limited 3D resolution, no full assemblies one-shot; quality
  degrades sharply with complexity; warped surfaces cause downstream assembly
  issues. **SGS-2** in progress. Public HuggingFace Space demo.
- **vs MEDA:** SGS-1 is the most direct **B-Rep-native** competitor and is a
  *trained foundation model* (not an agentic loop). It is primarily
  image/mesh→CAD, less text→CAD, which is complementary to MEDA's text focus.

#### Aurorin CAD — `aurorincad.com` (newest entrant, YC W26)
- **What:** A **from-scratch, AI-native professional mechanical CAD** app
  combining a traditional editing UI with a chat interface ("a part that takes
  20 min in SolidWorks takes seconds"). Founder **Michael Baron** (ex-SpaceX
  Raptor sim / Dragon guidance; ex-Apple GPU drivers).
- **Approach:** a **custom GPU-optimized parametric + B-Rep kernel built from
  scratch** (explicitly rejecting Parasolid/ACIS), with chat → editable solid
  geometry. Mac + Windows.
- **Funding:** ~$500K at a $40M valuation (YC W26); essentially a one-person team
  at launch.
- **vs MEDA:** the most ambitious *technical* bet (its own kernel); years from
  parity, and the AI layer alone is replicable. MEDA's open verification loop is
  orthogonal and could in principle sit on top of any kernel.

#### Leo AI — `getleo.ai` (engineering copilot, now CAD-generating)
- **What:** AI copilot for mechanical engineers (Q&A, part retrieval, concept
  generation) built on a proprietary **"Large Mechanical Model."** Crucially, Leo
  now **generates full CAD assemblies from text** — a major leap past its
  "concept images only" status in the (older) Xometry test.
- **Funding/traction:** **$9.7M seed** (Sep 2025, led by Flint Capital); customers
  cited include Scania, HP, Siemens.
- **vs MEDA:** Leo's value is the engineering-knowledge + retrieval layer (the
  "reuse validated geometry" thesis the Leo AI test itself argues for); MEDA's is
  verifiable from-scratch part synthesis. Complementary, not identical.

#### CAD incumbents (parametric, adding AI)
- **Autodesk** — **Project Bernini** (research, *not* shipping): generative model
  for 3D shapes from text / images / sketches / voxels / point clouds; ~10M
  shapes, ~3B params. Productizing via **Neural CAD** (announced AU 2025,
  pre-GA) — a foundation-model category aiming to generate **editable B-Rep
  geometry from a prompt** in Fusion/Forma — and **Autodesk Assistant** in Fusion
  (prompt → native editable geometry), plus shipping Sketch AutoConstrain,
  generative design, and Fusion MCP servers.
- **PTC** — **Onshape AI Advisor** (GA Oct 2025, built with AWS): conversational
  assistant embedded in the design environment; guidance, troubleshooting, and
  **FeatureScript code generation**, with autonomous geometry generation on the
  roadmap. **Creo** adds the Generative Design Extension (cloud optimization →
  editable B-Rep). Onshape itself is the *integration target* for Adam et al.
- **Siemens** — **Designcenter X NX Copilot** (natural language → NX commands via
  Siemens knowledge); **Solid Edge 2026** ships an AI design copilot.
- **Dassault Systèmes** — **AURA** (AI assistant in SOLIDWORKS Connected,
  IP-in-tenant), joined by **Marie** (materials) and **LEO** (engineering/
  simulation, mid-2026) — *Dassault's LEO is distinct from the Leo AI startup* —
  plus multiple SOLIDWORKS AI agents announced at 3DEXPERIENCE World 2026.
- **vs MEDA:** incumbents own workflow, data, and trust; their generative AI is
  mostly copilot/guidance/optimization today, with full prompt→editable-part the
  2026+ frontier (Autodesk furthest along publicly). MEDA is a research analog of
  what these aim to ship — and far more transparent/verifiable.

### 4B. Mesh generators — **not** CAD competitors (catalogued for completeness)

These are the viral, well-funded "text/image-to-3D" tools. All output **polygon
meshes** (STL/OBJ/GLB/FBX/USDZ), **no feature tree, no constraints, no B-Rep**.
They do not compete with MEDA; included so the doc is exhaustive.

| Tool | Origin | Latest (≤Jun 2026) | Funding | Output | CAD? |
|---|---|---|---|---|---|
| **Meshy** | Sunnyvale (Ethan Hu) | Meshy-6 (Jan 2026) | ~$50M (Sequoia/GGV); ~$30M ARR | Mesh + PBR | No |
| **Tripo / VAST** | Beijing/SF (Simon Song) | H3.1/P1.0/W1.0 (Mar 2026) | $50M Mar 2026; unicorn; open TripoSR/SG/SF | Mesh | No |
| **Rodin (Hyper3D / Deemos)** | Shanghai/LA | Gen-2.5 (May 2026, 10M-poly) | A + Jun-2026 round (hundreds of M RMB); ByteDance | Mesh | No |
| **Luma Genie** | USA (Luma AI) | Genie (de-emphasized; Luma now video-first) | $900M Series C (Nov 2025, ~$4B val) | Mesh | No |
| **Spline AI** | Santiago, Chile | ongoing | ~$32M (Third Point) | Mesh (web 3D) | No |
| **Sloyd** | Oslo (Antler) | parametric-template + gen | €3M (a16z GAMES, Autodesk strategic) | Mesh (procedural) | No |

> Note: Sloyd's "parametric" = procedural *mesh* sliders, not CAD parameters;
> Tripo's "editable parts" = mesh segmentation, not a feature tree. Neither is
> B-Rep.

### 4C. What the independent hands-on tests found

- **Xometry — "We Tested 7 Text-to-CAD Tools – Are They Actually Useful for
  Engineers?"** (published **2025-08-27**; free tiers only; consistent prompts
  spanning functional vs. creative tasks — e.g. a 20 mm dimensioned cylinder vs.
  organic objects like a cat/tree/volcano island). The **7 tools and verdicts:**
  - **Zoo** — best of the group for simple parametric parts; accurate, *editable*,
    STEP export; standout parametric sliders + text-to-edit.
  - **AdamCAD** — usable models for simple parametric parts; refine/export.
  - **CADGPT** (via YesChat) — emits precise **DWG/DXF drafting code**; no preview/
    login; for code-comfortable users.
  - **CADScribe** (HEC Paris) — LLM → query language → **STEP**; praised for
    iterative chat; adjustable-dimension parametrics were on the roadmap.
  - **Vondy** — generic AI; claims DWG/DXF/STL/STEP but **inconsistent delivery**.
  - **Leo AI** — at test time, **concept images only, not CAD** (now outdated — Leo
    ships CAD assemblies, see §4A).
  - **OpenArt** — **not CAD**; blueprint-*style images* only.
  - *Conclusion:* five attempt genuine CAD-ready output, two are concept-only; none
    replaces professional CAD, but several help with simple prototyping/education.
  *(The article is older; Zoo's Zookeeper agent, Adam's $4.1M + Onshape plugin,
  Leo's CAD-assembly pivot, SGS-1, and Aurorin all postdate or update it.)*
- **Leo AI — Zoo vs Adam vs SGS-1 (hands-on):** **none achieved
  production-readiness.** Zoo → clean-enough mesh/concept output; Adam → editable
  parametric code but fails on complexity; SGS-1 → real B-Rep but quality
  degrades with complexity. Their editorial take: the bigger near-term win is
  **retrieving/reusing validated existing geometry** (PDM search), not generating
  new parts — a useful strategic caution for MEDA.

---

## 5. Open source

### 5A. Foundational libraries (the substrate everyone builds on)
- **CadQuery** — Python parametric CAD on OpenCASCADE B-Rep; exports STEP/STL.
  **MEDA's representation.** Apache-2.0.
- **build123d** — modern successor-style Python B-Rep API (CadQuery-adjacent),
  growing adoption.
- **OpenSCAD** — script-based CSG modeler; popular LLM target (Adam emits SCAD)
  but CSG, not feature-history B-Rep.
- **FreeCAD + MCP / Blender-MCP** — open-source parametric CAD/3D driven by LLM
  agents via Model Context Protocol; the open analog of an "agent drives the CAD
  app" approach (cf. CAD-Assistant, which uses FreeCAD's Python API).

### 5B. Research code / weights / datasets that are actually usable

| Project | Approach | Repr. | License | Stars (Jun 2026) | Weights? | Maturity |
|---|---|---|---|---|---|---|
| **DeepCAD** ([2105.09492](https://arxiv.org/abs/2105.09492)) | Transformer AE + latent-GAN | CAD seq (sketch-extrude) | MIT | ~781 | Yes | Foundational; dormant since 2024 |
| **SkexGen** ([2207.04632](https://arxiv.org/abs/2207.04632)) / **HNC-CAD** | VQ-VAE disentangled codebooks | CAD seq | MIT | ~144 | Yes | Stable, inactive since 2023 |
| **Text2CAD** ([2409.17106](https://arxiv.org/abs/2409.17106)) | BERT enc + Transformer dec | CAD seq | CC BY-NC-SA 4.0 | ~437 | Yes (+Gradio) | Reproducible; non-commercial; quiet since 2025 |
| **CAD-MLLM** ([2411.04954](https://arxiv.org/abs/2411.04954)) | Vicuna-7B + LoRA multimodal | CAD seq → STEP | None (all-rights-reserved) | ~265 | **No** (model code unreleased) | Dataset/metrics only; not runnable |
| **CADFusion** ([2501.19054](https://arxiv.org/abs/2501.19054)) | LLaMA-3-8B + DPO visual feedback | CAD seq | **MIT** | ~81 | **Yes** (HF) | Maintained (Microsoft); usable |
| **CAD-Recode** ([2412.14042](https://arxiv.org/abs/2412.14042)) | Qwen2-1.5B + point projector | **CadQuery** | CC BY-NC 4.0 | ~244 | Yes (HF) | Active to late-2025; non-commercial |
| **cadrille** ([2505.22914](https://arxiv.org/abs/2505.22914)) | Qwen2-VL-2B + SFT→RL | **CadQuery** | Apache-2.0 | ~153 | Yes (SFT+RL on HF) | Inference released; **RL training code withheld** |
| **Text-to-CadQuery** ([2505.06507](https://arxiv.org/abs/2505.06507)) | 6 LLMs fine-tuned | **CadQuery** | Unspecified | ~104 | Yes (6 sizes) | Reproducible baseline; quiet since 2025 |
| **CAD-Coder (MIT)** ([2505.14646](https://arxiv.org/abs/2505.14646)) | LLaVA-1.5 / Vicuna-13B (image→) | **CadQuery** | **Apache-2.0** | ~180 | Yes (HF) | Active; image-to-CAD |
| **CAD-Coder (Beihang)** ([2505.19713](https://arxiv.org/abs/2505.19713)) | Qwen2.5-7B SFT+GRPO+CoT (text→) | **CadQuery** | **Apache-2.0** | ~7 | Yes (HF) | NeurIPS 2025; vLLM batch |
| **MEDA** (this repo) | ADK multi-agent loop + verify | **CadQuery** | Apache-2.0 | — | n/a (training-free) | Active; the system under study |

**Datasets:** **ABC** (~1M B-Rep STEP, CVPR 2019; Onshape ToU license),
**DeepCAD** (178,238 seq, MIT code), **Fusion360 Gallery** (8,625 reconstruction
seqs + segmentation/assembly; non-commercial), **Text2CAD** (~170k models / ~660k
prompts; non-commercial), **Omni-CAD** (~450k multimodal), **CAD-Recode** (~1M
procedural CadQuery; CC BY-NC), **GenCAD-Code** (163k image–CadQuery), **ExeCAD /
CADExpert** (16–17k executable-CadQuery triplets).

### 5C. Open-source agentic peers (MEDA's closest open-source neighbours)

These are the other *open* generate→execute→render→critique→repair loops — MEDA's
direct family in open source:

- **CADSmith** ([2603.26512](https://arxiv.org/abs/2603.26512), CMU; ~21★, active
  through Jun 2026 — the **engineering-strongest** agentic repo): 5-agent loop
  Planner → Coder (**RAG over CadQuery API docs**) → sandboxed Executor →
  Validator (independent **Claude-Opus VLM judge + exact OpenCASCADE kernel
  measurements**) → Refiner. Outputs **CadQuery → STEP/STL**. Reports mean Chamfer
  28.37 → 0.74 vs zero-shot. **This is the most important peer to study** — it
  already implements two of MEDA's top roadmap items (API RAG + kernel-measured
  validation).
- **Query2CAD** ([2406.00144](https://arxiv.org/abs/2406.00144), CMU; ~53★, the
  early baseline): LLM emits **FreeCAD Python macros**, critiqued by BLIP2 +
  optional human feedback; GPT-4-Turbo 53.6% → ~76.7% success.
- **CAD-Assistant** ([2412.13810](https://arxiv.org/abs/2412.13810)) and
  **build123d-mcp** (pzfreo, ~25★, Apache-2.0; drove GPT-5.5 to top CADGenBench)
  are the FreeCAD- and build123d-side agentic loops.

### 5D. MCP / in-app copilot ecosystem

The fastest-growing *open* surface is LLM-agents-drive-the-CAD-app via MCP:

- **blender-mcp** (~23.3k★, MIT) — by far the most-starred tool in the whole
  space, but **mesh-only** (Blender `bpy`); not CAD. A cautionary data point on
  where mindshare actually is.
- **FreeCAD MCP** (all ✅ parametric FCStd): **neka-nat/freecad-mcp** (~1.2k★, the
  reference, incl. CalculiX FEM), **ghbalf/freecad-ai** (~348★, in-app workbench,
  ~20 providers), **spkane robust-mcp-server** (~131★, 150+ tools, Dockerized).
- **CadQuery/build123d MCP**: build123d-mcp (above), mcp-cadquery (~16★).
- **OpenSCAD wrappers** (⚠️ parametric *source* but compiles to **mesh**, not
  B-Rep): openscad-agent (~96★), TalkCAD, text-2-cad. **CadQuery wrappers**
  (✅ B-Rep): CQAsk (~185★), C3D.

**Open-source takeaways for MEDA:**
1. The open weights are now **CadQuery-emitting** (cadrille, CAD-Recode,
   CAD-Coder, Text-to-CadQuery) — *exactly MEDA's representation*. MEDA could swap
   its modeler LLM for one of these fine-tuned open models and keep its
   verification loop on top.
2. Many strong datasets are **non-commercial** (Text2CAD, Fusion360, CAD-Recode) —
   relevant if MEDA is ever productized.
3. MEDA is one of very few **fully-open agentic** systems; most agentic work
   (CADCodeVerify, Seek-CAD, CADDesigner) is paper-only or partially released.

---

## 6. Academic / industry research (verified, through June 2026)

### Family A — Fine-tuned domain models

| Paper | Venue / Date | Base model | Headline result |
|---|---|---|---|
| DeepCAD ([2105.09492](https://arxiv.org/abs/2105.09492)) | ICCV 2021 | Transformer AE | Foundational dataset (178k) + baseline |
| Text2CAD ([2409.17106](https://arxiv.org/abs/2409.17106)) | **NeurIPS 2024 Spotlight** | BERT + Transformer dec | Median CD 0.37×10⁻³, Invalid 0.93% |
| CAD-MLLM ([2411.04954](https://arxiv.org/abs/2411.04954)) | arXiv Nov 2024 | Vicuna-7B+LoRA (multimodal) | CD 1.85 vs DeepCAD 4.51; F-score 90.88 |
| CADFusion ([2501.19054](https://arxiv.org/abs/2501.19054)) | **ICML 2025** | LLaMA-3-8B + DPO | CD 19.89 vs Text2CAD 30.23; 8.5× less data |
| CAD-Recode ([2412.14042](https://arxiv.org/abs/2412.14042)) | **ICCV 2025** | Qwen2-1.5B (point→code) | DeepCAD CD 0.30, IoU 92% (point-cloud SOTA) |
| Text-to-CadQuery ([2505.06507](https://arxiv.org/abs/2505.06507)) | arXiv May 2025 | 6 LLMs | Exact-match 69.3%, CD −48.6% |
| CAD-Coder/MIT ([2505.14646](https://arxiv.org/abs/2505.14646)) | IDETC 2025 | LLaVA-1.5 (image→code) | 100% valid syntax; IoU_best 0.675 |
| CAD-Coder/Beihang ([2505.19713](https://arxiv.org/abs/2505.19713)) | **NeurIPS 2025** | Qwen2.5-7B SFT+GRPO+CoT | Mean CD 6.54×10³ (−77.7% vs Text2CAD) |

*Adjacent:* CAD-Llama (2505.04481), CADmium (2507.09792), NURBGen (2511.06194),
GeoCAD (2506.10337), GACO-CAD (2510.17157).

### Family B — RL post-training (**the current benchmark frontier**)

| Paper | Venue / Date | RL recipe | Base | Headline |
|---|---|---|---|---|
| **cadrille** ([2505.22914](https://arxiv.org/abs/2505.22914)) | ICLR 2026 *(author-claimed; see flag)* | GRPO online, reward = IoU + invalid penalty | Qwen2-VL-2B | DeepCAD IoU 87.1→**90.2**, invalid 2.1→**0.0%** |
| **ReCAD** ([2512.06328](https://arxiv.org/abs/2512.06328)) | **AAAI 2026 Oral** | GRPO + DINOv2 visual reward | Qwen2.5-VL-7B | Image→CAD CD 73.47→**29.61** |
| **CME-CAD** ([2512.23333](https://arxiv.org/abs/2512.23333)) | arXiv Dec 2025 | Multi-Expert RL + GRPO | Qwen3-4B | IoU **80.71%**, exec 98.25% |
| **From Intent to Execution** ([2508.10118](https://arxiv.org/abs/2508.10118)) | arXiv Aug 2025 | CoT cold-start + goal-driven RL | Qwen2.5-VL | IoU 78.7%, exec 98.83% |
| **TOOLCAD** ([2604.07960](https://arxiv.org/abs/2604.07960)) | arXiv Apr 2026 | Online curriculum RL, tool-using agent | open LLM | Beats agentic baselines (held-out) |
| **CAD-Coder/Beihang** ([2505.19713](https://arxiv.org/abs/2505.19713)) | NeurIPS 2025 | GRPO + Chamfer reward | Qwen2.5-7B | (see Family A) |

**Dominant recipe:** GRPO + execution/geometric reward (IoU + Chamfer + invalid/
format penalty), Qwen-VL backbone, sometimes a visual (DINOv2) or LLM-judge
reward. This is the family that has overtaken training-free agentic loops on the
leaderboards.

### Family C — Training-free agentic loops with visual/execution feedback — **MEDA's family**

| System | Venue / Date | Feedback | Base model | Key benchmark result |
|---|---|---|---|---|
| **CADCodeVerify** ([2410.05340](https://arxiv.org/abs/2410.05340)) | ICLR 2025 | VLM Q&A visual + compile; ≤2 iters | GPT-4 | CADPrompt: PC-dist 0.127, compile 96.5% |
| **Seek-CAD** ([2505.17702](https://arxiv.org/abs/2505.17702)) | ICLR 2026 | Step-wise render critique + CoT; local | DeepSeek-R1:32B + Gemini-2.0 | Chamfer 0.198, IoGT 0.723 |
| **CADDesigner** ([2508.01031](https://arxiv.org/abs/2508.01031)) | preprint Aug 2025 | Render inspection, ReAct, knowledge base | Claude-4-Sonnet + Gemini-2.5 | Text2CAD IoU 0.277, 100% success |
| **EvoCAD** ([2510.11631](https://arxiv.org/abs/2510.11631)) | IEEE ICTAI 2025 | **Evolutionary** VLM-guided mutation/selection | GPT-4V + GPT-4o | Best on topological (Euler) metrics |
| **CAD-Assistant** ([2412.13810](https://arxiv.org/abs/2412.13810)) | ICCV 2025 | FreeCAD exec + visual; tool-augmented | GPT-4o | 2D-QA 79.1%, autoconstrain CF1 0.484 |
| **CADSmith** ([2603.26512](https://arxiv.org/abs/2603.26512)) | CMU, preprint Mar 2026 | 5-agent + RAG + Claude-Opus judge + OCCT measure | Claude-Opus | Mean Chamfer 28.37→0.74 vs zero-shot |
| **Query2CAD** ([2406.00144](https://arxiv.org/abs/2406.00144)) | CMU, 2024 | FreeCAD macros + BLIP2 self-refine | GPT-4-Turbo | Success 53.6%→76.7% |
| **MEDA** (this repo) | **ASME IDETC-CIE 2025** (DOI 10.1115/DETC2025-163946) | Multi-agent compile/exec + CAD-image VLM | GPT-4o (paper); Gemini (code) | **CADPrompt: PC-dist 0.0555, compile 99%** |

**MEDA leads its own family on point-cloud distance and compile rate** (vs its
direct baseline CADCodeVerify), and adds two things most peers lack: a **surgical
feature timeline** (edit one step, not the whole script) and **hard B-Rep
topology gating** in the reward. Its IoGT (0.9413) is fractionally below
CADCodeVerify's (0.944). Note MEDA is a **conference paper, not on arXiv** — any
arXiv ID for MEDA would be fabricated.

### Family D — Benchmarks, verifiers, datasets, grounding

- **CADPrompt** (introduced by CADCodeVerify, 2410.05340): 200 NL prompts +
  expert CadQuery. **This is MEDA's evaluation benchmark.**
- **Text2CAD-Bench** ([2605.18430](https://arxiv.org/abs/2605.18430), preprint
  May 2026): 600 human-curated prompts, L1–L4; finds LLMs degrade sharply on
  **complex topology + advanced features** — the field's stated frontier.
- **CAD-Judge** ([2508.04002](https://arxiv.org/abs/2508.04002)): compiler-as-a-
  judge reward + compiler-as-a-review verifier; cheap alternative to VLM judging.
- **B-Rep primitive grounding / FutureCAD**
  ([2603.11831](https://arxiv.org/abs/2603.11831), preprint Mar 2026): LLM emits a
  natural-language geometric query, a BERT+UV-Net module (**BRepGround**) resolves
  it to specific faces/edges — the missing piece for fillet/chamfer/shell.
- **Survey** ([2505.08137](https://arxiv.org/abs/2505.08137)): "Large Language
  Models for Computer-Aided Design: A Survey" (Zhang et al.).
- **Key metrics:** Chamfer Distance (avg, dominant; beware ×10³ vs ×10⁻³ scaling),
  Hausdorff (worst-case), IoU/IoGT, valid/compile rate (Invalid Ratio), primitive
  F1 (Hungarian-matched), and topology metrics (SegE/DangEL/SIR/FluxEE).
  **MEDA already uses Chamfer/Hausdorff/IoGT + compile rate** — methodologically
  aligned with the field.

### Current SOTA leaders (mid-2026)
- **Point-cloud → CAD:** CAD-Recode (DeepCAD CD 0.30 / IoU 92%); cadrille after RL.
- **Text → CAD (learned):** CAD-Coder/Beihang (CD 6.54×10³) and CADFusion lead
  the fine-tuned/RL text track; ReCAD leads image→CAD.
- **Multimodal + RL, broadest:** cadrille (SOTA across DeepCAD/Fusion360/CC3D).
- **Training-free agentic:** MEDA and CADDesigner are the strongest of this
  family on their respective benchmarks, but the family as a whole trails the
  RL-fine-tuned models on the shared benchmarks.

---

## 7. Where MEDA sits — honest assessment

**Modern and well-engineered, but one generation behind the benchmark frontier.**

### Genuinely strong / current
- **Parametric feature-timeline canvas with surgical edits** (`core/canvas.py`):
  add/modify/remove a *single* step with named parameters — more advanced than
  most agentic baselines that regenerate whole scripts.
- **Hard B-Rep topology gating** as reward (`core/reward_engine.py`): closer to
  CAD-Judge / B-Rep-grounding rigor than pure visual critique.
- **Multi-view orthographic critique** (iso/top/front/right collage): aligns with
  the field-wide "automated visual feedback wins" finding.
- **Google ADK multi-agent handoffs** (orchestrator → modeler → critic): a current
  orchestration style.
- **Distance-mode reward** (Chamfer/Hausdorff on normalized point clouds): matches
  the eval-harness methodology used across the literature.
- **Fully open + verifiable**: exposes executable CadQuery + B-Rep metrics that
  Zoo/Adam/SGS-1 largely do not.

### Where the frontier has moved past MEDA
1. **Nothing is learned.** RL-fine-tuned models (cadrille, ReCAD, CAD-Coder,
   CME-CAD) now top the benchmarks; MEDA's reward only steers an in-context loop.
2. **Brittle exact-match constraints** (partly addressed): inferring the *exact*
   `num_faces`/`num_edges` of an unbuilt model and demanding integer equality is
   unreliable — a fillet shifts counts. The graded `geom_score` +
   `topology_tolerance` help, but the inferred-target approach is still weak.
3. **No retrieval/grounding of the CadQuery API** — newer systems ground on B-Rep
   primitives or API docs to cut hallucinated methods.
4. **Sandbox isn't truly isolated** (`core/sandbox.py` is a `subprocess`, not a
   container).
5. **Single-attempt loop** — no parallel candidates or judge panel (cf. EvoCAD's
   evolutionary search, the judge-panel pattern).
6. **No advanced-feature grounding** — fillet/chamfer/shell via face/edge
   selection (the Text2CAD-Bench / BRepGround frontier) is unaddressed.
7. **Default model drift** — code defaults to `gemini-3.5-flash`; the paper's
   results are GPT-4o. Benchmarks and README should state the model explicitly.

### How MEDA compares head-to-head

| Dimension | MEDA | RL models (cadrille/ReCAD) | Agentic peers (Seek-CAD/CADDesigner) | Startups (Zoo/Adam/SGS-1) |
|---|---|---|---|---|
| Editable parametric output | ✅ CadQuery + STEP | ✅ CadQuery | ✅ code | ✅ KCL/SCAD/STEP |
| Learns from reward | ❌ | ✅ GRPO | ❌ | ✅ (trained) |
| Closed verification loop | ✅ compile+topo+visual | partial (reward only) | ✅ visual | ❌/limited |
| Surgical feature edits | ✅ | ❌ | ❌ | partial |
| Advanced features (fillet/chamfer grounding) | ❌ | partial | ❌ | partial |
| Open & transparent | ✅ | mostly ✅ | mostly ❌ | ❌ |
| Distribution / UX | research | weights | research | ✅ plugins/app |

---

## 8. How MEDA can improve — prioritized roadmap

Ordered by leverage (impact ÷ effort):

1. **RAG-ground the CadQuery API into the modeler agent.** Index CadQuery/
   build123d method signatures + worked examples; retrieve into the modeler's
   context. Directly attacks hallucinated methods — the cheapest reliability win.
   *(Aligns with the architecture-vision "retrieval-augmented CAD memory."* The
   open-source **CADSmith** already does exactly this — RAG over CadQuery docs +
   an independent VLM judge + exact OCCT kernel measurements — and is the best
   reference implementation to study/borrow from.)

2. **Rejection-sampling / RL fine-tune the modeler on MEDA's own reward.** MEDA
   already computes a graded geometric reward — feed it. Start with best-of-N
   rejection-sampling SFT on an open CadQuery model (cadrille / CAD-Coder /
   Text-to-CadQuery weights), then GRPO. This is the **cadrille/CAD-Coder leap**
   from "steer in-context" to "learn." Highest ceiling.

3. **Advanced-feature grounding (fillet/chamfer/shell).** Add a face/edge
   selection resolver: LLM emits a natural-language geometric query → resolve to
   B-Rep entities (à la BRepGround). This is the explicit field frontier per
   Text2CAD-Bench; it's where every system currently fails.

4. **Replace brittle exact-count targets with distance-first gating.** Lean on
   the existing distance mode + graded `geom_score`; treat inferred integer
   counts as soft hints, never hard gates, in pure-text runs.

5. **Containerize the sandbox.** Move `subprocess` execution into a resource-
   limited container (or `nsjail`/`firejail`) — needed for safety the moment MEDA
   runs untrusted prompts or is productized.

6. **Parallel candidates + judge panel / evolutionary search.** Generate N feature
   timelines with different decomposition strategies, rank by compile + topology +
   visual + simplicity, refine the best (EvoCAD-style). Trades tokens for
   reliability on hard prompts.

7. **Benchmark rigor.** Run MEDA on **CADPrompt** *and* **Text2CAD-Bench**
   (L1–L4) and the standard **DeepCAD/Fusion360** splits; report Chamfer/Hausdorff/
   IoGT/compile **per model and provider**, with token cost and repair-turn counts.
   Promote `eval_metrics/` into an automated regression suite
   (per `architecture_vision.md` §6).

8. **Adopt CAD-Judge-style compiler-as-verifier** to cut latency/cost of the
   visual-critic gate, keeping the VLM critic as an explanation layer.

9. **Productization parity.** The startups' real moat is **distribution** (Adam's
   Onshape/Fusion plugins, Zoo's Design Studio). If MEDA ever ships, a
   FreeCAD-MCP or Onshape integration + an editable parameter table would close
   the UX gap while keeping the open, verifiable core.

10. **Refresh README/docs** to match the current ADK/Gemini architecture and the
    corrected citations below.

---

## 9. Citation corrections to the prior doc

While verifying every arXiv ID, these errors in
[`sota_comparison.md`](./sota_comparison.md) surfaced (all IDs there *resolve*,
but some are mis-attributed):

- ❌ **"Text2CAD — arXiv:2505.19490"** is **wrong**. `2505.19490` is *"Automated
  CAD Modeling Sequence Generation from Text Descriptions via Transformer-Based
  LLMs"* (Liao et al., ACL 2025) — a different paper. **Real Text2CAD =
  [2409.17106](https://arxiv.org/abs/2409.17106)** (Khan et al., NeurIPS 2024
  Spotlight).
- ⚠️ **CADFusion** had no ID in the prior doc; it is
  [2501.19054](https://arxiv.org/abs/2501.19054) (Microsoft, ICML 2025), title
  *"…Through Infusing Visual Feedback…"*.
- ⚠️ **CAD-Coder** is **two distinct papers**:
  [2505.14646](https://arxiv.org/abs/2505.14646) (MIT, image→CadQuery, SFT) and
  [2505.19713](https://arxiv.org/abs/2505.19713) (Beihang, text→CadQuery,
  SFT+GRPO+CoT, NeurIPS 2025). The prior doc's description matches the Beihang one.
- ⚠️ **CADCodeVerify** (MEDA's baseline) had no ID; it is
  [2410.05340](https://arxiv.org/abs/2410.05340) (Alrashedy et al., ICLR 2025),
  and it introduced the **CADPrompt** benchmark MEDA evaluates on.
- ⚠️ **cadrille's "ICLR 2026" acceptance is author-claimed** (repo + arXiv say so),
  but the OpenReview forum located reads *desk-rejected* and the ICLR virtual page
  404'd. The paper is unquestionably real; treat the acceptance as
  author-claimed-but-unverified.
- ✅ The remaining prior-doc IDs (ReCAD, TOOLCAD, CME-CAD, From-Intent, Seek-CAD,
  CADDesigner, EvoCAD, CAD-Judge, Text2CAD-Bench, B-Rep grounding, survey) all
  resolve and are correctly attributed. The `2603.*`/`2604.*`/`2605.*` IDs are
  March–May 2026 preprints — in the past relative to today (2026-06-29), not
  fabricated future dates.

---

## 10. Sources

**Research (verified arXiv):** DeepCAD [2105.09492](https://arxiv.org/abs/2105.09492) ·
SkexGen [2207.04632](https://arxiv.org/abs/2207.04632) ·
Fusion360 Gallery [2010.02392](https://arxiv.org/abs/2010.02392) ·
ABC [1812.06216](https://arxiv.org/abs/1812.06216) ·
Text2CAD [2409.17106](https://arxiv.org/abs/2409.17106) ·
CAD-MLLM [2411.04954](https://arxiv.org/abs/2411.04954) ·
CADFusion [2501.19054](https://arxiv.org/abs/2501.19054) ·
CAD-Recode [2412.14042](https://arxiv.org/abs/2412.14042) ·
Text-to-CadQuery [2505.06507](https://arxiv.org/abs/2505.06507) ·
CAD-Coder/MIT [2505.14646](https://arxiv.org/abs/2505.14646) ·
CAD-Coder/Beihang [2505.19713](https://arxiv.org/abs/2505.19713) ·
cadrille [2505.22914](https://arxiv.org/abs/2505.22914) ·
ReCAD [2512.06328](https://arxiv.org/abs/2512.06328) ·
CME-CAD [2512.23333](https://arxiv.org/abs/2512.23333) ·
From Intent to Execution [2508.10118](https://arxiv.org/abs/2508.10118) ·
TOOLCAD [2604.07960](https://arxiv.org/abs/2604.07960) ·
CADCodeVerify [2410.05340](https://arxiv.org/abs/2410.05340) ·
Seek-CAD [2505.17702](https://arxiv.org/abs/2505.17702) ·
CADDesigner [2508.01031](https://arxiv.org/abs/2508.01031) ·
EvoCAD [2510.11631](https://arxiv.org/abs/2510.11631) ·
CAD-Assistant [2412.13810](https://arxiv.org/abs/2412.13810) ·
CAD-Judge [2508.04002](https://arxiv.org/abs/2508.04002) ·
B-Rep grounding [2603.11831](https://arxiv.org/abs/2603.11831) ·
Text2CAD-Bench [2605.18430](https://arxiv.org/abs/2605.18430) ·
Survey [2505.08137](https://arxiv.org/abs/2505.08137) ·
MEDA (ASME IDETC-CIE 2025, DOI 10.1115/DETC2025-163946 — not on arXiv).

**Commercial / startup:** Zoo [zoo.dev](https://zoo.dev/research/introducing-text-to-cad) ·
Adam [adam.new](https://adam.new/) ·
[TechCrunch — Adam $4.1M](https://techcrunch.com/2025/10/31/yc-alum-adam-raises-4-1m-to-turn-viral-text-to-3d-tool-into-ai-copilot/) ·
CADAM (OSS) [github.com/Adam-CAD/CADAM](https://github.com/Adam-CAD/CADAM) ·
Spectral Labs SGS-1 [spectrallabs.ai/research/SGS-1](https://www.spectrallabs.ai/research/SGS-1) ·
Aurorin CAD (YC W26) [ycombinator.com/launches](https://www.ycombinator.com/launches/PWy-aurorin-cad-w26-the-next-generation-mechanical-cad-software) ·
Leo AI [getleo.ai](https://www.getleo.ai/) · [Leo $9.7M](https://www.getleo.ai/blog/leo-ai-raises-9-7m-to-build-the-world-s-first-ai-for-mechanical-engineering) ·
CADScribe · CADGPT · Vondy · OpenArt (Xometry-tested) ·
Autodesk Project Bernini / Neural CAD [research.autodesk.com](https://www.research.autodesk.com/projects/project-bernini/) ·
PTC Onshape AI Advisor · Siemens NX Copilot · Dassault AURA/LEO.

**Open-source agentic/MCP:** CADSmith [github.com/jabarkle/CADSmith](https://github.com/jabarkle/CADSmith) ·
Query2CAD [2406.00144](https://arxiv.org/abs/2406.00144) ·
build123d-mcp · neka-nat/freecad-mcp · blender-mcp (mesh).

**Mesh tools:** Meshy [meshy.ai](https://www.meshy.ai/) ·
Tripo [tripo3d.ai](https://www.tripo3d.ai/) ·
Rodin [hyper3d.ai](https://hyper3d.ai/) ·
Luma [lumalabs.ai](https://lumalabs.ai/) ·
Spline [spline.design](https://spline.design/ai-generate) ·
Sloyd [sloyd.ai](https://www.sloyd.ai/).

**Hands-on tests:** Xometry [text-to-cad-tools-test](https://xometry.pro/en/articles/text-to-cad-tools-test/) ·
Leo AI [text-to-cad-tools-comparison-guide](https://www.getleo.ai/blog/text-to-cad-tools-comparison-guide).
</content>
</invoke>
