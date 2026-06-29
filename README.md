# MEDA
A multi-agent system for parametric CAD model creation

> [!NOTE]
> This version of MEDA has been redesigned to utilize **`uv`** for dependency/workspace management and **Google ADK** for multi-agent orchestration, building on the original implementation and research presented in our ASME IDETC 2025 paper (original repository: [AnK-Accelerated-Komputing/MEDA](https://github.com/AnK-Accelerated-Komputing/MEDA)).

![MEDA Overview](docs/images/MEDA_Overview.png)


## Introduction
MEDA is a Google ADK-based multi-agent system for autonomous parametric CAD generation. The current runtime centers on `ReasoningCADCore`, which coordinates a `MEDAOrchestrator`, `CADModelerAgent`, and `VisualCriticAgent`: the modeler edits a shared CadQuery feature timeline, while the critic compiles generated geometry in a hardened sandbox, extracts B-Rep metrics, renders orthographic views, and feeds reward/visual feedback back into the loop until success or configured iteration limits.

### Current CAD generation algorithm

MEDA now has three generation modes that share the same sandbox and artifact contract:

1. **Default ADK feature-timeline loop** — the orchestrator delegates feature edits to the modeler and verification to the critic. The modeler uses tools for parameters, feature insertion/modification, CadQuery API lookup, and B-Rep selector inspection. The critic calls the sandbox, computes reward from compile success plus configured constraints, renders successful iterations, and runs multimodal visual critique once geometric reward reaches 1.0.
2. **Fast Mode** — a complete CadQuery script is generated in one shot and retried up to three times using compile errors as feedback. This is faster than the full agent loop but does not perform the same iterative VLM critique.
3. **Multi-candidate search** — when the candidate slider is greater than one, MEDA generates up to five complete scripts using diverse strategy hints, executes them serially in the sandbox for isolation, ranks them by compile success, geometry score, positive-volume validity, and simplicity, then promotes the winning script and artifacts. The visual-match hook is supported in the scorer for future pipelines, but the current candidate branch is a geometry-first selector rather than a full visual-critique loop.

Common preprocessing resets or keeps the canvas, retrieves CadQuery API references and past lessons/skills, infers only tolerance-checked constraints such as volume and center of mass, and merges those with explicit user constraints. The sandbox deletes stale artifacts before every run, validates generated code with an AST allow-list, scrubs secret-bearing environment variables, applies POSIX resource limits when available, and emits metrics including volume, area, B-Rep counts, center of mass, solid count, and validity.

## Demo
![MEDA Demo](docs/images/MEDA_demo.gif)



## Performance
We compare our multi-agent architecture MEDA with recent state-of-the-art CAD generation framework CADCodeVerify across three CAD evaluation metrics and compilation rate as shown in the table below.

**Historical paper benchmark (GPT-4o-era MEDA, not a benchmark of the current default Gemini/ADK runtime): Comparison of MEDA against CadCodeVerify**

| **Framework**     | **MLLM used** | **Point Cloud dist. ↓** | **Hausdorff dist. ↓** | **IoGT ↑**           | **Compile Rate ↑** |
|-------------------|---------------|--------------------------|------------------------|----------------------|---------------------|
| **MEDA**          | GPT-4o        | **0.0555** (0.095)       | **0.2628** (0.401)     | 0.9413 (0.0275)      | **99%**             |
| CADCodeVerify     | GPT-4         | 0.127 (0.135)            | 0.419 (0.356)          | **0.944** (0.028)    | 96.5%               |


## Why MEDA is different

MEDA uses an executable multi-agent loop instead of one-shot text-to-code generation. A lead orchestrator delegates parametric feature construction to a modeler agent and verification to a critic agent. The critic compiles CadQuery code in a sandbox, extracts B-Rep topology metrics, renders orthographic views, and feeds failures back into the loop. This makes the generated CAD easier to repair, measure, and reuse.

### Reliability features (June 2026)

Building on the competitive analysis in [`docs/text_to_cad_landscape_2026.md`](docs/text_to_cad_landscape_2026.md), MEDA now borrows the strongest ideas from its open-source and commercial peers (see [`docs/agent_improvements_2026.md`](docs/agent_improvements_2026.md) for the full mapping):

- **CadQuery API grounding (RAG)** — a curated, kernel-verified API reference is retrieved into the modeler's context and exposed as a `lookup_cadquery_api` tool, so the model uses real methods instead of hallucinating them *(from CADSmith)*.
- **B-Rep selector grounding** — an `inspect_current_model` tool reports the model's real faces/edges (orientations, lengths, selector hints, safe fillet/chamfer ceiling) so advanced features land on geometry that actually exists *(from BRepGround / CADSmith)*.
- **Multi-candidate search** — generate up to 5 candidates with different modeling strategies and automatically keep the best by compile + geometry + simplicity score *(from EvoCAD)*.
- **Hardened sandbox** — an AST allow-list, POSIX resource limits, and secret scrubbing close the untrusted-code execution surface, plus cheap CAD-Judge-style validity signals on every build.

## Research and product roadmap

The next architecture direction is documented in [`docs/architecture_vision.md`](docs/architecture_vision.md), and the self-improving agent strategy is documented in [`docs/self_improving_meda_strategy.md`](docs/self_improving_meda_strategy.md). Key priorities include a structured CAD intermediate representation, constraint solving before code generation, retrieval over successful CAD traces, multi-candidate search, learned visual reward models, procedural skill memory, and benchmark-driven geometry regression.

For a full competitive map of Text-to-CAD through June 2026 — open source, startups/proprietary products, and academic/industry research — plus where MEDA sits and a prioritized improvement roadmap, see [`docs/text_to_cad_landscape_2026.md`](docs/text_to_cad_landscape_2026.md).

## Setup Instructions

### 1. **Clone the Repository**
   ```bash
   git clone https://github.com/srtux/MEDA.git
   cd MEDA
   ```
### 2. **Environment Setup**

We use **`uv`** for lightweight Python virtual environment and dependency management.

1.  **Install `uv`** (if not already installed):
    ```bash
    curl -LsSf https://astral.sh/uv/install.sh | sh
    ```

2.  **Sync Dependencies**:
    Initialize the virtual environment and install all packages declared in `pyproject.toml` automatically:
    ```bash
    uv sync
    ```

3.  **Activate Virtual Environment**:
    ```bash
    source .venv/bin/activate
    ```

### 3. API Configuration

- Create a `.env` file in the root directory with the following content:
   
   ```
   GEMINI_API_KEY='your_gemini_api_key'
   GROQ_API_KEY='your_groq_api_key'
   AZURE_API_KEY='your_azure_api_key'
   AZURE_OPENAI_BASE='your_azure_openai_base_url'
   ```

- Alternatively, you can export them directly in your shell:

   ```bash
   export GEMINI_API_KEY='your_gemini_api_key'
   export GROQ_API_KEY='your_groq_api_key'
   export AZURE_API_KEY='your_azure_api_key'
   export AZURE_OPENAI_BASE='your_azure_openai_base_url'
   ```


### 4. **Run the Application**
- For command line application:
   ```bash
   python main.py
   ```
- For streamlit app:
   ```bash
   streamlit run streamlitapp.py
   ```
You can insert the api key for different models with streamlit app.
Then, follow the on-screen instructions to interact with MEDA.

## Paper
Our paper is accepted to IDETC 2025. More details to follow.

## Citation
If you use this in your work, please consider citing the following publications.
```
@inproceedings{MEDA2025,
  author    = {Nirmal Prasad Panta and Saugat Kafley and Rujal Acharya and Sashank Parajuli and Dikshya Parajuli and Prince Panta and Saroj Belbase and Sudikshya Pant and Amit Regmi and Akio Tanaka and Christopher McComb},
  title     = {{MEDA: A Multi-Agent System for Parametric CAD Model Creation}},
  booktitle = {Proceedings of the ASME 2025 International Design Engineering Technical Conferences
               and Computers and Information in Engineering Conference (IDETC/CIE 2025)},
  year      = {2025},
  address   = {Anaheim, CA, USA},
  paperid   = {IDETC2025-163946},
  publisher = {American Society of Mechanical Engineers (ASME)},
  doi       = {10.1115/DETC2025-163946}
  url       = {https://doi.org/10.1115/DETC2025-163946}
}
```
