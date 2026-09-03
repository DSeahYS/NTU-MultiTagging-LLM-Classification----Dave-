# 🏷️ NTU Multi-Tagging LLM Classification & Reranking Engine

> **High-Throughput, Multi-Label Document Classification and Semantic Reranking Pipeline for 3D Bioprinting & Additive Manufacturing Literature.**  
> Powered by an asynchronous multi-model LLM fan-out architecture mapping peer-reviewed research across a rigorous 30-category domain taxonomy.

---

## 👨‍🔬 Research Collaboration & Attribution

- **Lead Researcher & Developer**: **Dave Seah Yong Sheng** — Visiting Researcher, Nanyang Technological University (NTU), Singapore
- **Research Collaborator**: **Dr. Xi Huang (Huang Xi)** — Research Fellow, Singapore Centre for 3D Printing (SC3DP), Nanyang Technological University (NTU)

### 📄 Scientific Foundation
This classification framework forms Phase 1 of the knowledge distillation pipeline engineered to resolve the critical challenges identified in the **"Future Work"** section of the foundation paper:
> **Xi Huang, Hanqi Su, Zhengjie Cui, Jia Min Lee, Xinchao Gao, et al.**  
> *"BioPrint-LKM: An evidence-grounded large knowledge model for bioprinting knowledge retrieval and parameter initialization"*, **International Journal of Bioprinting**, 2026. DOI: [10.36922/ijb.026110094](https://doi.org/10.36922/ijb.026110094).

---

## 🎯 The Core Problem: Why Structured Document Tagging Matters

Large-scale literature retrieval in specialized scientific domains fails when documents are treated as undifferentiated text blocks. In additive manufacturing and regenerative medicine, a single paper typically intersects multiple engineering and biological domains (e.g., *a pneumatic extrusion study using GelMA/nanocellulose bioinks containing human dermal fibroblasts for pre-vascularized skin tissue engineering calibrated via Bayesian optimization*).

Without fine-grained multi-label categorization:
1. **Retrieval Noise**: Flat vector RAG retrieves papers with superficial keyword matches rather than methodologically compatible techniques.
2. **Knowledge Graph Construction Bottlenecks**: OpenSPG and KAG engines require clean category priors to instantiate entity-class edges (`TAGGED_WITH`, `CO_OCCURS_WITH`).
3. **Manual Curation Infeasibility**: Screening thousands of papers across dozens of criteria requires hundreds of human-hours.

**The Multi-Tagging LLM Classifier automates this completely.** It extracts abstracts, introductions, and section markers from technical PDFs and performs zero-shot multi-label classification into **30 official schema categories** across 5 distinct domains.

---

## 🏛️ The 30-Category Bioprinting Taxonomy

The engine classifies literature across 5 orthogonal engineering axes:

| Domain | Tag Identifier | Description & Representative Sub-Technologies |
| :--- | :--- | :--- |
| **1. Bioprinting Modality** | `extrusion_based` | Pneumatic, piston, or screw-driven filament deposition |
| | `droplet_based_inkjet` | Piezoelectric, thermal, or micro-valve droplet jetting |
| | `light_based_vat` | DLP, SLA, Stereolithography, Two-photon polymerization |
| | `laser_assisted` | Laser-induced forward transfer (LIFT) |
| | `in_situ_bioprinting` | Direct deposition onto living tissue defects during surgery |
| | `4d_bioprinting` | Stimuli-responsive smart materials (shape-morphing post-print) |
| **2. Biomaterial Chemistry** | `natural_polymers` | GelMA, Alginate, Collagen, Hyaluronic acid, Chitosan, Fibrin |
| | `synthetic_polymers` | Pluronic F127, PCL, PEG, PLA, polyurethane |
| | `dmatrix_bioinks` | Decellularized extracellular matrix (dECM) tissue lysates |
| | `nanomaterial_composite` | Graphene oxide, carbon nanotubes, nanocellulose, MXene |
| | `hydrogel_rheology` | Shear-thinning, yield stress, storage modulus ($G'$), thixotropy |
| | `sacrificial_support` | Temporary baths (FRESH) or fugitive inks for perfusable channels |
| **3. Biological Systems** | `stem_cells` | iPSCs, MSCs, embryonic stem cells, neural precursors |
| | `primary_cells` | HUVECs, dermal fibroblasts, hepatocytes, cardiomyocytes |
| | `multicellular_co_culture`| Controlled spatial patterning of $\ge 2$ distinct cell lines |
| | `spheroids_organoids` | High-density cellular aggregates, organ-on-a-chip units |
| | `cell_viability_proliferation` | Post-print biological viability assays, LIVE/DEAD, metabolic tests |
| | `vascularization` | Capillary formation, lumen sprouting, perfusable vascular trees |
| **4. Engineering Applications** | `tissue_engineering` | Functional scaffolds for organ repair and regenerative medicine |
| | `disease_modeling` | *In vitro* pathological replication, tumor microenvironments |
| | `drug_screening_discovery` | High-throughput toxicological and pharmaceutical testing assays |
| | `anatomical_models` | Patient-matched anatomical replicas, implants, surgical guides |
| | `conductive_electronics` | Bio-integrated electronics, biosensors, smart conductive patches |
| **5. Computational & AI** | `process_calibration` | Design of Experiments (DoE), printability windows, response surfaces |
| | `bayesian_optimization` | Gaussian process surrogate modeling for multi-objective tuning |
| | `computer_vision_monitoring`| High-speed cameras, closed-loop droplet/filament monitoring |
| | `deep_learning_models` | Convolutional neural networks, YOLO anomaly detection |
| | `large_language_models` | RAG, Knowledge Graphs, LKM agents, automated literature parsing |
| | `finite_element_analysis` | Fluid-structure interaction (FSI), nozzle shear stress simulation |
| | `open_source_software` | Custom G-code slicers, open-hardware toolpaths, code packages |

---

## ⚡ Architecture: Parallel Multi-Model Fan-Out

To eliminate rate limits and accelerate screening across hundreds of academic papers, the backend implements a **round-robin multi-model fan-out pipeline**:

```
                       PDF Input Directory (uploads/)
                                     │
                                     ▼
                        [PyPDF Text & Section Extractor]
                       Extracts Title, Abstract, & Intro
                                     │
                                     ▼
                    [Asyncio Semaphore Concurrency Limiter]
                        (Configurable 1 to 50 Parallel)
                                     │
             ┌───────────────────────┼───────────────────────┐
             ▼                       ▼                       ▼
    [NVIDIA Nemotron 550B]   [Google Gemma 4 26B]    [NVIDIA Nemotron 30B]
             │                       │                       │
             └───────────────────────┬───────────────────────┘
                                     │
                                     ▼
                      [Strict JSON Array Validator]
                     Enforces Taxonomy Schema Rules
                                     │
                                     ▼
                     [Thread-Safe Progress Tracker]
                   Saves to progress.json & Live WS HUD
                                     │
             ┌───────────────────────┴───────────────────────┐
             ▼                                               ▼
   categorized_papers.json                           30 Category Markdown Lists
    (Master Classification)                           (e.g., extrusion_based.md)
```

### Key Engineering Features:
1. **Zero-Shot Multi-Label Prompting**: The system prompt forces the LLM to return *all* applicable tags as a strict JSON array without conversational preamble.
2. **Asynchronous Semaphore Throttling**: Users can adjust the concurrency limit on the fly via the web UI slider, safely bursting up to 50 parallel requests.
3. **Resilient Pause & Resume**: The `progress_tracker.py` persists every classified paper to `backend/data/progress.json`. Interrupted jobs resume instantly without re-processing completed papers.

---

## 📂 Directory Structure

```
NTU-MultiTagging-LLM-Classification----Dave-/
├── backend/
│   ├── server.py             # FastAPI backend with REST & WebSocket telemetry (Port 8044)
│   ├── reranker_core.py      # Schema definitions, 30-category prompt logic, API router
│   ├── pdf_processor.py      # Text extraction and header/abstract boundary detection
│   ├── progress_tracker.py   # Thread-safe state tracker and incidence counter
│   ├── test_llm.py           # Diagnostic script for API connectivity
│   ├── requirements.txt      # Python dependencies
│   └── data/                 # Local progress state (progress.json)
├── css/
│   └── style.css             # High-contrast dark dashboard styling
├── js/
│   └── app.js                # WebSocket client, real-time counter animations, live tables
├── output/                   # Master classification exports:
│   ├── categorized_papers.json   # Master JSON mapping (528 papers -> tags)
│   ├── master_tags.csv           # Tabular binary incidence matrix (58 KB)
│   └── *.md                      # 30 Individual markdown lists per category
├── uploads/                  # Input folder for research PDFs (.gitkeep preserved)
├── index.html                # Interactive Real-Time Web Dashboard (Port 8044)
├── requirements.txt          # Root Python dependencies
├── .env.example              # Sanitized API key template
└── .gitignore                # Production exclusions (ignoring PDFs, .venv, .env)
```

---

## 📊 Pre-Classified Dataset Included

This repository comes pre-loaded with the verified classification output for **528 bioprinting research papers**:

* [`output/categorized_papers.json`](output/categorized_papers.json): Complete dictionary mapping each paper filename to its assigned taxonomy tags.
* [`output/master_tags.csv`](output/master_tags.csv): Comma-separated matrix suitable for Pandas, R, or Gephi network analysis.
* **30 Markdown Category Digests** ([`output/*.md`](output/)): Each file catalogs all matching papers, total count, and domain metadata.

### Top Category Distribution Summary (528 Papers):
* **Tissue Engineering**: 445 papers
* **Extrusion-Based Modality**: 135 papers
* **Natural Polymers**: 128 papers
* **Hydrogel Rheology & Viscosity**: 89 papers
* **Stem Cells**: 58 papers
* **Disease Modeling**: 47 papers
* **Light-Based / Vat Bioprinting**: 42 papers
* **Primary Cells**: 39 papers
* **Drug Screening & Discovery**: 31 papers
* **Vascularization**: 31 papers
* **4D Bioprinting**: 18 papers

---

## 🚀 Quickstart Guide

### 1. Prerequisites
* Python 3.10 or 3.11
* An OpenRouter API Key

### 2. Installation
```bash
# Clone repository
git clone https://github.com/DSeahYS/NTU-MultiTagging-LLM-Classification----Dave-.git
cd NTU-MultiTagging-LLM-Classification----Dave-

# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Environment Configuration
Create a `.env` file in the repository root:
```bash
cp .env.example .env
```
Edit `.env` and supply your OpenRouter key:
```env
openrouterkey=sk-or-v1-your-openrouter-key-here
OPENROUTER_MODEL=nvidia/nemotron-3-ultra-550b-a55b:free
```

### 4. Running the Dashboard
```bash
cd backend
python server.py
```
Open your browser and navigate to:
👉 **`http://localhost:8044`**

* Place new research PDFs into `uploads/`.
* Set your desired concurrency (1 to 50) using the slider.
* Click **⚡ Start Parallel Tagging** to classify papers in real time.
* Click **Export Data** to compile updated `.json`, `.csv`, and `.md` files in `output/`.

---

## 📜 Citation & Academic Use

If you utilize this multi-tag classification system, prompt directives, or the resulting 30-category bioprinting taxonomy dataset in your research, please cite:

```bibtex
@software{seah2026multitag_classifier,
  author       = {Seah, Dave Yong Sheng and Huang, Xi},
  title        = {NTU Multi-Tagging LLM Classification & Reranking Engine for Additive Manufacturing},
  year         = {2026},
  publisher    = {GitHub},
  journal      = {GitHub repository},
  howpublished = {\url{https://github.com/DSeahYS/NTU-MultiTagging-LLM-Classification----Dave-}},
  institution  = {Singapore Centre for 3D Printing (SC3DP), Nanyang Technological University (NTU)}
}
```

---

## 📄 License

This software and taxonomy dataset are released for **Academic & Research Use Only** under the NTU Visiting Researcher Programme in Additive Manufacturing.
