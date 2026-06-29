# hakim_ai

**End-to-end explainable multi-agent histopathology AI for gastric cancer / STAD**

> ⚠️ Research prototype — not approved for clinical diagnostic use without pathologist oversight.

---

## Overview

`hakim_ai` implements the layered multi-agent architecture described in the research synthesis document, targeting gastric cancer (stomach adenocarcinoma, STAD) as the primary diagnostic domain. It acts as a comprehensive multimodal pipeline capable of integrating whole-slide images (WSI), clinical electronic health records (EHR), radiology imaging (DICOM), and molecular history.

### Why gastric cancer?

- **3rd in global cancer mortality** (GLOBOCAN 2022) with highest burden in Asia-Pacific
- **Moderate AI saturation** — explainable AI lags far behind breast/lung cancer
- **Multiple tractable H&E tasks**: subtype (Lauren), MSI/dMMR prediction, HER2 approximation, EBV detection
- **Rich public multimodal data**: TCGA-STAD, GasHisSDB, GCHTID with TME annotations
- **Clear clinical value**: MSI-H predicts pembrolizumab eligibility; HER2+ predicts trastuzumab benefit

### Architecture at a glance

```
[WSI + EHR + Radiology (optional)]
        │
  Layer 0: Input & QC          stain normalisation · artefact detection · reject/pass
        │
  Layer 1: Router / Triage     benign fast-path · complexity scoring · human escalation
        │
  Layer 2: Evidence Collection (Concurrent ThreadPool)
        ├── Navigation Agent   multi-scale WSI browsing · importance map · ROI selection
        ├── Segmentation Agent tissue compartments · TIL density · TME profile
        └── Description Agent  NL patch descriptions via VLM (PathChat / CONCH)
        │
  Layer 3: Multimodal Fusion
        ├── Molecular Agent    MSI · Lauren · HER2 · EBV prediction from H&E
        ├── Clinical Context   Attention-based cross-modal fusion (CONCH text embedding vs image features)
        ├── Knowledge RAG      FAISS vector store semantic retrieval · WHO criteria · Guidelines
        └── Radiology Agent    DICOM parsing · CT/MRI ↔ H&E cross-modal correlation
        │
  Layer 4: Verification
        ├── Logic Agent        clinical and radiological discordance · internal consistency checks
        ├── WHO Validator      taxonomy compliance against WHO 5th Edition (12 gastric subtypes)
        └── Confidence Calibrator  temperature scaling · energy-based OOD detection (LogSumExp)
        │
  Layer 5: Synthesis & Reporting
        ├── Diagnosis Agent    dynamic holistic diagnosis · grade · differentials · TNM
        ├── Explanation Agent  NL explanation · evidence citations · counterfactual note
        └── Report Agent       structured pathology report (WHO/Lauren/TNM/biomarkers)
        │
  Layer 6: Human Interface
        ├── UI Renderer        self-contained HTML pathologist report
        ├── MDT Exporter       MDT presentation text
        └── Feedback Capture   structured pathologist disagreement logging (JSONL)
```

**Key design principles** (from architecture document):
- Separate LLM planning from image execution (TissueLab pattern)
- Natural language explanations over heatmaps (PathFinder pattern)
- Explicit verification before every diagnosis output (WSI-Agents pattern)
- Graceful degradation when modalities are missing
- Calibrated uncertainty with actionable abstention

---

## Quick start

### Install (mock mode — no GPU required)

```bash
git clone https://github.com/sudouserx/hakim_ai
cd hakim_ai
pip install -e ".[dev]"
```

### Run the demo

```bash
python scripts/demo.py
```

This runs the full pipeline with synthetic data, saves an HTML report and MDT summary, and demonstrates feedback capture — no WSI files or GPU needed.

### Run on a real slide

```bash
python scripts/run_pipeline.py \
    --patient-id TCGA-BR-4253 \
    --wsi-path /data/slides/TCGA-BR-4253.svs \
    --radiology-path /data/dicom/CT_study.dcm \
    --age 67 --sex M \
    --biopsy-location antrum \
    --h-pylori \
    --endoscopy "3.2cm ulcerative lesion, antrum" \
    --save-report --save-mdt
```

### Run tests

```bash
pytest                          # all tests (202 pass)
pytest tests/test_pipeline.py   # integration only
pytest tests/test_real_mode.py  # end-to-end integration against real WSI
pytest --cov=hakim_ai       # with coverage
```

---

## Project structure

```
hakim_ai/
├── config/
│   ├── default.yaml            # default configuration
│   └── test.yaml               # CI/CD test configuration
├── hakim_ai/
│   ├── __init__.py             # public API
│   ├── types.py                # all typed dataclasses
│   ├── config.py               # PipelineConfig + sub-configs + dotenv support
│   ├── pipeline.py             # HistopathologyPipeline orchestrator (Thread-safe, GPU cleanup)
│   ├── foundation_models/
│   │   ├── base_encoder.py     # ABC for encoders and VLMs
│   │   ├── uni_adapter.py      # UNI 2 patch encoder (stub + real path)
│   │   └── conch_adapter.py    # CONCH slide encoder + PathChat VLM
│   ├── layer0_input/
│   │   ├── wsi_loader.py       # MockWSILoader + OpenSlideWSILoader
│   │   └── qc_agent.py         # quality control agent
│   ├── layer1_router/
│   │   └── router_agent.py     # triage / routing agent
│   ├── layer2_evidence/
│   │   ├── navigation_agent.py # multi-scale ROI selection
│   │   ├── segmentation_agent.py # tissue compartment mapping
│   │   └── description_agent.py  # NL patch descriptions
│   ├── layer3_fusion/
│   │   ├── molecular_agent.py      # MSI/Lauren/HER2/EBV prediction
│   │   ├── clinical_context_agent.py # CONCH attention-based fusion
│   │   ├── knowledge_retrieval_agent.py  # FAISS RAG semantic search
│   │   └── radiology_agent.py      # DICOM metadata extraction
│   ├── layer4_verification/
│   │   ├── logic_agent.py          # consistency and discordance rules
│   │   └── confidence_calibrator.py  # Energy-based OOD + calibration
│   ├── layer5_synthesis/
│   │   ├── diagnosis_agent.py      # dynamic diagnostic labeling
│   │   ├── explanation_agent.py    # NL explanation generation
│   │   └── report_agent.py         # structured pathology report
│   ├── layer6_interface/
│   │   ├── ui_renderer.py          # standalone HTML report
│   │   └── feedback_capture.py     # JSONL feedback + MDT export
│   └── utils/
│       ├── logging_utils.py
│       ├── image_utils.py          # stain normalization, tissue masking
│       └── rag_store.py            # FAISS / BM25 knowledge store
├── scripts/
│   ├── run_pipeline.py         # CLI entry point
│   └── demo.py                 # annotated walkthrough
├── tests/
│   ├── conftest.py             # shared fixtures
│   ├── test_pipeline.py        # integration tests
│   ├── test_real_mode.py       # real-world data testing
│   └── test_*.py               # component tests
└── pyproject.toml
```

---

## Configuration

All configuration lives in `config/default.yaml`. The `PipelineConfig` object is the single source of truth passed to every agent, and securely loads environment variables from a `.env` file for API keys and tokens.

```yaml
mock_mode: true           # false → load real model weights (requires HF_TOKEN)
log_level: INFO

qc:
  min_stain_quality: 0.50
  min_focus_quality: 0.50
  min_coverage: 0.30

molecular:
  msi_threshold: 0.50     # P(MSI-H) above this → MSI-H label

verification:
  temperature: 1.50       # temperature scaling; >1 softens confidence
  abstention_threshold: 0.35
```

Load a custom config:

```python
from hakim_ai import HistopathologyPipeline, PipelineConfig

cfg = PipelineConfig.from_yaml("config/prod.yaml")
pipeline = HistopathologyPipeline(cfg)
```

---

## Foundation Models & GPU Inference

The mock adapters are designed to be dropped in place. To use real model weights and enable hardware-accelerated processing:

```bash
pip install "hakim_ai[models]"
```

Provide a `.env` file in the root directory:
```env
HF_TOKEN=hf_your_huggingface_token
```

Then in your config:

```yaml
mock_mode: false
foundation_models:
  patch_encoder: uni2       # loads MahmoodLab/UNI2-h from HuggingFace
  slide_encoder: conch      # loads MahmoodLab/conch
  vlm: pathchat             # loads microsoft/llava-med-v1.5-mistral-7b (4-bit)
  use_gpu: true
  mock_mode: false
```

Real encoder loading is integrated in `foundation_models/uni_adapter.py` and `conch_adapter.py`. The pipeline utilizes PyTorch `autocast` for memory-efficient forward passes and sequentially offloads models during execution. This memory lifecycle management ensures that the pipeline can run smoothly on memory-constrained hardware, such as a 15GB Kaggle T4 GPU.

---

## Core Capabilities

### FAISS Semantic Retrieval (RAG)
The `KnowledgeRetrievalAgent` maintains a comprehensive vector database of the WHO 5th Edition Gastric Tumour criteria. It retrieves clinical evidence and morphological subtypes (e.g., Medullary, Micropapillary, Adenosquamous) dynamically using SentenceTransformers and FAISS, enabling the pipeline to ground diagnoses in standard pathology literature.

### Advanced Verification
Robustness is guaranteed via the `ConfidenceCalibrator` and `LogicAgent`. The calibrator calculates an energy-based Out-of-Distribution (OOD) score using the `LogSumExp` distribution across molecular logits to abstain from evaluating ambiguous samples. The logic layer identifies multimodal discordances, such as a high-stage radiological presentation contrasting a low tumor-fraction histology result.

### Attention-Based Fusion
The system extracts real DICOM metadata via `pydicom` to provide radiology context. Cross-modal correlation fuses this and other unstructured clinical histories with the histopathology features by computing the dot-product cosine similarity between the CONCH NLP embeddings and visual patch vectors, assigning an intelligent relevance weight to textual evidence.

---

## Datasets

| Dataset | Size | Access | Used for |
|---|---|---|---|
| TCGA-STAD | ~380 WSIs + RNAseq + mutation | Public (GDC portal) | Multi-label training: MSI, EBV, Lauren, HER2 |
| GasHisSDB | 245K patches | Public (Zenodo) | Patch encoder fine-tuning |
| GCHTID | 31K images, TME labels | Public (figshare 2024) | Segmentation agent fine-tuning |
| NCT-CRC-HE-100K | 100K patches | Public (Zenodo) | Pretraining patch encoder |
| KCCH (Yokohama) | ~185 patients | Controlled access | Asian cohort external validation |

---

## Explainability

The system produces three complementary explanation modalities:

1. **Natural language narratives** — patch-level descriptions generated by the Description Agent (PathChat/CONCH VLM), aggregated by the Explanation Agent into a diagnostic narrative
2. **Visual importance map** — 16×16 heat map rendered in the HTML report showing which WSI regions contributed most to the routing and molecular predictions
3. **Counterfactual note** — verbal approximation of "what would change the prediction" (e.g. "If TIL density were <5%, MSI-H prediction would shift toward MSS")

Known limitations (from architecture synthesis):
- Attention ≠ causation: importance maps indicate correlation, not causal evidence
- Counterfactuals at gigapixel scale are technically unsolved; verbal approximations are used
- HER2 from H&E is indicative only — IHC/FISH is always required for clinical decisions

---

## Research gaps addressed

This implementation explicitly targets the gaps identified in the synthesis document:

| Gap | How addressed |
|---|---|
| No gastric cancer multi-agent system | Full STAD-focused pipeline with per-layer agents |
| Attention ≠ explanation | NL description generation replaces heatmap-only explanation |
| No verification layer | Logic Agent + WHO Validator + Energy-based OOD Calibration |
| Over-reliance risk | Uncertainty statements, IHC recommendation flags, abstention mechanism |
| Missing modality robustness | Radiology and clinical data are optional; attention-based fusion |
| No feedback loop | FeedbackCapture records structured disagreements to JSONL |
| Clinical adoption blockers | MDT export, structured WHO/TNM reporting, pathologist sign-off prompts |

---

## Adding a new agent

1. Create `hakim_ai/layer{N}_{name}/my_agent.py`
2. Define typed `run(inputs) -> Output` method
3. Add to the layer's `__init__.py`
4. Wire into `pipeline.py` at the appropriate layer
5. Add fixtures to `tests/conftest.py` and tests to `tests/test_layer{N}_*.py`

---

## Citation

If you use this codebase in research, please cite:

```bibtex
@software{histopath_ai_2025,
  title  = {histopath\_ai: End-to-end explainable multi-agent histopathology AI for gastric cancer},
  year   = {2025},
  note   = {Architecture based on PathFinder (ICCV 2025), WSI-Agents (MICCAI 2025),
            and TissueLab (arXiv 2509.20279)},
}
```

Key references implemented:
- **PathFinder** (Ghezloo et al., ICCV 2025) — 4-agent NL explanation design
- **WSI-Agents** (Lyu et al., MICCAI 2025 Oral) — verification mechanism
- **TissueLab** (arXiv:2509.20279) — LLM planning vs. local model execution
- **UNI 2** (Chen et al., Nature Medicine 2024) — patch encoder backbone
- **CONCH / TITAN** (Lu et al., Nature Medicine 2024) — vision-language encoder
- **Kather et al.** (Nature Medicine 2019) — H&E-based MSI prediction anchor
- **Liu et al.** (NeurIPS 2020) — Energy-based Out-of-Distribution Detection

---

## License

MIT License — see `LICENSE` for details.

> This software is provided for research purposes only. It is not a medical device and has not received regulatory clearance (FDA, CE-IVD, or equivalent). All AI-generated diagnostic outputs require review and sign-off by a qualified pathologist before any clinical action.