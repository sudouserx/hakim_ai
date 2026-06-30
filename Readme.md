# hakim_ai

**End-to-end explainable multi-agent histopathology AI for gastric cancer / STAD**

> ⚠️ Research prototype — not approved for clinical diagnostic use without pathologist oversight.

---

## Overview

`hakim_ai` implements the layered multi-agent architecture described in the research synthesis document, targeting gastric cancer (stomach adenocarcinoma, STAD) as the primary diagnostic domain. It acts as a comprehensive multimodal pipeline capable of integrating whole-slide images (WSI), clinical electronic health records (EHR), radiology imaging (DICOM), and molecular history.

### Why gastric cancer?

- **3rd in global cancer mortality** (GLOBOCAN 2022) with highest burden in Asia-Pacific
- **Moderate AI saturation** — explainable AI lags far behind breast/lung cancer
- **Multiple tractable H&E tasks**: subtype (Lauren), MSI/dMMR prediction, EBV detection
- **Rich public multimodal data**: TCGA-STAD, GasHisSDB, GCHTID with TME annotations
- **Clear clinical value**: MSI-H predicts pembrolizumab eligibility; HER2+ predicts trastuzumab benefit

### Architecture at a glance

```
[WSI + EHR + Radiology (optional)]
        │
  Layer 0: Input & QC          stain normalisation · artefact detection · dynamic area-based coverage gating
        │
  Layer 1: Router / Triage     benign fast-path · complexity scoring · human escalation
        │
  Layer 2: Evidence Collection (Concurrent ThreadPool)
        ├── Navigation Agent   K-Means phenotype clustering · importance map · ROI selection
        ├── Segmentation Agent tissue compartments · TIL density · TME profile
        └── Description Agent  NL patch descriptions via VLM (PathChat / CONCH)
        │
  Layer 3: Multimodal Fusion
        ├── Molecular Agent    MSI · Lauren · EBV prediction via Multi-Head Attention MIL
        ├── Clinical Context   EHR HER2 extraction · Attention-based cross-modal fusion
        ├── Knowledge RAG      FAISS vector store semantic retrieval · WHO criteria · Guidelines
        └── Radiology Agent    DICOM parsing · CT/MRI ↔ H&E cross-modal correlation
        │
  Layer 4: Verification
        ├── Logic Agent        clinical and radiological discordance · internal consistency checks
        ├── WHO Validator      taxonomy compliance against WHO 5th Edition (independent of Lauren class)
        └── Confidence Calibrator  temperature scaling · energy-based OOD detection (LogSumExp)
        │
  Layer 5: Synthesis & Reporting
        ├── Diagnosis Agent    dynamic holistic diagnosis · safe-fail human escalation · grade · TNM
        ├── Explanation Agent  NL explanation · evidence citations · counterfactual note
        └── Report Agent       structured pathology report (WHO/Lauren/TNM/biomarkers)
        │
  Layer 6: Human Interface
        ├── UI Renderer        self-contained HTML pathologist report
        ├── MDT Exporter       MDT presentation text
        └── Feedback Capture   structured pathologist disagreement logging (JSONL)
```

**Key design principles**:
- Separate LLM planning from image execution (TissueLab pattern)
- Natural language explanations over heatmaps (PathFinder pattern)
- Explicit verification before every diagnosis output (WSI-Agents pattern)
- Graceful degradation when modalities are missing
- Calibrated uncertainty with actionable abstention and safe-fail human escalation

---

## Quick start

### Install

```bash
git clone https://github.com/sudouserx/hakim_ai
cd hakim_ai
pip install -e ".[dev,data]"
```

### Dataset Preparation

The pipeline includes an automated data manager to acquire and preprocess datasets from their official sources:

```bash
# Download and prepare all datasets (TCGA-STAD, GasHisSDB, GCHTID)
python scripts/prepare_data.py --all

# Download a specific dataset
python scripts/prepare_data.py --dataset tcga-stad

# Preview operations without downloading
python scripts/prepare_data.py --all --dry-run
```

### Train the models

The training scripts are located within the `hakim_ai/training/` directory. They all accept a `--config` argument to parse hyperparameters.

You execute them from the command line using:

    For the MIL Classifier: python hakim_ai/training/train_mil_classifier.py --config config/kaggle.yaml

    For the Router Classifier: python hakim_ai/training/train_router.py --config config/kaggle.yaml

    For the Segmentation Model: python hakim_ai/training/train_segmentation.py --config config/kaggle.yaml

### Run the demo

```bash
python scripts/demo.py
```

This runs the full pipeline, saves an HTML report and MDT summary, and demonstrates feedback capture.

### Run on a real slide

```bash
python scripts/run_pipeline.py \
    --config config/kaggle.yaml \
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
pytest                          # all tests
pytest --cov=hakim_ai           # with coverage
```

---

## Project structure

```
hakim_ai/
├── config/
│   ├── default.yaml            # default configuration
│   ├── prod.yml                # Production multi-GPU configuration
│   ├── kaggle.yaml             # Kaggle T4 single-GPU constraint configuration
│   └── test.yaml               # CI/CD test configuration
├── hakim_ai/
│   ├── __init__.py             # public API
│   ├── types.py                # all typed dataclasses
│   ├── config.py               # PipelineConfig + sub-configs + dotenv support
│   ├── pipeline.py             # HistopathologyPipeline orchestrator (Thread-safe, GPU cleanup)
│   ├── multi_slide_pipeline.py # ThreadPoolExecutor for concurrent slide batch processing
│   ├── foundation_models/      # Encoders and VLM adapters
│   ├── layer0_input/           # WSI Loading & QC Agents
│   ├── layer1_router/          # Triage / routing agents
│   ├── layer2_evidence/        # Navigation, Segmentation, Description agents
│   ├── layer3_fusion/          # Molecular, Clinical, Knowledge (RAG), and Radiology agents
│   ├── layer4_verification/    # Logic checks, WHO validation, Confidence calibration
│   ├── layer5_synthesis/       # Diagnosis and Report construction
│   ├── layer6_interface/       # HTML rendering and structured feedback capture
│   ├── models/                 # PyTorch architectures (ABMIL, MultiTaskHead)
│   ├── training/               # Model fine-tuning logic, dataset loaders, and data manager
│   │   └── data_manager.py     # Central orchestrator for dataset acquisition
│   └── utils/                  # Core image processing, masking, and FAISS store utilities
├── scripts/
│   ├── run_pipeline.py         # CLI entry point
│   ├── calibrate_thresholds.py # Threshold calibration via PyTorch inference
│   ├── prepare_data.py         # CLI for downloading/preparing datasets
│   └── demo.py                 # annotated walkthrough
└── pyproject.toml
```

---

## Configuration

The configuration resides in the `config/` directory and is governed by `hakim_ai/config.py`. The `PipelineConfig` dataclass acts as a robust, typed central truth for the entire pipeline, exposing dataset paths, scaling hyper-parameters, hardware toggles, and model selections.

```yaml

log_level: INFO
parallel_multi_slide: false   # Enable to process multiple WSIs concurrently via ThreadPoolExecutor

qc:
  min_stain_quality: 0.50
  min_coverage: 0.30          # Scales dynamically for biopsies vs large resections

segmentation:
  num_classes: 7
  tissue_classes:
    - background
    - tumour
    - stroma
    - til
    - necrosis
    - normal_gland
    - muscle

molecular:
  msi_threshold: 0.50         # P(MSI-H) above this → MSI-H label

verification:
  calibrated: true            # utilizes optimized thresholds
  temperature: 1.50           # temperature scaling parameters
  abstention_threshold: 0.35

data:
  data_root: data/
  tcga_stad_url: "https://www.cancerimagingarchive.net/collection/tcga-stad/"
  gashis_url: "https://figshare.com/ndownloader/files/28969725"
  gchtid_url: "https://figshare.com/ndownloader/files/46765759"
  train_ratio: 0.70
  val_ratio: 0.15
  test_ratio: 0.15
  gashis_patch_sizes: [160]

training:
  tcga_data_root: data/tcga-stad/
  tcga_feature_dir: data/tcga-stad/features/
  gashis_data_root: data/gashis/
  gchtid_data_root: data/gchtid/
  batch_size: 16
  device: cuda
```

Load a custom config:

```python
from hakim_ai import HistopathologyPipeline, PipelineConfig

cfg = PipelineConfig.from_yaml("config/prod.yml")
pipeline = HistopathologyPipeline(cfg)
```

---

## Foundation Models & GPU Inference

To use real model weights and enable hardware-accelerated processing:

```bash
pip install "hakim_ai[models]"
```

Provide a `.env` file in the root directory:
```env
HF_TOKEN=hf_your_huggingface_token
```

Then in your config:

```yaml
foundation_models:
  patch_encoder: uni2       # loads MahmoodLab/UNI2-h from HuggingFace
  slide_encoder: conch      # loads MahmoodLab/conch
  vlm: pathchat             # loads microsoft/llava-med-v1.5-mistral-7b (4-bit)
  use_gpu: true
```

Real encoder loading utilizes PyTorch `autocast` for memory-efficient forward passes and sequentially offloads models during execution. This memory lifecycle management ensures that the pipeline can run smoothly on memory-constrained hardware, such as a 15GB Kaggle T4 GPU. The multi-slide architecture streams WSI data to singleton GPU model instances via a thread-based concurrent queue to prevent VRAM thrashing and redundant instantiations.

---

## Core Capabilities

### Clinical Safety & Quality Control
The pipeline prioritizes clinical safety by implementing a robust escalation system. The `DiagnosisAgent` strictly escalates indeterminate cases (e.g., low tumor fraction with no WHO criteria match) for manual review rather than defaulting to negative diagnoses. The `QCAgent` calculates a dynamic `min_coverage` threshold, appropriately adapting to the massive difference in tissue area between endoscopic biopsies and full surgical resections.

### Model Training & Calibration
The repository contains native training logic to fine-tune the pipeline's intelligence layers. The `train_mil_classifier.py` loop utilizes **Focal Loss** to penalize majority classes and learn rare minority features (EBV+, MSI-H). Memory-efficient unbatched gradient accumulation prevents tensor padding memory bombs when evaluating variable-length gigapixel MIL bags.

The `calibrate_thresholds.py` script automatically runs inference over a validation set and uses Youden's J Statistic to calculate the optimal cutoffs balancing false-positives and false-negatives. 

### Multi-Head Attention & Morphological Phenotyping
The `GatedAttentionMIL` module features independent attention branches for each molecular prediction task (MSI, EBV, Lauren). During evidence collection, the `NavigationAgent` applies `K-Means` clustering on the patch feature embeddings. Selecting the highest-scoring patches per cluster ensures the VLM evaluates a comprehensive set of morphological phenotypes and avoids the diffuse carcinoma blindspot.

### Advanced Verification
Robustness is guaranteed via the `ConfidenceCalibrator` and `LogicAgent`. The calibrator calculates an energy-based Out-of-Distribution (OOD) score using the `LogSumExp` distribution across molecular logits to abstain from evaluating ambiguous samples. The `WHOValidator` rigorously evaluates morphological descriptions against the WHO 5th Edition taxonomy, completely independent of the Lauren classification taxonomy.

### Attention-Based Fusion
The system extracts real DICOM metadata via `pydicom` to provide radiology context. Cross-modal correlation fuses this and other unstructured clinical histories with the histopathology features by computing the dot-product cosine similarity between the CONCH NLP embeddings and visual patch vectors. HER2 status is extracted exclusively via the `ClinicalContextAgent` (EHR/IHC reports) to bypass inaccurate H&E morphological hallucinations.

---

## Datasets

The pipeline expects data directories configured directly in the `TrainingConfig` layer, targeting the specific schemas of distinct open-source datasets. Use the included `prepare_data.py` script to automatically download, split, and format the sources into training-ready artifacts.

| Dataset | Content | Access | Handler Mechanics |
|---|---|---|---|
| TCGA-STAD | ~400 WSIs + Clinical | Public (GDC portal) | Queries SVS UUIDs from GDC API. Reconstructs clinical metadata by merging demography from cBioPortal `stad_tcga` with molecular targets (MSI, EBV, Lauren) from `stad_tcga_pub`. |
| GasHisSDB | 245K patches | Public (Zenodo) | Downloads raw `.rar` archive, strictly filters patches to 160×160 resolution, and generates a standardized train/val/test split. |
| GCHTID | 31K images, TME labels | Public (figshare 2024) | Automatically translates the source 9-class annotation scheme into a robust 7-class taxonomy, generating multi-class pseudo-masks. |
| NCT-CRC-HE-100K | 100K patches | Public (Zenodo) | Pretraining patch encoder (external dependency). |

> [!WARNING]
> GCHTID labels are weak proxy labels derived from cross-domain transfer (NCT-CRC-HE-100K colorectal annotations applied to gastric WSIs). Downstream models should be treated as tile-level tissue-composition estimators rather than pixel-precise boundary segmenters.

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

This project integrates foundation models that require proper attribution. If you use this software, you must cite **CONCH** and **UNI**:

### CONCH
```bibtex
@article{lu2024avisionlanguage,
  title={A visual-language foundation model for computational pathology},
  author={Lu, Ming Y and Chen, Bowen and Williamson, Drew FK and Chen, Richard J and Liang, Ivy and Ding, Tong and Jaume, Guillaume and Odintsov, Igor and Le, Long Phi and Gerber, Georg and others},
  journal={Nature Medicine},
  pages={863–874},
  volume={30},
  year={2024},
  publisher={Nature Publishing Group}
}
```

### UNI
```bibtex
@article{chen2024uni,
  title={Towards a General-Purpose Foundation Model for Computational Pathology},
  author={Chen, Richard J and Ding, Tong and Lu, Ming Y and Williamson, Drew FK and Jaume, Guillaume and Chen, Bowen and Zhang, Andrew and Shao, Daniel and Song, Andrew H and Shaban, Muhammad and others},
  journal={Nature Medicine},
  publisher={Nature Publishing Group},
  year={2024}
}
```
*Note: Works that use UNI should also attribute ViT and DINOv2.*

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