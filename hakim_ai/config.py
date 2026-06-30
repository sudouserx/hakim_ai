"""
Configuration loader for the histopathology AI pipeline.

Design: PipelineConfig is the single object passed to every component.
Components read only the sub-config they need. YAML is the canonical
format; the dataclass approach avoids pydantic as an optional dependency.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional
import os

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

try:
    import yaml
    _YAML_AVAILABLE = True
except ImportError:
    _YAML_AVAILABLE = False


# ---------------------------------------------------------------------------
# Sub-configs per layer
# ---------------------------------------------------------------------------

@dataclass
class QCConfig:
    min_stain_quality: float = 0.5
    min_focus_quality: float = 0.5
    min_coverage: float = 0.3
    stain_normalizer: str = "macenko"      # "macenko" | "reinhard" | "passthrough"
    reject_on_artifact: bool = True


@dataclass
class RouterConfig:
    benign_confidence_threshold: float = 0.85
    escalation_confidence_threshold: float = 0.40
    complexity_thresholds: Dict[str, float] = field(
        default_factory=lambda: {"routine": 0.70, "intermediate": 0.40}
    )


@dataclass
class NavigationConfig:
    magnification_levels: List[int] = field(default_factory=lambda: [5, 20, 40])
    top_k_patches: int = 20
    patch_size: int = 512
    encoder_model: str = "uni2"   # "uni2"


@dataclass
class SegmentationConfig:
    model: str = "segformer"                # "segformer" | "hovernet" | "cellvit"
    checkpoint_path: str = "checkpoints/segformer_gchtid.pt"
    num_classes: int = 6
    patch_size: int = 512
    tissue_classes: List[str] = field(
        default_factory=lambda: [
            "tumour", "stroma", "til", "necrosis", "normal_gland", "background"
        ]
    )


@dataclass
class DescriptionConfig:
    vlm_model: str = "pathchat"   # "pathchat" | "conch"
    max_patches_to_describe: int = 5
    max_tokens: int = 256


@dataclass
class MolecularConfig:
    model: str = "multi_task_head"
    msi_threshold: float = 0.53
    her2_threshold: float = 0.5
    ebv_threshold: float = 0.48


@dataclass
class RAGConfig:
    knowledge_base_path: Optional[str] = None
    top_k_similar_cases: int = 3
    top_k_guidelines: int = 5
    embedding_model: str = "biomed_bert"      # "biomed_bert" | "text-embedding-3"


@dataclass
class VerificationConfig:
    temperature: float = 1.5           # temperature scaling for calibration
    abstention_threshold: float = 0.35
    ood_threshold: float = 0.25
    calibrated: bool = False
    calibration_config: Optional[str] = None


@dataclass
class TrainingConfig:
    tcga_data_root: Optional[str] = "data/tcga-stad/"
    tcga_feature_dir: Optional[str] = "data/tcga-stad/features/"
    tcga_manifest_csv: Optional[str] = "data/tcga-stad/manifest.csv"
    gashis_data_root: Optional[str] = "data/gashis/"
    gchtid_data_root: Optional[str] = "data/gchtid/"
    batch_size: int = 32
    learning_rate: float = 1e-4
    num_epochs: int = 50
    num_folds: int = 5
    checkpoint_dir: str = "checkpoints/"
    device: str = "auto"


@dataclass
class FoundationModelConfig:
    patch_encoder: str = "uni2"        # "uni2" | "virchow2" | "gigapath"
    slide_encoder: str = "conch"       # "conch" | "titan"
    vlm: str = "pathchat"
    use_gpu: bool = True


@dataclass
class UIConfig:
    output_dir: str = "outputs"
    html_report: bool = True
    mdt_export: bool = True
    feedback_db_path: str = "outputs/feedback.jsonl"


# ---------------------------------------------------------------------------
# Root config
# ---------------------------------------------------------------------------

@dataclass
class PipelineConfig:
    """Root configuration object — single source of truth for all agents."""
    pipeline_name: str = "histopath_ai_gastric"
    version: str = "0.1.0"
    output_dir: str = "outputs"
    log_level: str = "INFO"

    qc: QCConfig = field(default_factory=QCConfig)
    router: RouterConfig = field(default_factory=RouterConfig)
    navigation: NavigationConfig = field(default_factory=NavigationConfig)
    segmentation: SegmentationConfig = field(default_factory=SegmentationConfig)
    description: DescriptionConfig = field(default_factory=DescriptionConfig)
    molecular: MolecularConfig = field(default_factory=MolecularConfig)
    rag: RAGConfig = field(default_factory=RAGConfig)
    verification: VerificationConfig = field(default_factory=VerificationConfig)
    foundation_models: FoundationModelConfig = field(default_factory=FoundationModelConfig)
    ui: UIConfig = field(default_factory=UIConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    parallel_multi_slide: bool = False

    # ------------------------------------------------------------------ #
    # Constructors
    # ------------------------------------------------------------------ #

    @classmethod
    def from_yaml(cls, path: str | Path) -> "PipelineConfig":
        if not _YAML_AVAILABLE:
            raise ImportError("Install pyyaml: pip install pyyaml")
        with open(path) as fh:
            raw: Dict[str, Any] = yaml.safe_load(fh) or {}
        return cls._from_dict(raw)

    @classmethod
    def _from_dict(cls, d: Dict[str, Any]) -> "PipelineConfig":
        """Shallow merge: top-level keys override defaults; sub-dicts merge field by field."""
        sub_config_map: Dict[str, type] = {
            "qc": QCConfig,
            "router": RouterConfig,
            "navigation": NavigationConfig,
            "segmentation": SegmentationConfig,
            "description": DescriptionConfig,
            "molecular": MolecularConfig,
            "rag": RAGConfig,
            "verification": VerificationConfig,
            "foundation_models": FoundationModelConfig,
            "ui": UIConfig,
            "training": TrainingConfig,
        }
        cfg = cls()
        for key, val in d.items():
            if key in sub_config_map:
                sub_obj = getattr(cfg, key)
                if isinstance(val, dict):
                    for sub_key, sub_val in val.items():
                        if hasattr(sub_obj, sub_key):
                            setattr(sub_obj, sub_key, sub_val)
            elif hasattr(cfg, key):
                setattr(cfg, key, val)
        return cfg

    @classmethod
    def default(cls) -> "PipelineConfig":
        return cls()

    @classmethod
    def for_testing(cls) -> "PipelineConfig":
        cfg = cls()
        cfg.log_level = "WARNING"
        cfg.navigation.top_k_patches = 5
        cfg.description.max_patches_to_describe = 2
        return cfg