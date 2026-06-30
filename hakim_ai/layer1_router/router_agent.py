"""
Layer 1 — Router / Triage Agent.

Classifies a slide as benign | suspicious | malignant and assigns:
  - Case complexity (routine | intermediate | complex)
  - Task type (classification | biomarker_prediction | report_generation)
  - Human escalation flag

The router is a lightweight fast classifier that gates the expensive
multi-agent pipeline. It uses patch-level feature statistics from the
encoder plus QC metadata.

Real implementation: uses a fine-tuned ViT-S or
a CONCH zero-shot classifier.
"""
from __future__ import annotations

import random
from typing import Optional

from hakim_ai.foundation_models.base_encoder import BaseEncoder
from hakim_ai.config import RouterConfig
from hakim_ai.types import (
    CaseComplexity,
    DiagnosticLabel,
    QCResult,
    RouterDecision,
    TaskType,
    WSIData,
)
from hakim_ai.utils.logging_utils import get_logger

logger = get_logger("layer1.router")


class RouterAgent:
    """
    Triage gate — fast binary/categorical slide classifier.

    Inputs:  WSIData, QCResult
    Outputs: RouterDecision
    """

    def __init__(self, cfg: RouterConfig, encoder: Optional[BaseEncoder] = None, seed: int = 42):
        self.cfg = cfg
        self.encoder = encoder
        self._rng = random.Random(seed)
        
        self.router_model = None
        if encoder is not None:
            try:
                import torch
                from hakim_ai.training.train_router import RouterClassifier
                import os
                
                self.device = torch.device("cuda" if torch.cuda.is_available() and getattr(encoder, "use_gpu", False) else "cpu")
                self.router_model = RouterClassifier(input_dim=encoder.embedding_dim).to(self.device)
                ckpt_path = "checkpoints/router_head.pt"
                if os.path.exists(ckpt_path):
                    self.router_model.load_state_dict(torch.load(ckpt_path, map_location=self.device, weights_only=True))
                self.router_model.eval()
            except ImportError:
                logger.warning("Torch not available; router model disabled.")
            except Exception as e:
                logger.warning(f"Could not load router model: {e}")

    def run(self, wsi_data: WSIData, qc_result: QCResult) -> RouterDecision:
        logger.info("Router started for patient %s", wsi_data.patient_id)

        # In a real system: run a lightweight patch encoder + MIL aggregator
        if self.router_model is not None and wsi_data.thumbnail is not None:
            # Mean-pool features from thumbnail patches
            import numpy as np
            from PIL import Image
            
            thumb_np = np.array(wsi_data.thumbnail, dtype=np.uint8)
            h, w = thumb_np.shape[:2]
            patch_size = 224
            patches = []
            
            for y in range(0, h, patch_size):
                for x in range(0, w, patch_size):
                    patch_array = thumb_np[y:y+patch_size, x:x+patch_size]
                    if patch_array.shape[0] == patch_size and patch_array.shape[1] == patch_size:
                        patches.append(Image.fromarray(patch_array))
                        
            if not patches:
                patches.append(Image.fromarray(thumb_np).resize((224, 224)))
            
            try:
                import torch
                # Real feature extraction (batch)
                feats = self.encoder.encode_batch(patches)
                mean_feat = np.mean(feats, axis=0)
                
                with torch.no_grad():
                    feat_t = torch.tensor(mean_feat, dtype=torch.float32).unsqueeze(0).to(self.device)
                    logits = self.router_model(feat_t)
                    probs = torch.softmax(logits, dim=-1).squeeze().tolist()
                    
                # Classes: 0=Benign, 1=Suspicious, 2=Malignant
                class_idx = max(range(3), key=lambda i: probs[i])
                label_confidence = round(probs[class_idx], 4)
                
                if class_idx == 0:
                    label = DiagnosticLabel.BENIGN
                elif class_idx == 1:
                    label = DiagnosticLabel.SUSPICIOUS
                else:
                    label = DiagnosticLabel.MALIGNANT
            except Exception as e:
                logger.error(f"Encoder routing failed: {e}")
                raise RuntimeError(f"Router model inference failed: {e}")
        else:
            raise RuntimeError("Router model or thumbnail is not available for real inference.")

        complexity = self._assess_complexity(label_confidence, qc_result)
        task_type = self._assign_task(label)
        escalate = self._should_escalate(label_confidence)

        rationale = self._build_rationale(label, label_confidence, qc_result)

        decision = RouterDecision(
            label=label,
            complexity=complexity,
            confidence=label_confidence,
            task_type=task_type,
            escalate_to_human=escalate,
            routing_rationale=rationale,
        )

        logger.info(
            "Router decision: %s | complexity=%s | confidence=%.2f | escalate=%s",
            label.value,
            complexity.value,
            label_confidence,
            escalate,
        )
        return decision

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------


    def _assess_complexity(
        self, confidence: float, qc: QCResult
    ) -> CaseComplexity:
        thresholds = self.cfg.complexity_thresholds
        if confidence >= thresholds.get("routine", 0.70) and not qc.artifacts_detected:
            return CaseComplexity.ROUTINE
        if confidence >= thresholds.get("intermediate", 0.40):
            return CaseComplexity.INTERMEDIATE
        return CaseComplexity.COMPLEX

    def _assign_task(self, label: DiagnosticLabel) -> TaskType:
        if label == DiagnosticLabel.BENIGN:
            return TaskType.CLASSIFICATION
        # Malignant / suspicious: run full biomarker prediction
        return TaskType.BIOMARKER_PREDICTION

    def _should_escalate(self, confidence: float) -> bool:
        return confidence < self.cfg.escalation_confidence_threshold

    def _build_rationale(
        self,
        label: DiagnosticLabel,
        confidence: float,
        qc: QCResult,
    ) -> str:
        return (
            f"Slide classified as {label.value} with confidence {confidence:.2f}. "
            f"Stain quality={qc.stain_quality_score:.2f}, "
            f"focus={qc.focus_quality_score:.2f}, "
            f"coverage={qc.coverage_score:.2f}."
        )