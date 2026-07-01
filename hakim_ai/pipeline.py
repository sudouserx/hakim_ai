"""
End-to-end Histopathology AI Pipeline Orchestrator.

Wires together all six processing layers following the architecture
described in the research synthesis document:

  Layer 0: Input management and quality control
  Layer 1: Router / triage agent
  Layer 2: Evidence collection (navigation, segmentation, description)
  Layer 3: Multimodal fusion (molecular, clinical context, RAG, radiology)
  Layer 4: Verification (logic, WHO validation, confidence calibration)
  Layer 5: Synthesis (diagnosis, explanation, report generation)
  Layer 6: Human interface (HTML report, MDT export, feedback)

Design principles enforced here:
  - Separation of orchestration (this file) from execution (agents)
  - Every layer failure is caught and surfaced in PipelineResult.error
  - Benign fast-path: abbreviated workflow for clearly benign cases
  - Abstention: low-confidence cases escalate before diagnosis is emitted
  - Layer 2 agents run sequentially (models assumed persistently managed)
"""
from __future__ import annotations

import time
import uuid
from typing import Optional

from hakim_ai.config import PipelineConfig
from hakim_ai.foundation_models import build_patch_encoder, build_vlm
from hakim_ai.layer0_input import QCAgent, build_wsi_loader, ClinicalLoader
from hakim_ai.layer1_router import RouterAgent
from hakim_ai.layer2_evidence import NavigationAgent, SegmentationAgent, DescriptionAgent
from hakim_ai.layer3_fusion import (
    MolecularPredictionAgent,
    ClinicalContextAgent,
    KnowledgeRetrievalAgent,
    RadiologyPathologyAgent,
)
from hakim_ai.layer4_verification import LogicAgent, WHOValidator, ConfidenceCalibrator
from hakim_ai.layer5_synthesis import DiagnosisAgent, ExplanationAgent, ReportAgent
from hakim_ai.layer6_interface import UIRenderer, MDTExporter
from hakim_ai.types import (
    DiagnosticLabel,
    EvidenceBundle,
    FusionResult,
    PipelineInput,
    PipelineResult,
)
from hakim_ai.utils import RAGStore, setup_logging
from hakim_ai.utils.image_utils import build_normalizer
from hakim_ai.utils.logging_utils import get_logger


class HistopathologyPipeline:
    """
    End-to-end histopathology AI pipeline for gastric cancer / STAD.

    Usage::

        cfg = PipelineConfig.default()
        pipeline = HistopathologyPipeline(cfg)
        result = pipeline.run(pipeline_input)
        print(result.report.diagnosis.primary_diagnosis)
    """

    def __init__(self, config: Optional[PipelineConfig] = None):
        self.config = config or PipelineConfig.default()
        cfg = self.config
        setup_logging(cfg.log_level)
        self._logger = get_logger("pipeline")

        # Foundation models (shared across agents)
        encoder = build_patch_encoder(cfg.foundation_models)
        vlm = build_vlm(cfg.foundation_models)

        # Layer 0
        self.wsi_loader = build_wsi_loader()
        self.clinical_loader = ClinicalLoader()
        self.qc_agent = QCAgent(cfg.qc)

        # Layer 1
        self.router = RouterAgent(cfg.router, encoder)

        # Extract the global checkpoint dir (defaults to "checkpoints" if missing)
        ckpt_dir = getattr(cfg.training, "checkpoint_dir", "checkpoints")

        # Layer 2
        normalizer = build_normalizer(cfg.qc.stain_normalizer) if hasattr(cfg.qc, "stain_normalizer") else None
        self.navigation_agent = NavigationAgent(cfg.navigation, encoder, normalizer=normalizer, checkpoint_dir=ckpt_dir)
        self.segmentation_agent = SegmentationAgent(cfg.segmentation, normalizer=normalizer)
        self.description_agent = DescriptionAgent(cfg.description, vlm, normalizer=normalizer)

        # Layer 3
        from hakim_ai.foundation_models.conch_adapter import CONCHEncoder
        conch_encoder = CONCHEncoder()
        
        if not cfg.rag.knowledge_base_path:
            # Fallback path if none provided but store is needed
            kb_path = "data/knowledge_base.json" 
        else:
            kb_path = cfg.rag.knowledge_base_path
            
        import os, json
        if not os.path.exists(kb_path):
            os.makedirs(os.path.dirname(kb_path), exist_ok=True)
            with open(kb_path, 'w') as f:
                json.dump({"documents": [], "cases": []}, f)

        rag_store = RAGStore(knowledge_base_path=kb_path)
        self.molecular_agent = MolecularPredictionAgent(cfg.molecular, conch_encoder, checkpoint_dir=ckpt_dir)
        self.clinical_context_agent = ClinicalContextAgent(encoder=conch_encoder)
        self.knowledge_agent = KnowledgeRetrievalAgent(cfg.rag, store=rag_store)
        self.radiology_agent = RadiologyPathologyAgent()

        # Layer 4
        self.logic_agent = LogicAgent()
        self.who_validator = WHOValidator()
        self.calibrator = ConfidenceCalibrator(cfg.verification)

        # Layer 5
        self.diagnosis_agent = DiagnosisAgent(cfg=cfg.training)
        self.explanation_agent = ExplanationAgent()
        self.report_agent = ReportAgent()

        # Layer 6
        self.ui_renderer = UIRenderer(output_dir=cfg.ui.output_dir)
        self.mdt_exporter = MDTExporter()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self, pipeline_input: PipelineInput) -> PipelineResult:
        """Execute the full pipeline and return a PipelineResult."""
        run_id = pipeline_input.run_id or str(uuid.uuid4())[:8]
        patient_id = pipeline_input.wsi_input.patient_id
        start = time.monotonic()

        self._logger.info("=" * 60)
        self._logger.info(
            "Pipeline run %s | patient %s | wsi=%s",
            run_id, patient_id, pipeline_input.wsi_input.wsi_path,
        )

        # Placeholder result; populated progressively
        result = PipelineResult(
            patient_id=patient_id,
            run_id=run_id,
            qc_result=None,       # type: ignore[arg-type]
            router_decision=None, # type: ignore[arg-type]
        )

        try:
            result = self._run_layers(pipeline_input, result, run_id)
        except Exception as exc:
            self._logger.exception("Unhandled pipeline error: %s", exc)
            result.error = str(exc)

        result.pipeline_duration_seconds = round(time.monotonic() - start, 3)
        self._logger.info(
            "Pipeline complete in %.2fs | success=%s",
            result.pipeline_duration_seconds,
            result.is_successful(),
        )
        return result

    # ------------------------------------------------------------------
    # Layer execution
    # ------------------------------------------------------------------

    def _unload_model(self, obj: Any) -> None:
        """Call unload() on an object if parallel multi-slide is disabled to save VRAM."""
        if not self.config.parallel_multi_slide and hasattr(obj, "unload"):
            obj.unload()

    def _run_layers(
        self,
        inp: PipelineInput,
        result: PipelineResult,
        run_id: str,
    ) -> PipelineResult:

        # ── Layer 0: Input & QC ─────────────────────────────────────── #
        wsi_data = None
        try:
            wsi_data = self.wsi_loader.load(inp.wsi_input)
            clinical_data = self.clinical_loader.load(inp.clinical_input)

            qc_result = self.qc_agent.run(wsi_data)
            result.qc_result = qc_result

            if not qc_result.passed:
                result.error = f"QC failed: {qc_result.rejection_reason}"
                self._logger.warning("Stopping pipeline: QC failure")
                return result

            # ── Layer 1: Router ──────────────────────────────────────────── #
            router_decision = self.router.run(wsi_data, qc_result)
            result.router_decision = router_decision

            if router_decision.escalate_to_human:
                result.escalated_to_human = True
                self._logger.warning(
                    "Case flagged for human escalation (router confidence=%.2f)",
                    router_decision.confidence,
                )

            # Benign fast-path: abbreviated report without full multi-agent analysis
            if (
                router_decision.label == DiagnosticLabel.BENIGN
                and not router_decision.escalate_to_human
            ):
                self._logger.info("Benign fast-path: abbreviated report")
                result.report = self.report_agent.generate_benign_report(
                    inp, qc_result, router_decision
                )
                return result

            # ── Layer 2: Evidence Collection (sequential) ─────────────────── #
            # Models are assumed to be persistently loaded and managed by an
            # external inference server or initialised once at pipeline
            # construction.  No manual .load() / .unload() cycling — this
            # avoids GPU thrashing from moving weights in and out of VRAM.
            # Agents run sequentially on the main thread to avoid GIL
            # bottlenecks and CUDA context collisions from ThreadPoolExecutor.
            nav_result = self.navigation_agent.run(wsi_data, qc_result)
            self._unload_model(self.navigation_agent)
            if hasattr(self.router, "encoder"):
                self._unload_model(self.router.encoder)

            seg_result = self.segmentation_agent.run(wsi_data, nav_result)
            self._unload_model(self.segmentation_agent)

            desc_result = self.description_agent.run(wsi_data, nav_result)
            self._unload_model(self.description_agent)
            if hasattr(self.description_agent, "vlm"):
                self._unload_model(self.description_agent.vlm)

            evidence = EvidenceBundle(
                navigation=nav_result,
                segmentation=seg_result,
                descriptions=desc_result,
            )
            result.evidence = evidence

            # ── Layer 3: Multimodal Fusion ───────────────────────────────── #
            molecular = self.molecular_agent.run(evidence)
            self._unload_model(self.molecular_agent)
            
            clinical_ctx = self.clinical_context_agent.run(clinical_data, evidence)
            if hasattr(self.clinical_context_agent, "encoder"):
                self._unload_model(self.clinical_context_agent.encoder)
            knowledge = self.knowledge_agent.run(molecular, evidence)

            radiology_findings = None
            if inp.radiology_path and not router_decision.skip_radiology_fusion:
                radiology_findings = self.radiology_agent.run(inp.radiology_path, evidence)

            fusion = FusionResult(
                molecular=molecular,
                clinical_context=clinical_ctx,
                knowledge=knowledge,
                radiology_findings=radiology_findings,
            )
            result.fusion = fusion

            # ── Layer 4: Verification ────────────────────────────────────── #
            logic_check = self.logic_agent.run(evidence, fusion)
            who_validation = self.who_validator.run(fusion, evidence)
            verification = self.calibrator.run(logic_check, who_validation, fusion, evidence)
            result.verification = verification
    
            # Abstention path
            if verification.abstain:
                result.escalated_to_human = True
                self._logger.warning(
                    "Abstention: %s", verification.abstention_reason
                )
                result.report = self.report_agent.generate_uncertainty_report(
                    inp, evidence, fusion, verification
                )
                return result
    
            # ── Layer 5: Synthesis & Report ──────────────────────────────── #
            diagnosis = self.diagnosis_agent.run(evidence, fusion, verification, result.router_decision)
            self._unload_model(self.diagnosis_agent)
            explanation = self.explanation_agent.run(diagnosis, evidence, fusion, verification)
            report = self.report_agent.run(inp, diagnosis, explanation, fusion, verification)
            result.report = report
    
            return result
        
        finally:
            if wsi_data is not None and getattr(wsi_data, "slide_handle", None) is not None:
                try:
                    wsi_data.slide_handle.close()
                except Exception as e:
                    self._logger.warning(f"Failed to close slide handle: {e}")

    # NOTE: _cleanup_gpu() has been intentionally removed.
    # Aggressive gc.collect() + torch.cuda.empty_cache() between every agent
    # step caused GPU thrashing.  Models are now assumed to be persistently
    # managed (either loaded once at init or served by an inference server).

    # ------------------------------------------------------------------
    # Layer 6 helpers (called explicitly by the user/CLI after run())
    # ------------------------------------------------------------------

    def save_html_report(self, result: PipelineResult) -> str:
        """Render and save the HTML pathologist report; return file path."""
        return self.ui_renderer.save(result)

    def save_mdt_summary(self, result: PipelineResult) -> str:
        """Save the MDT presentation text; return file path."""
        return self.mdt_exporter.save(
            result, output_dir=self.config.ui.output_dir
        )


# ---------------------------------------------------------------------------
# Module-level convenience constructor
# ---------------------------------------------------------------------------

def build_pipeline(config_path: Optional[str] = None) -> HistopathologyPipeline:
    """Build a pipeline from a YAML config file or default config."""
    if config_path:
        cfg = PipelineConfig.from_yaml(config_path)
    else:
        cfg = PipelineConfig.default()
    return HistopathologyPipeline(cfg)