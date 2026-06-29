"""
hakim_ai — End-to-end explainable histopathology AI for gastric cancer / STAD.

Quick start::

    from hakim_ai import HistopathologyPipeline
    from hakim_ai.types import PipelineInput, WSIInput

    pipeline = HistopathologyPipeline()
    inp = PipelineInput(wsi_input=WSIInput(wsi_path="slide.svs", patient_id="P001"))
    result = pipeline.run(inp)
    print(result.report.diagnosis.primary_diagnosis)
"""
from hakim_ai.pipeline import HistopathologyPipeline, build_pipeline
from hakim_ai.config import PipelineConfig
from hakim_ai.types import PipelineInput, WSIInput, ClinicalInput, PipelineResult

__version__ = "0.1.0"
__all__ = [
    "HistopathologyPipeline",
    "build_pipeline",
    "PipelineConfig",
    "PipelineInput",
    "WSIInput",
    "ClinicalInput",
    "PipelineResult",
]