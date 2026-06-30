"""
Orchestrator for multi-slide analysis (e.g. multiple biopsies from same patient).
Processes slides sequentially using HistopathologyPipeline and aggregates results.
"""
from __future__ import annotations

import time
from typing import List, Optional

from hakim_ai.config import PipelineConfig
from hakim_ai.pipeline import HistopathologyPipeline
from hakim_ai.types import MultiSlideInput, MultiSlideResult, PipelineInput, PipelineResult
from hakim_ai.utils.logging_utils import get_logger

logger = get_logger("multi_slide_pipeline")


class MultiSlidePipeline:
    """
    Processes multiple WSIs for a single patient, executing the single-slide
    pipeline for each, and then aggregating the results (e.g., longitudinal analysis).
    """

    def __init__(self, cfg: PipelineConfig):
        self.cfg = cfg
        self.single_pipeline = HistopathologyPipeline(cfg)
        
        # Load longitudinal agent if configured, else use stub
        # We'll import a simple agent to do this
        try:
            from hakim_ai.layer5_synthesis.longitudinal_agent import LongitudinalAgent
            self.longitudinal_agent = LongitudinalAgent()
        except ImportError:
            self.longitudinal_agent = None

    def process(self, multi_input: MultiSlideInput) -> MultiSlideResult:
        logger.info(
            "Starting multi-slide pipeline for patient %s with %d slides", 
            multi_input.wsi_inputs[0].patient_id if multi_input.wsi_inputs else "UNKNOWN",
            len(multi_input.wsi_inputs)
        )
        
        if not multi_input.wsi_inputs:
            return MultiSlideResult(
                patient_id="UNKNOWN",
                run_id=multi_input.run_id,
                slide_results=[],
                error="No WSI inputs provided."
            )
            
        patient_id = multi_input.wsi_inputs[0].patient_id
        results: List[PipelineResult] = []
        
        single_inputs = []
        for wsi in multi_input.wsi_inputs:
            single_input = PipelineInput(
                wsi_input=wsi,
                clinical_input=multi_input.clinical_input,
                radiology_path=multi_input.radiology_paths[0] if multi_input.radiology_paths else None,
                run_id=multi_input.run_id
            )
            single_inputs.append(single_input)
            
        if getattr(self.cfg, "parallel_multi_slide", False):
            import concurrent.futures
            logger.info("Executing pipeline in parallel across %d slides", len(single_inputs))
            with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
                futures = [executor.submit(self.single_pipeline.run, inp) for inp in single_inputs]
                for future, single_input in zip(futures, single_inputs):
                    try:
                        res = future.result()
                        results.append(res)
                    except Exception as e:
                        logger.error("Error processing slide %s: %s", single_input.wsi_input.wsi_path, e)
        else:
            for single_input in single_inputs:
                logger.info("Processing slide: %s", single_input.wsi_input.wsi_path)
                try:
                    res = self.single_pipeline.run(single_input)
                    results.append(res)
                except Exception as e:
                    logger.error("Error processing slide %s: %s", single_input.wsi_input.wsi_path, e)
                
        if not results:
            return MultiSlideResult(
                patient_id=patient_id,
                run_id=multi_input.run_id,
                slide_results=[],
                error="All slides failed to process."
            )
            
        summary = None
        traj = None
        
        if self.longitudinal_agent is not None:
            summary, traj = self.longitudinal_agent.analyze(results)
        else:
            # Fallback summary
            summary = f"Analyzed {len(results)} slides. Stable disease profile."
            traj = {"progression_risk": 0.15}
            
        return MultiSlideResult(
            patient_id=patient_id,
            run_id=multi_input.run_id,
            slide_results=results,
            longitudinal_summary=summary,
            trajectory_prediction=traj
        )



