"""
Longitudinal analysis agent for multi-slide results.
"""
from __future__ import annotations

from typing import List, Tuple, Dict, Optional
from hakim_ai.types import PipelineResult


class LongitudinalAgent:
    """
    Analyzes multiple PipelineResults (e.g. from different time points or regions)
    to summarize overall findings and predict trajectory.
    """
    
    def __init__(self, use_gpu: bool = True):
        try:
            import torch
            import os
            self.device = torch.device("cuda" if use_gpu and torch.cuda.is_available() else "cpu")
            self.traj_model = None
            ckpt_path = "checkpoints/trajectory_predictor.pt"
            if os.path.exists(ckpt_path):
                # Basic placeholder for real trajectory model
                self.traj_model = torch.jit.load(ckpt_path, map_location=self.device)
                self.traj_model.eval()
        except ImportError:
            self.traj_model = None
            self.device = None

    def analyze(self, results: List[PipelineResult]) -> Tuple[str, Dict[str, float]]:
        if not results:
            return "No data", {}
            
        # Data-driven feature aggregation for trajectory analysis
        msi_high_count = sum(
            1 for r in results 
            if r.fusion and r.fusion.molecular and r.fusion.molecular.msi_status.value == "MSI-H"
        )
        
        avg_tumour = sum(
            r.evidence.segmentation.tumour_fraction for r in results if r.evidence and r.evidence.segmentation
        ) / max(1, len([r for r in results if r.evidence and r.evidence.segmentation]))
        
        summary = f"Analyzed {len(results)} slides. "
        if msi_high_count > 0:
            summary += f"Found MSI-H in {msi_high_count} samples. "
            
        summary += f"Average tumour fraction: {avg_tumour:.2f}."
        
        progression_risk = 0.5
        if self.traj_model is not None:
            import torch
            with torch.no_grad():
                features = torch.tensor([[avg_tumour, msi_high_count]], dtype=torch.float32).to(self.device)
                pred = self.traj_model(features)
                progression_risk = float(torch.sigmoid(pred).item())
        else:
            raise RuntimeError("Trajectory prediction model not available.")
            
        traj = {
            "progression_risk": round(progression_risk, 3)
        }
        
        return summary, traj
