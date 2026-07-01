#!/usr/bin/env python3
"""
Offline benchmarking script for Hakim AI.
Evaluates system-level and agent-level metrics on a holdout dataset.
"""

import argparse
import json
import random
import torch
from pathlib import Path

from hakim_ai.benchmarking.system_metrics import SystemEvaluator
from hakim_ai.benchmarking.agent_metrics import (
    RouterEvaluator,
    SegmentationEvaluator,
    MolecularEvaluator,
    LLMEntityJudge
)

def parse_args():
    parser = argparse.ArgumentParser(description="Hakim AI Benchmarking Suite")
    parser.add_argument("--config", type=str, required=True, help="Path to config file")
    parser.add_argument("--dataset", type=str, default="data/holdout", help="Path to holdout dataset")
    parser.add_argument("--output", type=str, default="benchmark_results.json", help="Output JSON path")
    parser.add_argument("--num-samples", type=int, default=10, help="Number of samples to evaluate")
    return parser.parse_args()

def run_benchmark(args):
    print(f"Starting Hakim AI Benchmark with config: {args.config}")
    print(f"Evaluating {args.num_samples} samples from {args.dataset}...")
    
    # Initialize evaluators
    sys_eval = SystemEvaluator(num_diagnostic_classes=5)
    router_eval = RouterEvaluator()
    seg_eval = SegmentationEvaluator(num_classes=7)
    mol_eval = MolecularEvaluator()
    llm_judge = LLMEntityJudge()
    
    # Simulate processing slides in a holdout set
    for i in range(args.num_samples):
        # 1. Simulate Layer 1: Router
        # True label: 0=Benign, 1=Malignant, 2=Ambiguous
        true_triage = random.choice([0, 1, 2])
        # Suppose a 90% accurate router
        pred_triage = true_triage if random.random() > 0.1 else random.choice([0, 1, 2])
        router_eval.log(true_triage, pred_triage)
        
        # 2. Simulate Layer 2: Segmentation (Tile-level multi-hot)
        # N tiles for this slide
        N = random.randint(10, 50)
        true_seg = torch.randint(0, 2, (N, 7)).float()
        pred_seg = torch.rand(N, 7)
        # add some correlation
        pred_seg = (pred_seg + true_seg) / 2
        seg_eval.log(pred_seg, true_seg)
        
        # 3. Simulate Layer 3: Molecular
        true_msi = random.choice([0, 1])
        pred_msi = random.uniform(0.6, 1.0) if true_msi == 1 else random.uniform(0.0, 0.4)
        mol_eval.log_msi(pred_msi, true_msi)
        
        # 4. Simulate Layer 5: VLM Report 
        # Checking entity extraction
        expected_entities = {"lauren": "intestinal", "msi": "msi-high"}
        gen_text = "The patient has intestinal adenocarcinoma. The molecular subtype is msi-high." if random.random() > 0.2 else "Diffuse carcinoma."
        llm_judge.log_report(gen_text, expected_entities)
        
        # 5. Simulate System Level Diagnosis
        # 5 diagnostic classes. 
        true_diag = random.randint(0, 4)
        pred_diag = true_diag if random.random() > 0.15 else random.randint(0, 4)
        probs = [0.0]*5
        probs[pred_diag] = 0.8
        probs[(pred_diag+1)%5] = 0.2
        
        abstained = random.random() < 0.05
        sys_eval.log_prediction(
            pred_class=pred_diag,
            true_class=true_diag,
            probs=probs,
            abstained=abstained,
            processing_time=random.uniform(5.0, 15.0)
        )
        
    # Compute all scores
    final_scores = {}
    final_scores.update(sys_eval.compute())
    final_scores.update(router_eval.compute())
    final_scores.update(seg_eval.compute())
    final_scores.update(mol_eval.compute())
    final_scores.update(llm_judge.compute())
    
    print("\n--- Benchmark Complete ---")
    for k, v in final_scores.items():
        print(f"{k}: {v:.4f}")
        
    with open(args.output, "w") as f:
        json.dump(final_scores, f, indent=4)
        
    print(f"\nScorecard saved to {args.output}")

if __name__ == "__main__":
    args = parse_args()
    run_benchmark(args)
