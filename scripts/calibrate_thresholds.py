"""
Threshold calibration script.
Outputs a JSON configuration file with optimal probability thresholds based on real model inference.
"""
import argparse
import json
import sys
import os
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

# Add project root to Python path
sys.path.append(str(Path(__file__).parent.parent))

from hakim_ai.training.calibration import calibrate_model_thresholds
from hakim_ai.config import PipelineConfig
from hakim_ai.models.abmil import GatedAttentionMIL
from hakim_ai.models.multi_task_head import MultiTaskHead
from hakim_ai.training.dataset_tcga_stad import TCGAStadDataset, collate_mil_bags


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/default.yaml", help="Path to PipelineConfig YAML")
    parser.add_argument("--checkpoint", default="checkpoints/abmil_multi_task.pt", help="Path to model checkpoint")
    parser.add_argument("--output", default="thresholds.json")
    parser.add_argument("--batch_size", type=int, default=1, help="Inference batch size")
    args = parser.parse_args()
    
    cfg = PipelineConfig.from_yaml(args.config)
    
    if not cfg.training.tcga_feature_dir or not cfg.training.tcga_manifest_csv:
        print("Error: TCGA dataset paths not configured in PipelineConfig.")
        sys.exit(1)
        
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if cfg.training.device != "auto":
        device = torch.device(cfg.training.device)
        
    print(f"Loading dataset from {cfg.training.tcga_feature_dir}...")
    dataset = TCGAStadDataset(cfg.training.tcga_feature_dir, cfg.training.tcga_manifest_csv)
    if len(dataset) == 0:
        print("Error: Dataset is empty.")
        sys.exit(1)
        
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_mil_bags)
    
    # Initialize Models (Assuming embed_dim=1536 from UNI2)
    embed_dim = 1536
    mil = GatedAttentionMIL(input_dim=embed_dim, hidden_dim=256).to(device)
    head = MultiTaskHead(input_dim=embed_dim).to(device)
    
    if os.path.exists(args.checkpoint):
        print(f"Loading weights from {args.checkpoint}...")
        checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=True)
        mil.load_state_dict(checkpoint['mil'])
        head.load_state_dict(checkpoint['head'])
    else:
        print(f"Warning: Checkpoint {args.checkpoint} not found. Running with random weights!")
        
    mil.eval()
    head.eval()
    
    val_labels = {'msi': [], 'ebv': []}
    val_probs = {'msi': [], 'ebv': []}
    
    print("Running inference...")
    with torch.no_grad():
        for features, mask, labels in tqdm(dataloader, desc="Validating"):
            features = features.to(device)
            mask = mask.to(device)
            
            slide_embed, _ = mil(features, mask)
            logits = head(slide_embed)
            
            # Convert logits to probabilities via sigmoid
            msi_prob = torch.sigmoid(logits['msi']).cpu().numpy()
            ebv_prob = torch.sigmoid(logits['ebv']).cpu().numpy()
            
            val_probs['msi'].extend(msi_prob.flatten().tolist())
            val_probs['ebv'].extend(ebv_prob.flatten().tolist())
            
            val_labels['msi'].extend(labels['msi'].cpu().numpy().flatten().tolist())
            val_labels['ebv'].extend(labels['ebv'].cpu().numpy().flatten().tolist())
            
    # Run core calibration logic
    print("Calibrating thresholds...")
    results = calibrate_model_thresholds(val_labels, val_probs)
    print("Calibration Results:", results)
    
    # Add other defaults that might not be calibrated by ROC curve
    if "her2_threshold" not in results:
        results["her2_threshold"] = 0.5
    if "ood_threshold" not in results:
        results["ood_threshold"] = 0.25
    
    with open(args.output, "w") as f:
        json.dump(results, f, indent=4)
        
    print(f"Saved optimized thresholds to {args.output}")


if __name__ == "__main__":
    main()
