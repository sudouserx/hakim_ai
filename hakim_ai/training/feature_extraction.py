"""
Offline feature extraction script.
Extracts patch features from WSIs using UNI 2 / CONCH and saves them as PyTorch tensors.
"""
from __future__ import annotations

import argparse
import os
import torch
from pathlib import Path
from tqdm import tqdm

from hakim_ai.config import FoundationModelsConfig, PipelineConfig
from hakim_ai.foundation_models import build_patch_encoder
from hakim_ai.layer0_input import build_wsi_loader
from hakim_ai.utils.image_utils import compute_tissue_mask, extract_patch_from_wsi, get_patch_coordinates, build_normalizer


def extract_features(cfg: PipelineConfig, level: int = 1, patch_size: int = 256):
    if not cfg.training.tcga_data_root or not cfg.training.tcga_feature_dir:
        print("TCGA data root and feature dir must be configured for extraction. Skipping.")
        return
        
    wsi_dir = os.path.join(cfg.training.tcga_data_root, "slides")
    output_dir = cfg.training.tcga_feature_dir
    os.makedirs(output_dir, exist_ok=True)
    
    encoder = build_patch_encoder(cfg.foundation_models)
    wsi_loader = build_wsi_loader()
    normalizer = build_normalizer(cfg.qc.stain_normalizer) if hasattr(cfg.qc, "stain_normalizer") else None
    
    wsi_files = [f for f in os.listdir(wsi_dir) if f.endswith(('.svs', '.ndpi', '.tif', '.tiff'))]
    
    for fname in tqdm(wsi_files, desc="Extracting WSIs"):
        patient_id = Path(fname).stem
        out_path = os.path.join(output_dir, f"{patient_id}.pt")
        
        if os.path.exists(out_path):
            continue
            
        wsi_path = os.path.join(wsi_dir, fname)
        try:
            wsi_data = wsi_loader.load(wsi_path)
            tissue_mask = compute_tissue_mask(wsi_data.thumbnail)
            coords = get_patch_coordinates(wsi_data, tissue_mask, level=level, patch_size=patch_size)
            
            features = []
            for x, y, _ in coords:
                patch = extract_patch_from_wsi(
                    wsi_path, x, y, level, size=(patch_size, patch_size),
                    slide_handle=getattr(wsi_data, "slide_handle", None),
                    normalizer=normalizer
                )
                if patch is not None:
                    feat = encoder.encode_patch(patch)
                    features.append(feat)
                    
            if features:
                features_t = torch.tensor(features, dtype=torch.float32)
                torch.save(features_t, out_path)
                
        except Exception as e:
            print(f"Error processing {fname}: {e}")
        finally:
            if hasattr(wsi_data, "slide_handle") and wsi_data.slide_handle:
                wsi_data.slide_handle.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/default.yaml", help="Path to YAML config")
    args = parser.parse_args()
    
    cfg = PipelineConfig.from_yaml(args.config)
    extract_features(cfg)
