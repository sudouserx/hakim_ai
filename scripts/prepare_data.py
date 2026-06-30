#!/usr/bin/env python3
"""
CLI tool for downloading and preparing datasets for hakim_ai.
"""
from __future__ import annotations

import argparse
import logging
import sys
import os

# Allow running from project root without installing
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from hakim_ai.config import PipelineConfig
from hakim_ai.training.data_manager import DatasetRegistry
from hakim_ai.utils.logging_utils import setup_logging

logger = logging.getLogger("prepare_data")

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="hakim_ai — Dataset Acquisition and Preprocessing",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    
    p.add_argument("--config", default=None, help="Path to YAML config file")
    
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--all", action="store_true", help="Download and prepare all datasets defined in config")
    g.add_argument("--dataset", type=str, help="Name of the dataset to process (e.g. gashis, gchtid, tcga-stad)")
    g.add_argument("--url", type=str, help="Official URL or download link of the dataset to process")
    
    p.add_argument("--output-dir", type=str, default=None, help="Override default data_root")
    p.add_argument("--dry-run", action="store_true", help="Print actions without actually downloading")
    p.add_argument("--max-slides", type=int, default=None, help="Limit number of WSIs downloaded for TCGA-STAD")
    p.add_argument("--verify", action="store_true", help="Only run verification on existing dataset")
    p.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    
    return p

def main():
    parser = build_parser()
    args = parser.parse_args()
    
    setup_logging(args.log_level)
    
    if args.config:
        cfg = PipelineConfig.from_yaml(args.config)
    else:
        cfg = PipelineConfig.default()
        
    data_root = args.output_dir or cfg.data.data_root
    
    # Base kwargs from config
    kwargs = {
        "train_ratio": cfg.data.train_ratio,
        "val_ratio": cfg.data.val_ratio,
        "test_ratio": cfg.data.test_ratio,
        "gashis_patch_sizes": cfg.data.gashis_patch_sizes,
        "chunk_size_mb": cfg.data.chunk_size_mb
    }
    
    targets = []
    
    if args.all:
        targets = [
            ("tcga-stad", cfg.data.tcga_stad_url),
            ("gashis", cfg.data.gashis_url),
            ("gchtid", cfg.data.gchtid_url)
        ]
    elif args.dataset:
        # Resolve URL from config if possible
        url = None
        if args.dataset.lower() == "tcga-stad":
            url = cfg.data.tcga_stad_url
        elif args.dataset.lower() == "gashis":
            url = cfg.data.gashis_url
        elif args.dataset.lower() == "gchtid":
            url = cfg.data.gchtid_url
        targets = [(args.dataset, url)]
    elif args.url:
        targets = [(args.url, args.url)]
        
    for identifier, url in targets:
        logger.info(f"Processing {identifier}...")
        handler = DatasetRegistry.get_handler(identifier, data_root=data_root, url=url, **kwargs)
        
        if not handler:
            logger.error(f"No handler found for {identifier} or {url}")
            continue
            
        if args.verify:
            handler.verify()
        else:
            handler.download(dry_run=args.dry_run, max_slides=args.max_slides)
            if not args.dry_run:
                handler.preprocess()
                handler.verify()
                
    logger.info("Done.")
    return 0

if __name__ == "__main__":
    sys.exit(main())
