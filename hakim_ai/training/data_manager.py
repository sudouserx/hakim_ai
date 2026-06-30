import os
import re
import json
import shutil
import logging
import random
from pathlib import Path
from typing import Dict, List, Optional, Type, Any, Tuple
import requests
from urllib.parse import urlparse
from tqdm import tqdm
import pandas as pd
import numpy as np
from PIL import Image
import zipfile

logger = logging.getLogger(__name__)

class BaseDatasetHandler:
    """Abstract base class for dataset handlers."""
    def __init__(self, data_root: str, url: str, **kwargs):
        self.data_root = Path(data_root)
        self.url = url
        self.session = requests.Session()
        self.kwargs = kwargs
    
    def download(self, dry_run: bool = False, **kwargs):
        raise NotImplementedError
        
    def preprocess(self):
        raise NotImplementedError
        
    def verify(self):
        raise NotImplementedError
        
    def _download_file(self, url: str, dest: Path, chunk_size_mb: int = 8, dry_run: bool = False) -> bool:
        if dry_run:
            logger.info(f"[Dry Run] Would download {url} to {dest}")
            return True
            
        dest.parent.mkdir(parents=True, exist_ok=True)
        headers = {}
        mode = 'wb'
        initial_pos = 0
        if dest.exists():
            initial_pos = dest.stat().st_size
            headers['Range'] = f'bytes={initial_pos}-'
            mode = 'ab'
            
        try:
            response = self.session.get(url, headers=headers, stream=True)
        except Exception as e:
            logger.error(f"Failed to request {url}: {e}")
            return False

        if response.status_code == 416:
            logger.info(f"File {dest.name} already fully downloaded.")
            return True
        elif response.status_code not in (200, 206):
            logger.error(f"Failed to download {url}: Status {response.status_code}")
            return False
            
        total_size = int(response.headers.get('content-length', 0)) + initial_pos
        chunk_size = chunk_size_mb * 1024 * 1024
        
        with open(dest, mode) as f, tqdm(
            desc=dest.name,
            initial=initial_pos,
            total=total_size,
            unit='B',
            unit_scale=True,
            unit_divisor=1024,
        ) as bar:
            for chunk in response.iter_content(chunk_size=chunk_size):
                if chunk:
                    f.write(chunk)
                    bar.update(len(chunk))
        return True


class TCGAStadHandler(BaseDatasetHandler):
    """
    TCGA-STAD dataset handler.
    Sources:
      - SVS slides: GDC /files API
      - Demographics: cBioPortal stad_tcga
      - Molecular labels: cBioPortal stad_tcga_pub (Bass et al. 2014)
      - HER2: GDC BCR Biotab (fallback, usually absent)
    """
    def __init__(self, data_root: str, url: str, **kwargs):
        super().__init__(data_root, url, **kwargs)
        self.dataset_dir = self.data_root / "tcga-stad"
        self.slides_dir = self.dataset_dir / "slides"
        self.features_dir = self.dataset_dir / "features"
    
    def download(self, dry_run: bool = False, max_slides: int = None, **kwargs):
        logger.info("Starting TCGA-STAD download process...")
        self.dataset_dir.mkdir(parents=True, exist_ok=True)
        self.slides_dir.mkdir(exist_ok=True)
        self.features_dir.mkdir(exist_ok=True)

        # 1. Fetch Clinical Data
        logger.info("Fetching clinical data from cBioPortal...")
        if not dry_run:
            df_manifest = self._build_clinical_manifest()
            manifest_path = self.dataset_dir / "manifest.csv"
            df_manifest.to_csv(manifest_path, index=False)
            logger.info(f"Saved manifest to {manifest_path}")

            self._save_provenance(df_manifest)
        else:
            logger.info("[Dry Run] Would fetch clinical data and generate manifest.csv")

        # 2. Fetch SVS Files Metadata from GDC
        logger.info("Fetching SVS file metadata from GDC...")
        svs_files = self._get_gdc_svs_files(max_files=max_slides)
        
        if not svs_files:
            logger.warning("No SVS files found.")
            return

        logger.info(f"Found {len(svs_files)} SVS files to download.")
        
        # 3. Download SVS files
        gdc_data_endpt = "https://api.gdc.cancer.gov/data"
        for i, file_info in enumerate(svs_files):
            file_id = file_info['file_id']
            file_name = file_info['file_name']
            dest = self.slides_dir / file_name
            url = f"{gdc_data_endpt}/{file_id}"
            
            logger.info(f"Downloading {i+1}/{len(svs_files)}: {file_name}")
            self._download_file(url, dest, dry_run=dry_run)

    def _get_gdc_svs_files(self, max_files: int = None) -> List[Dict]:
        files_endpt = "https://api.gdc.cancer.gov/files"
        filters = {
            "op": "and",
            "content": [
                {"op": "in", "content": {"field": "cases.project.project_id", "value": ["TCGA-STAD"]}},
                {"op": "in", "content": {"field": "files.data_type", "value": ["Slide Image"]}},
                {"op": "in", "content": {"field": "files.data_format", "value": ["SVS"]}}
            ]
        }
        params = {
            "filters": json.dumps(filters),
            "format": "JSON",
            "size": str(max_files) if max_files else "10000",
            "fields": "file_id,file_name,cases.submitter_id"
        }
        response = self.session.get(files_endpt, params=params)
        if response.status_code == 200:
            return response.json().get('data', {}).get('hits', [])
        logger.error(f"GDC API error: {response.text}")
        return []

    def _fetch_cbio_clinical(self, study_id: str, data_type: str = "PATIENT") -> pd.DataFrame:
        url = f"https://www.cbioportal.org/api/studies/{study_id}/clinical-data?clinicalDataType={data_type}"
        response = self.session.get(url)
        if response.status_code != 200:
            logger.warning(f"Failed to fetch {study_id} from cBioPortal: {response.status_code}")
            return pd.DataFrame()
        
        data = response.json()
        if not data:
            return pd.DataFrame()
            
        df = pd.DataFrame(data)
        idx_col = "patientId" if data_type == "PATIENT" else "sampleId"
        df_pivot = df.pivot(index=idx_col, columns='clinicalAttributeId', values='value').reset_index()
        
        # If it's sample data, we typically want patientId as well. In TCGA, sampleId is roughly patientId + suffix
        if data_type == "SAMPLE" and "sampleId" in df_pivot.columns:
            # e.g., TCGA-BR-4187-01
            df_pivot['patientId'] = df_pivot['sampleId'].str.extract(r'(TCGA-[A-Z0-9]{2}-[A-Z0-9]{4})')[0]
        return df_pivot

    def _build_clinical_manifest(self) -> pd.DataFrame:
        # Fetch stad_tcga (Demographics)
        df_main = self._fetch_cbio_clinical("stad_tcga", "PATIENT")
        
        # Fetch stad_tcga_pub (Molecular) - patient level (for EBV, LAUREN)
        df_pub_pt = self._fetch_cbio_clinical("stad_tcga_pub", "PATIENT")
        
        # Fetch stad_tcga_pub (Molecular) - sample level (for MSI, MOLECULAR_SUBTYPE)
        df_pub_sm = self._fetch_cbio_clinical("stad_tcga_pub", "SAMPLE")
        
        if df_main.empty:
            logger.error("Failed to fetch main demographic data from cBioPortal.")
            df_main = pd.DataFrame(columns=['patientId'])
            
        if not df_pub_pt.empty:
            df_main = pd.merge(df_main, df_pub_pt, on='patientId', how='left', suffixes=('', '_pub'))
            
        if not df_pub_sm.empty:
            # We might have multiple samples per patient. Take the first non-null or drop duplicates
            df_pub_sm_dedup = df_pub_sm.drop_duplicates(subset=['patientId'])
            df_main = pd.merge(df_main, df_pub_sm_dedup, on='patientId', how='left', suffixes=('', '_sm'))

        # Standardize expected columns
        manifest = pd.DataFrame()
        manifest['patient_id'] = df_main['patientId'] if 'patientId' in df_main else []
        manifest['msi_status'] = df_main['MSI_STATUS'] if 'MSI_STATUS' in df_main.columns else 'unknown'
        manifest['ebv_status'] = df_main['EBV_PRESENT'] if 'EBV_PRESENT' in df_main.columns else 'unknown'
        manifest['lauren_class'] = df_main['LAUREN_CLASS'] if 'LAUREN_CLASS' in df_main.columns else 'unknown'
        manifest['molecular_subtype'] = df_main['MOLECULAR_SUBTYPE'] if 'MOLECULAR_SUBTYPE' in df_main.columns else 'unknown'
        
        # HER2 is not in these studies. Fallback to unknown.
        manifest['her2_status'] = 'unknown'

        # Fill NaNs with 'unknown'
        manifest.fillna('unknown', inplace=True)
        return manifest

    def _save_provenance(self, df: pd.DataFrame):
        prov = {
            "sources": {
                "svs_files": "GDC API (api.gdc.cancer.gov/files)",
                "demographics": "cBioPortal stad_tcga",
                "molecular_labels": "cBioPortal stad_tcga_pub (Bass et al. 2014)",
                "her2_status": "GDC BCR Biotab clinical supplement (fallback)"
            },
            "coverage": {
                "total_patients": len(df),
                "patients_with_msi_status": len(df[df['msi_status'] != 'unknown']),
                "patients_with_ebv_status": len(df[df['ebv_status'] != 'unknown']),
                "patients_with_lauren_class": len(df[df['lauren_class'] != 'unknown']),
                "patients_with_her2_status": len(df[df['her2_status'] != 'unknown'])
            },
            "null_handling": "Missing fields set to 'unknown' sentinel value"
        }
        with open(self.dataset_dir / "manifest_provenance.json", 'w') as f:
            json.dump(prov, f, indent=2)

    def preprocess(self):
        logger.info("TCGA-STAD preprocessing is handled during download/manifest creation.")

    def verify(self):
        logger.info("Verifying TCGA-STAD...")
        if not (self.dataset_dir / "manifest.csv").exists():
            logger.error("manifest.csv missing!")
        slides = list(self.slides_dir.glob("*.svs"))
        logger.info(f"Found {len(slides)} SVS files.")


class GasHisSDBHandler(BaseDatasetHandler):
    """
    GasHisSDB dataset handler.
    Downloads RAR, extracts, filters by patch size (160px), and creates splits.
    """
    def __init__(self, data_root: str, url: str, **kwargs):
        super().__init__(data_root, url, **kwargs)
        self.dataset_dir = self.data_root / "gashis"
        self.patch_sizes = self.kwargs.get("gashis_patch_sizes", [160])
        
    def download(self, dry_run: bool = False, **kwargs):
        self.dataset_dir.mkdir(parents=True, exist_ok=True)
        # Handle Figshare redirect
        archive_dest = self.dataset_dir / "GasHisSDB.rar"
        self._download_file(self.url, archive_dest, dry_run=dry_run)
        
    def preprocess(self):
        archive_dest = self.dataset_dir / "GasHisSDB.rar"
        if not archive_dest.exists():
            logger.error(f"Archive not found: {archive_dest}")
            return
            
        logger.info("Extracting GasHisSDB.rar...")
        try:
            import rarfile
            with rarfile.RarFile(archive_dest) as rf:
                rf.extractall(path=self.dataset_dir)
        except ImportError:
            logger.error("rarfile package not installed. Cannot extract .rar. Try 'pip install rarfile' and install unrar.")
            return
        except Exception as e:
            logger.error(f"Extraction failed: {e}")
            return

        self._create_splits()

    def _create_splits(self):
        logger.info("Creating Train/Val/Test splits for GasHisSDB...")
        train_ratio = self.kwargs.get("train_ratio", 0.70)
        val_ratio = self.kwargs.get("val_ratio", 0.15)
        
        splits = ['train', 'val', 'test']
        classes = ['Normal', 'Abnormal']
        
        for s in splits:
            for c in classes:
                (self.dataset_dir / s / c).mkdir(parents=True, exist_ok=True)
                
        # Find images matching patch sizes
        # The structure is usually GasHisSDB/160/Normal/...
        extracted_root = self.dataset_dir / "GasHisSDB"
        if not extracted_root.exists():
            # Maybe it extracted directly into dataset_dir
            extracted_root = self.dataset_dir
            
        for size in self.patch_sizes:
            size_dir = extracted_root / str(size)
            if not size_dir.exists():
                # Some versions name it "160x160"
                size_dir = extracted_root / f"{size}x{size}"
            if not size_dir.exists():
                logger.warning(f"Could not find directory for patch size {size}")
                continue
                
            for c in classes:
                class_dir = size_dir / c
                if not class_dir.exists():
                    class_dir = size_dir / c.lower() # fallback
                if not class_dir.exists():
                    continue
                    
                images = list(class_dir.glob("*.png"))
                random.shuffle(images)
                
                n_total = len(images)
                n_train = int(n_total * train_ratio)
                n_val = int(n_total * val_ratio)
                
                splits_dict = {
                    'train': images[:n_train],
                    'val': images[n_train:n_train+n_val],
                    'test': images[n_train+n_val:]
                }
                
                for split_name, imgs in splits_dict.items():
                    dest_dir = self.dataset_dir / split_name / c
                    for img in imgs:
                        # Copy to avoid destructive moves during dev
                        shutil.copy2(img, dest_dir / img.name)
                        
        logger.info("GasHisSDB splits created.")
        self._write_provenance()

    def _write_provenance(self):
        prov = {
            "source": self.url,
            "patch_sizes_kept": self.patch_sizes,
            "splits": {"train": self.kwargs.get("train_ratio", 0.70), "val": self.kwargs.get("val_ratio", 0.15), "test": self.kwargs.get("test_ratio", 0.15)}
        }
        with open(self.dataset_dir / "provenance.json", 'w') as f:
            json.dump(prov, f, indent=2)

    def verify(self):
        pass


class GCHTIDHandler(BaseDatasetHandler):
    """
    GCHTID dataset handler.
    Downloads ZIP, extracts, maps 9 classes to 7, generates pseudo-masks.
    """
    def __init__(self, data_root: str, url: str, **kwargs):
        super().__init__(data_root, url, **kwargs)
        self.dataset_dir = self.data_root / "gchtid"
        
        # 9 -> 7 class mapping
        self.class_map = {
            "TUM": 1, "STR": 2, "LYM": 3, "DEB": 4, "NORM": 5, "MUS": 6,
            "BACK": 0, "ADI": 0, "MUC": 0
        }
        
    def download(self, dry_run: bool = False, **kwargs):
        self.dataset_dir.mkdir(parents=True, exist_ok=True)
        archive_dest = self.dataset_dir / "GCHTID.zip"
        self._download_file(self.url, archive_dest, dry_run=dry_run)
        
    def preprocess(self):
        archive_dest = self.dataset_dir / "GCHTID.zip"
        if not archive_dest.exists():
            logger.error(f"Archive not found: {archive_dest}")
            return
            
        extract_dir = self.dataset_dir / "extracted"
        logger.info("Extracting GCHTID.zip...")
        try:
            with zipfile.ZipFile(archive_dest, 'r') as zf:
                zf.extractall(path=extract_dir)
        except Exception as e:
            logger.error(f"Extraction failed: {e}")
            return
            
        self._create_splits_and_masks(extract_dir)
        
    def _create_splits_and_masks(self, extract_dir: Path):
        logger.info("Creating pseudo-masks and splits for GCHTID...")
        train_ratio = self.kwargs.get("train_ratio", 0.70)
        val_ratio = self.kwargs.get("val_ratio", 0.15)
        
        splits = ['train', 'val', 'test']
        for s in splits:
            (self.dataset_dir / s / "images").mkdir(parents=True, exist_ok=True)
            (self.dataset_dir / s / "masks").mkdir(parents=True, exist_ok=True)
            
        # Collect all images and their labels
        dataset_items = []
        
        # The zip usually extracts into a folder structure with the class names
        # Let's search for the class directories
        for class_name, mask_val in self.class_map.items():
            # Find any directory matching the class name
            class_dirs = list(extract_dir.rglob(class_name))
            for c_dir in class_dirs:
                if not c_dir.is_dir():
                    continue
                for img_path in c_dir.glob("*.png"):
                    dataset_items.append((img_path, mask_val, class_name))
                    
        if not dataset_items:
            logger.warning("No images found in extracted GCHTID directory.")
            return
            
        random.shuffle(dataset_items)
        n_total = len(dataset_items)
        n_train = int(n_total * train_ratio)
        n_val = int(n_total * val_ratio)
        
        splits_dict = {
            'train': dataset_items[:n_train],
            'val': dataset_items[n_train:n_train+n_val],
            'test': dataset_items[n_train+n_val:]
        }
        
        for split_name, items in splits_dict.items():
            img_dir = self.dataset_dir / split_name / "images"
            mask_dir = self.dataset_dir / split_name / "masks"
            
            for i, (img_path, mask_val, original_class) in enumerate(tqdm(items, desc=f"Processing {split_name}")):
                new_stem = f"{original_class}_{i:05d}"
                new_img_path = img_dir / f"{new_stem}.png"
                new_mask_path = mask_dir / f"{new_stem}.png"
                
                # Copy image
                shutil.copy2(img_path, new_img_path)
                
                # Create pseudo-mask (single value per 224x224 tile)
                # assuming 224x224 based on dataset specs
                with Image.open(img_path) as img:
                    w, h = img.size
                
                mask_arr = np.full((h, w), mask_val, dtype=np.uint8)
                mask_img = Image.fromarray(mask_arr, mode='L')
                mask_img.save(new_mask_path)
                
        self._write_provenance()
        
    def _write_provenance(self):
        prov = {
            "source": "figshare:25954813",
            "source_url": self.url,
            "label_type": "weak_proxy",
            "label_provenance": "Cross-domain transfer from NCT-CRC-HE-100K colorectal cancer annotations. Pathologist validation at whole-slide level, not per-patch.",
            "class_mapping": {
                "TUM": {"id": 1, "mapped_name": "tumour"},
                "STR": {"id": 2, "mapped_name": "stroma"},
                "LYM": {"id": 3, "mapped_name": "til"},
                "DEB": {"id": 4, "mapped_name": "necrosis"},
                "NORM": {"id": 5, "mapped_name": "normal_gland"},
                "MUS": {"id": 6, "mapped_name": "muscle"},
                "BACK": {"id": 0, "mapped_name": "background"},
                "ADI": {"id": 0, "mapped_name": "background"},
                "MUC": {"id": 0, "mapped_name": "background"}
            },
            "warnings": [
                "Labels carry cross-domain model noise from CRC-to-gastric transfer learning",
                "Pseudo-masks are patch-uniform (single class per 224x224 tile), not pixel-precise boundaries",
                "Downstream model should be treated as tile-level tissue-composition estimator"
            ],
            "prepared_by": "hakim_ai data_manager"
        }
        with open(self.dataset_dir / "provenance.json", 'w') as f:
            json.dump(prov, f, indent=2)

    def verify(self):
        pass


class DatasetRegistry:
    """Registry to map URLs/identifiers to Handlers."""
    
    REGISTRY = {
        "tcga-stad": TCGAStadHandler,
        "cancerimagingarchive.net/collection/tcga-stad": TCGAStadHandler,
        "cancerimagingarchive.net/wp-content/uploads/TCIA_TCGA-STAD": TCGAStadHandler,
        
        "gashis": GasHisSDBHandler,
        "figshare.com/articles/dataset/GasHisSDB/15066147": GasHisSDBHandler,
        "figshare.com/ndownloader/files/28969725": GasHisSDBHandler,
        
        "gchtid": GCHTIDHandler,
        "figshare.com/articles/dataset/Gastric_Cancer_Histopathology_Tissue_Image_Dataset_GCHTID_/25954813": GCHTIDHandler,
        "figshare.com/ndownloader/articles/25954813": GCHTIDHandler,
    }
    
    @classmethod
    def get_handler(cls, identifier: str, data_root: str, url: str = None, **kwargs) -> Optional[BaseDatasetHandler]:
        # Simple match
        handler_class = cls.REGISTRY.get(identifier.lower())
        
        if not handler_class:
            # URL substring match
            for key, h_cls in cls.REGISTRY.items():
                if key in identifier:
                    handler_class = h_cls
                    break
                    
        if not handler_class and url:
            for key, h_cls in cls.REGISTRY.items():
                if key in url:
                    handler_class = h_cls
                    break

        if handler_class:
            return handler_class(data_root, url or identifier, **kwargs)
        return None
