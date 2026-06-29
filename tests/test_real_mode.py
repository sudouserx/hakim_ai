import os
import pytest
from pathlib import Path

from hakim_ai.config import PipelineConfig
from hakim_ai.pipeline import HistopathologyPipeline
from hakim_ai.types import PipelineInput, WSIInput

# Define test SVS file
TEST_SVS_URL = "https://openslide.cs.cmu.edu/download/openslide-testdata/Aperio/CMU-1-Small-Region.svs"
TEST_SVS_PATH = Path(__file__).parent / "CMU-1-Small-Region.svs"

@pytest.fixture(scope="session")
def svs_file():
    """Download a small representative SVS file if not exists."""
    if not TEST_SVS_PATH.exists():
        import urllib.request
        print(f"Downloading {TEST_SVS_URL}...")
        urllib.request.urlretrieve(TEST_SVS_URL, TEST_SVS_PATH)
    return str(TEST_SVS_PATH)

def test_pipeline_real_mode_end_to_end(svs_file):
    """
    Test the pipeline in real mode.
    Note: Requires a machine with sufficient RAM/VRAM and HF_TOKEN set,
    otherwise models might fail to load.
    """
    # Force real mode
    cfg = PipelineConfig.default()
    cfg.mock_mode = False
    cfg.foundation_models.mock_mode = False
    
    # Enable CPU mode if no GPU, but typically we want GPU
    try:
        import torch
        cfg.foundation_models.use_gpu = torch.cuda.is_available()
        pipeline = HistopathologyPipeline(cfg)
    except ImportError as e:
        pytest.skip(f"Skipping real mode test due to missing dependencies: {e}")
        return
        
    inp = PipelineInput(
        wsi_input=WSIInput(
            wsi_path=svs_file,
            patient_id="TEST-REAL-001"
        )
    )

    result = pipeline.run(inp)
    
    assert result.is_successful(), f"Pipeline failed: {result.error}"
    assert result.report is not None
    assert "TEST-REAL-001" in result.report.patient_id
