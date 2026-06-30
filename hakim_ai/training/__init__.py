"""Training module for Hakim-AI pipeline."""
import os

if os.path.exists("/kaggle/tmp"):
    os.environ.setdefault("HF_HOME", "/kaggle/tmp/huggingface")
