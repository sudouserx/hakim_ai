"""Models module for Hakim-AI."""
import os

if os.path.exists("/kaggle/tmp"):
    os.environ.setdefault("HF_HOME", "/kaggle/tmp/huggingface")
