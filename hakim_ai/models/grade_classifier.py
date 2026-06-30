"""
Histological grading classification model for gastric adenocarcinoma.
Implements a Hybrid CNN-Transformer or a simple multi-layer perceptron over patch features.
"""
from __future__ import annotations

import os
import torch
import torch.nn as nn
from typing import Optional

from hakim_ai.utils.logging_utils import get_logger

logger = get_logger("models.grade_classifier")


class HistologicalGradeClassifier(nn.Module):
    """
    ML Model to predict histological grade (well, moderately, poorly differentiated).
    Takes a mean patch feature vector from the encoder as input.
    """
    def __init__(self, input_dim: int = 1536, num_classes: int = 3):
        super().__init__()
        self.input_dim = input_dim
        
        # Simple MLP for classification on top of rich ViT embeddings
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )
        
        self.class_names = [
            "well differentiated (Grade 1)",
            "moderately differentiated (Grade 2)",
            "poorly differentiated (Grade 3)"
        ]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(x)

    @classmethod
    def load_model(cls, checkpoint_path: str, input_dim: int = 1536, device: torch.device = torch.device("cpu")) -> Optional["HistologicalGradeClassifier"]:
        """Load trained model weights."""
        model = cls(input_dim=input_dim).to(device)
        if os.path.exists(checkpoint_path):
            try:
                state_dict = torch.load(checkpoint_path, map_location=device, weights_only=True)
                model.load_state_dict(state_dict)
                model.eval()
                return model
            except Exception as e:
                logger.error(f"Failed to load GradeClassifier weights: {e}")
                return None
        logger.warning(f"GradeClassifier checkpoint not found at {checkpoint_path}. Using uninitialized model.")
        model.eval()
        return model

    def predict_grade(self, feature_vector: list[float], device: torch.device) -> str:
        """Predict grade string from a single feature vector."""
        with torch.no_grad():
            x = torch.tensor(feature_vector, dtype=torch.float32).unsqueeze(0).to(device)
            logits = self.forward(x)
            pred_idx = torch.argmax(logits, dim=1).item()
            return self.class_names[pred_idx]
