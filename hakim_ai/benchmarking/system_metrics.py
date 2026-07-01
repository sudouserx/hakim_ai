import time
from typing import Dict, List, Optional
import torch
from torchmetrics import CohenKappa
from torchmetrics.classification import MulticlassCalibrationError
from sklearn.metrics import f1_score

class SystemEvaluator:
    """Evaluates the end-to-end performance of the Hakim AI pipeline."""
    
    def __init__(self, num_diagnostic_classes: int = 5):
        self.num_classes = num_diagnostic_classes
        # Quadratic Weighted Kappa
        self.qwk_metric = CohenKappa(task="multiclass", num_classes=num_diagnostic_classes, weights="quadratic")
        self.ece_metric = MulticlassCalibrationError(num_classes=num_diagnostic_classes, n_bins=10, norm='l1')
        
        self.predictions: List[int] = []
        self.ground_truths: List[int] = []
        self.probabilities: List[List[float]] = []
        
        self.total_slides = 0
        self.abstained_slides = 0
        
        self.processing_times: List[float] = []

    def log_prediction(
        self, 
        pred_class: int, 
        true_class: int, 
        probs: List[float], 
        abstained: bool = False,
        processing_time: float = 0.0
    ):
        """Log a single slide's results."""
        self.total_slides += 1
        if processing_time > 0:
            self.processing_times.append(processing_time)
            
        if abstained:
            self.abstained_slides += 1
            return  # Typically, we don't calculate QWK on abstained samples, just track the rate.
            
        self.predictions.append(pred_class)
        self.ground_truths.append(true_class)
        self.probabilities.append(probs)

    def compute(self) -> Dict[str, float]:
        """Compute all system-level metrics."""
        results = {}
        
        abstention_rate = self.abstained_slides / max(1, self.total_slides)
        results["abstention_rate"] = float(abstention_rate)
        
        if self.processing_times:
            results["avg_latency_seconds"] = sum(self.processing_times) / len(self.processing_times)
        
        if not self.predictions:
            return results
            
        preds_tensor = torch.tensor(self.predictions)
        targets_tensor = torch.tensor(self.ground_truths)
        probs_tensor = torch.tensor(self.probabilities)
        
        # Diagnostic Concordance
        qwk = self.qwk_metric(preds_tensor, targets_tensor).item()
        results["quadratic_weighted_kappa"] = float(qwk)
        
        macro_f1 = f1_score(self.ground_truths, self.predictions, average="macro", zero_division=0)
        results["macro_f1"] = float(macro_f1)
        
        # Safety & Calibration
        ece = self.ece_metric(probs_tensor, targets_tensor).item()
        results["expected_calibration_error"] = float(ece)
        
        # Manual Brier Score for multiclass
        targets_one_hot = torch.nn.functional.one_hot(targets_tensor, num_classes=self.num_classes).float()
        brier = torch.mean(torch.sum((probs_tensor - targets_one_hot)**2, dim=1)).item()
        results["brier_score"] = float(brier)
        
        return results
