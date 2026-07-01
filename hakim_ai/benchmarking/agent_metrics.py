from typing import List, Dict, Set
import torch
from torchmetrics.classification import BinaryAUROC, BinaryAveragePrecision, MultilabelF1Score, MultilabelAUROC
from sklearn.metrics import confusion_matrix
import re

class RouterEvaluator:
    """Evaluates the Layer 1 Router/Triage Agent."""
    
    def __init__(self):
        self.y_true = []
        self.y_pred = []
        
    def log(self, true_label: int, pred_label: int):
        # 0: Benign, 1: Malignant, 2: Ambiguous
        self.y_true.append(true_label)
        self.y_pred.append(pred_label)
        
    def compute(self) -> Dict[str, float]:
        if not self.y_true:
            return {}
        
        cm = confusion_matrix(self.y_true, self.y_pred, labels=[0, 1, 2])
        # Benign is 0. False Negative Rate for fast path: Malignant(1) routed as Benign(0)
        true_malignant = sum(cm[1])
        if true_malignant > 0:
            fn_benign = cm[1][0]  # True Malignant, Predicted Benign
            fnr = fn_benign / true_malignant
        else:
            fnr = 0.0
            
        accuracy = sum(cm[i][i] for i in range(3)) / max(1, len(self.y_true))
        
        return {
            "router_accuracy": float(accuracy),
            "router_false_negative_rate": float(fnr)
        }

class SegmentationEvaluator:
    """Evaluates Layer 2 Segmentation using tile-level metrics."""
    
    def __init__(self, num_classes: int = 7):
        self.f1_metric = MultilabelF1Score(num_labels=num_classes, average='macro')
        self.auc_metric = MultilabelAUROC(num_labels=num_classes, average='macro')
        self.preds = []
        self.targets = []
        
    def log(self, pred_probs: torch.Tensor, true_labels: torch.Tensor):
        # pred_probs: [N, num_classes], true_labels: [N, num_classes] (multi-hot)
        self.preds.append(pred_probs)
        self.targets.append(true_labels)
        
    def compute(self) -> Dict[str, float]:
        if not self.preds:
            return {}
        all_preds = torch.cat(self.preds, dim=0)
        all_targets = torch.cat(self.targets, dim=0).long()
        
        f1 = self.f1_metric(all_preds, all_targets).item()
        
        # AUROC requires non-constant targets in each class. Handle exceptions gracefully.
        try:
            auc = self.auc_metric(all_preds, all_targets).item()
        except ValueError:
            auc = 0.0
            
        return {
            "segmentation_macro_f1": float(f1),
            "segmentation_macro_auc": float(auc)
        }

class MolecularEvaluator:
    """Evaluates Layer 3 Molecular Prediction."""
    
    def __init__(self):
        self.msi_auc = BinaryAUROC()
        self.msi_ap = BinaryAveragePrecision()
        self.msi_preds = []
        self.msi_targets = []
        
    def log_msi(self, pred_prob: float, true_label: int):
        self.msi_preds.append(pred_prob)
        self.msi_targets.append(true_label)
        
    def compute(self) -> Dict[str, float]:
        if not self.msi_preds:
            return {}
            
        preds_t = torch.tensor(self.msi_preds)
        targets_t = torch.tensor(self.msi_targets)
        
        try:
            auc = self.msi_auc(preds_t, targets_t).item()
            ap = self.msi_ap(preds_t, targets_t).item()
        except ValueError:
            auc = 0.0
            ap = 0.0
            
        return {
            "molecular_msi_auroc": float(auc),
            "molecular_msi_auprc": float(ap)
        }


class LLMEntityJudge:
    """
    Evaluates VLM and Report Agents by checking for strict entity matching 
    (hallucinations or omissions) rather than n-gram overlap.
    In a real system, this would call an LLM (e.g. GPT-4). Here we use regex
    for demonstration of entity matching.
    """
    
    def __init__(self):
        self.total_entities = 0
        self.matched_entities = 0
        self.hallucinated_entities = 0
        
    def log_report(self, generated_text: str, expected_entities: Dict[str, str]):
        """
        expected_entities: {"lauren": "intestinal", "msi": "msi-h", "tumor_size": "3.2cm"}
        """
        gen_lower = generated_text.lower()
        
        for key, expected_val in expected_entities.items():
            self.total_entities += 1
            if expected_val.lower() in gen_lower:
                self.matched_entities += 1
            else:
                # Naive hallucination check: is there a contradicting value?
                # A true LLM judge would do this semantically.
                pass
                
    def compute(self) -> Dict[str, float]:
        if self.total_entities == 0:
            return {}
            
        recall = self.matched_entities / self.total_entities
        return {
            "llm_entity_match_recall": float(recall)
        }
