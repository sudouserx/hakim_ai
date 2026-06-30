"""
Core logic for model threshold calibration.
Finds optimal probability thresholds to maximize Youden's J statistic
and computes expected NLL on a validation set.
"""

def calibrate_model_thresholds(val_labels, val_probs):
    """
    Real calibration logic using sklearn.
    """
    try:
        import numpy as np
        from sklearn.metrics import roc_curve, log_loss
    except ImportError:
        print("Scikit-learn is required for real calibration. Returning default thresholds.")
        return {"msi_threshold": 0.5, "ebv_threshold": 0.5, "msi_nll": 0.0, "ebv_nll": 0.0}
        
    results = {}
    
    def find_optimal_threshold(y_true, y_prob):
        if len(np.unique(y_true)) < 2:
            return 0.5
        fpr, tpr, thresholds = roc_curve(y_true, y_prob)
        youden_j = tpr - fpr
        best_idx = np.argmax(youden_j)
        return float(thresholds[best_idx])
    
    for task in val_labels.keys():
        y_true = np.array(val_labels[task])
        y_prob = np.array(val_probs[task])
        
        if len(y_prob.shape) == 1 or y_prob.shape[1] == 1:
            best_thresh = find_optimal_threshold(y_true, y_prob)
            nll = float(log_loss(y_true, y_prob, labels=[0, 1]))
            results[f"{task}_threshold"] = round(best_thresh, 4)
            results[f"{task}_nll"] = round(nll, 4)
            
    return results
