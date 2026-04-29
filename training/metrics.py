"""
Evaluation metrics: accuracy, precision, recall, F1, and confusion matrices.
"""

import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, classification_report
)


def compute_metrics(all_preds, all_labels):
    """
    Compute per-task metrics and average F1.
    
    Args:
        all_preds: dict {task_name: list of predicted labels}
        all_labels: dict {task_name: list of true labels}
    Returns:
        dict with per-task and averaged metrics
    """
    results = {}
    f1_scores = []
    
    for task in ['sentiment', 'humor', 'sarcasm']:
        preds = all_preds[task]
        labels = all_labels[task]
        
        acc = accuracy_score(labels, preds)
        prec = precision_score(labels, preds, average='macro', zero_division=0)
        rec = recall_score(labels, preds, average='macro', zero_division=0)
        f1 = f1_score(labels, preds, average='macro', zero_division=0)
        cm = confusion_matrix(labels, preds)
        
        results[task] = {
            'accuracy': acc,
            'precision': prec,
            'recall': rec,
            'f1': f1,
            'confusion_matrix': cm,
        }
        f1_scores.append(f1)
    
    results['avg_f1'] = np.mean(f1_scores)
    return results


def print_metrics(metrics, prefix=""):
    """Print formatted metrics summary."""
    for task in ['sentiment', 'humor', 'sarcasm']:
        if task in metrics:
            m = metrics[task]
            print(f"{prefix} {task:>10s}: Acc={m['accuracy']:.3f} P={m['precision']:.3f} "
                  f"R={m['recall']:.3f} F1={m['f1']:.3f}")


def get_classification_report(preds, labels, label_names):
    """Get sklearn classification report as string."""
    return classification_report(labels, preds, target_names=label_names, zero_division=0)
