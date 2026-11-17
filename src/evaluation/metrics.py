"""
Classification Metrics Module for NLP Model Evaluation.

Computes exact multi-class confusion matrices, precision, recall, F1 scores,
macro averages, and support-weighted aggregates with zero division errors
and explicit handling of missing or abstained predictions.
"""

from collections import Counter, defaultdict
from typing import Any, Dict, List, Optional, Set, Tuple


def compute_confusion_matrix(
    y_true: List[str],
    y_pred: List[str],
    labels: List[str],
) -> Dict[str, Any]:
    """
    Construct a multi-class confusion matrix where rows represent ground-truth labels
    and columns represent model-predicted labels.

    Args:
        y_true: Ground-truth class labels.
        y_pred: Predicted class labels.
        labels: Exhaustive list of target class labels in desired order.

    Returns:
        Dictionary containing:
        - 'matrix': nested dictionary {true_class: {pred_class: count}}
        - 'labels': ordered list of class labels
        - 'grid': 2D list of counts [[count_00, count_01, ...], ...]
    """
    label_set = set(labels)
    matrix_dict: Dict[str, Dict[str, int]] = {
        true_lbl: {pred_lbl: 0 for pred_lbl in labels}
        for true_lbl in labels
    }

    # Also track predictions outside defined labels (e.g., abstained, unknown, missing)
    unmatched_predictions: Counter = Counter()
    unmatched_ground_truth: Counter = Counter()

    for true_val, pred_val in zip(y_true, y_pred):
        t_clean = str(true_val or "").strip().lower()
        p_clean = str(pred_val or "").strip().lower()

        if t_clean in label_set:
            if p_clean in label_set:
                matrix_dict[t_clean][p_clean] += 1
            else:
                unmatched_predictions[p_clean or "missing"] += 1
        else:
            unmatched_ground_truth[t_clean or "missing"] += 1

    grid = [
        [matrix_dict[t][p] for p in labels]
        for t in labels
    ]

    return {
        "labels": list(labels),
        "matrix": matrix_dict,
        "grid": grid,
        "unmatched_predictions": dict(unmatched_predictions),
        "unmatched_ground_truth": dict(unmatched_ground_truth),
    }


def compute_classification_metrics(
    y_true: List[str],
    y_pred: List[str],
    labels: List[str],
) -> Dict[str, Any]:
    """
    Calculate precision, recall, F1, accuracy, and macro/weighted summaries.

    Args:
        y_true: Ground-truth class labels.
        y_pred: Predicted class labels.
        labels: Exhaustive target class labels.

    Returns:
        Structured dictionary of per-class and aggregated classification metrics.
    """
    total_samples = len(y_true)
    if total_samples != len(y_pred):
        raise ValueError(f"Length mismatch: y_true ({total_samples}) vs y_pred ({len(y_pred)})")

    label_set = set(labels)
    cm = compute_confusion_matrix(y_true, y_pred, labels)
    mat = cm["matrix"]

    per_class_metrics: Dict[str, Dict[str, Any]] = {}
    total_tp = 0
    total_supported_samples = 0

    for c in labels:
        tp = mat[c].get(c, 0)
        total_tp += tp

        # False Negatives: True is c, but model predicted something else (or abstained)
        fn = sum(mat[c][p] for p in labels if p != c)

        # Add unmatched predictions for true class c to FN
        for true_val, pred_val in zip(y_true, y_pred):
            t_clean = str(true_val or "").strip().lower()
            p_clean = str(pred_val or "").strip().lower()
            if t_clean == c and p_clean not in label_set:
                fn += 1

        # False Positives: Model predicted c, but True was something else
        fp = sum(mat[t][c] for t in labels if t != c)

        support = tp + fn
        total_supported_samples += support

        precision = round(tp / (tp + fp), 4) if (tp + fp) > 0 else 0.0
        recall = round(tp / (tp + fn), 4) if (tp + fn) > 0 else 0.0
        f1 = round((2 * precision * recall) / (precision + recall), 4) if (precision + recall) > 0 else 0.0

        per_class_metrics[c] = {
            "true_positives": tp,
            "false_positives": fp,
            "false_negatives": fn,
            "support": support,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
        }

    # Macro Averages: unweighted average across classes with positive support
    supported_classes = [c for c in labels if per_class_metrics[c]["support"] > 0]
    num_supported = len(supported_classes)

    if num_supported > 0:
        macro_precision = round(sum(per_class_metrics[c]["precision"] for c in supported_classes) / num_supported, 4)
        macro_recall = round(sum(per_class_metrics[c]["recall"] for c in supported_classes) / num_supported, 4)
        macro_f1 = round(sum(per_class_metrics[c]["f1_score"] for c in supported_classes) / num_supported, 4)
    else:
        macro_precision = 0.0
        macro_recall = 0.0
        macro_f1 = 0.0

    # All-classes Macro F1 (including zero-support classes, where F1=0.0)
    all_classes_macro_f1 = round(sum(per_class_metrics[c]["f1_score"] for c in labels) / len(labels), 4) if labels else 0.0

    # Weighted Averages: weighted by class support
    if total_supported_samples > 0:
        weighted_precision = round(
            sum(per_class_metrics[c]["precision"] * per_class_metrics[c]["support"] for c in supported_classes) / total_supported_samples,
            4,
        )
        weighted_recall = round(
            sum(per_class_metrics[c]["recall"] * per_class_metrics[c]["support"] for c in supported_classes) / total_supported_samples,
            4,
        )
        weighted_f1 = round(
            sum(per_class_metrics[c]["f1_score"] * per_class_metrics[c]["support"] for c in supported_classes) / total_supported_samples,
            4,
        )
    else:
        weighted_precision = 0.0
        weighted_recall = 0.0
        weighted_f1 = 0.0

    overall_accuracy = round(total_tp / total_supported_samples, 4) if total_supported_samples > 0 else 0.0
    sample_accuracy = round(total_tp / total_samples, 4) if total_samples > 0 else 0.0

    return {
        "overall_accuracy": overall_accuracy,
        "sample_accuracy": sample_accuracy,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f1": macro_f1,
        "macro_f1_supported_classes": macro_f1,
        "macro_f1_all_classes": all_classes_macro_f1,
        "all_classes_macro_f1": all_classes_macro_f1,
        "weighted_precision": weighted_precision,
        "weighted_recall": weighted_recall,
        "weighted_f1": weighted_f1,
        "total_samples": total_samples,
        "total_supported_samples": total_supported_samples,
        "num_supported_classes": num_supported,
        "num_taxonomy_classes": len(labels),
        "per_class": per_class_metrics,
        "confusion_matrix": cm,
    }
