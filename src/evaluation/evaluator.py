"""
Domain Evaluators for Commodity NLP Pipelines.

Provides dedicated evaluation engines for news relevance assessment,
supply-chain event extraction, and financial sentiment analysis.
"""

from collections import Counter
from datetime import datetime, timezone
import json
import logging
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from src.features.builder import canonicalize_url, load_jsonl_records
from src.evaluation.config import (
    EvaluationConfig,
    VALID_COMMODITIES,
    VALID_RELEVANCE_CLASSES,
    VALID_SENTIMENT_LABELS,
    VALID_EVENT_TYPES,
    VALID_IMPACT_DIRECTIONS,
)
from src.evaluation.annotation import _generate_deterministic_id
from src.evaluation.data_quality import DataQualityValidator
from src.evaluation.metrics import (
    compute_classification_metrics,
    compute_confusion_matrix,
)

logger = logging.getLogger(__name__)


class RelevanceEvaluator:
    """Evaluates 4-way news relevance classification against human ground truth."""

    def __init__(self, config: Optional[EvaluationConfig] = None) -> None:
        self.config = config or EvaluationConfig()
        self.labels = [
            "directly_relevant",
            "potentially_relevant",
            "irrelevant",
            "uncertain",
        ]

    def evaluate(self, pairs: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Evaluate relevance predictions against ground-truth pairs.

        Args:
            pairs: List of dicts containing 'true_relevance' and 'predicted_relevance',
                   optionally with 'commodity'.

        Returns:
            Structured dictionary of relevance classification metrics and per-commodity breakdowns.
        """
        if not pairs:
            return {
                "status": "no_data",
                "sample_count": 0,
                "metrics": {},
                "commodity_breakdown": {},
            }

        y_true = [p.get("true_relevance", "") for p in pairs]
        y_pred = [p.get("predicted_relevance", "") for p in pairs]

        overall_metrics = compute_classification_metrics(y_true, y_pred, self.labels)

        # Per-commodity breakdown
        by_commodity: Dict[str, List[Dict[str, Any]]] = {}
        for p in pairs:
            comm = str(p.get("commodity") or "unknown").strip().lower()
            if comm not in by_commodity:
                by_commodity[comm] = []
            by_commodity[comm].append(p)

        commodity_breakdown: Dict[str, Any] = {}
        for comm, comm_pairs in by_commodity.items():
            c_true = [p.get("true_relevance", "") for p in comm_pairs]
            c_pred = [p.get("predicted_relevance", "") for p in comm_pairs]
            commodity_breakdown[comm] = compute_classification_metrics(c_true, c_pred, self.labels)

        return {
            "status": "evaluated",
            "sample_count": len(pairs),
            "metrics": overall_metrics,
            "commodity_breakdown": commodity_breakdown,
        }


class EventExtractionEvaluator:
    """
    Evaluates rule-based supply-chain event extraction.
    Strictly separates classification performance from extraction coverage.
    """

    def __init__(self, config: Optional[EvaluationConfig] = None) -> None:
        self.config = config or EvaluationConfig()
        self.labels = [
            "supply_disruption",
            "demand_change",
            "geopolitical_risk",
            "inventory_change",
            "policy_action",
            "production_change",
            "other",
        ]

    def evaluate(self, pairs: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Evaluate event extraction predictions against ground truth.

        Args:
            pairs: List of dicts containing 'true_event_type', 'predicted_event_type',
                   and optionally 'article_title', 'commodity'.

        Returns:
            Structured dictionary of classification metrics, extraction coverage,
            and false-positive / false-negative error cases.
        """
        if not pairs:
            return {
                "status": "no_data",
                "sample_count": 0,
                "classification_metrics": {},
                "coverage_analysis": {},
            }

        y_true = [p.get("true_event_type", "") for p in pairs]
        y_pred = [p.get("predicted_event_type", "") for p in pairs]

        # 1. Classification Metrics across all event classes (including 'other')
        classification_metrics = compute_classification_metrics(y_true, y_pred, self.labels)

        # 2. Binary Event Detection & Conditional Event-Type Classification
        missed_events: List[Dict[str, Any]] = []      # True != 'other', Pred == 'other' or missing (FN_det)
        false_alarms: List[Dict[str, Any]] = []       # True == 'other', Pred != 'other' (FP_det)
        type_confusions: List[Dict[str, Any]] = []    # True != 'other', Pred != 'other', True != Pred
        exact_type_matches: int = 0                  # True != 'other', Pred != 'other', True == Pred

        num_true_events = 0
        num_extracted_events = 0
        tp_detection = 0
        tn_detection = 0

        for idx, (t, p, pair) in enumerate(zip(y_true, y_pred, pairs)):
            t_clean = str(t or "").strip().lower()
            p_clean = str(p or "").strip().lower()

            has_true_event = t_clean in self.labels and t_clean != "other"
            has_pred_event = p_clean in self.labels and p_clean != "other"

            if has_true_event:
                num_true_events += 1
            if has_pred_event:
                num_extracted_events += 1

            title = pair.get("article_title") or pair.get("title") or ""
            comm = pair.get("commodity") or ""

            if has_true_event and has_pred_event:
                tp_detection += 1
                if t_clean == p_clean:
                    exact_type_matches += 1
                else:
                    type_confusions.append({
                        "sample_index": idx,
                        "commodity": comm,
                        "title": title[:100],
                        "true_event": t_clean,
                        "predicted_event": p_clean,
                    })
            elif has_true_event and not has_pred_event:
                missed_events.append({
                    "sample_index": idx,
                    "commodity": comm,
                    "title": title[:100],
                    "true_event": t_clean,
                    "predicted_event": p_clean or "missing",
                })
            elif not has_true_event and has_pred_event:
                false_alarms.append({
                    "sample_index": idx,
                    "commodity": comm,
                    "title": title[:100],
                    "true_event": t_clean or "other",
                    "predicted_event": p_clean,
                })
            else:
                tn_detection += 1

        total_detection_samples = len(pairs)
        detection_recall = round(tp_detection / num_true_events, 4) if num_true_events > 0 else 0.0
        detection_precision = round(tp_detection / num_extracted_events, 4) if num_extracted_events > 0 else 0.0
        detection_f1 = (
            round((2 * detection_precision * detection_recall) / (detection_precision + detection_recall), 4)
            if (detection_precision + detection_recall) > 0
            else 0.0
        )
        detection_accuracy = (
            round((tp_detection + tn_detection) / total_detection_samples, 4)
            if total_detection_samples > 0
            else 0.0
        )

        conditional_type_accuracy = (
            round(exact_type_matches / tp_detection, 4) if tp_detection > 0 else 0.0
        )

        coverage_analysis = {
            "total_samples": total_detection_samples,
            "true_event_count": num_true_events,
            "extracted_event_count": num_extracted_events,
            "correctly_detected_event_count": tp_detection,
            "event_detection_recall_rate": detection_recall,
            "event_detection_precision_rate": detection_precision,
            "event_detection_f1_score": detection_f1,
            "event_detection_accuracy": detection_accuracy,
            "missed_events_count": len(missed_events),
            "missed_events_samples": missed_events[:10],
            "false_alarms_count": len(false_alarms),
            "false_alarms_samples": false_alarms[:10],
            "type_confusions_count": len(type_confusions),
            "type_confusions_samples": type_confusions[:10],
            "conditional_event_type_classification": {
                "evaluated_instances_tp_det": tp_detection,
                "exact_type_matches": exact_type_matches,
                "type_confusions_count": len(type_confusions),
                "conditional_type_accuracy": conditional_type_accuracy,
            },
            "detection_confusion_matrix": {
                "true_positives_tp_det": tp_detection,
                "false_positives_fp_det_false_alarms": len(false_alarms),
                "false_negatives_fn_det_missed_events": len(missed_events),
                "true_negatives_tn_det_correct_rejections": tn_detection,
            },
        }

        return {
            "status": "evaluated",
            "sample_count": len(pairs),
            "classification_metrics": classification_metrics,
            "coverage_analysis": coverage_analysis,
        }


class SentimentEvaluator:
    """
    Evaluates financial sentiment analysis.
    Computes distribution statistics, and evaluates classification metrics
    ONLY when verified human-labeled sentiment examples exist.
    """

    def __init__(self, config: Optional[EvaluationConfig] = None) -> None:
        self.config = config or EvaluationConfig()
        self.labels = ["positive", "negative", "neutral"]

    def evaluate(
        self,
        prediction_records: List[Dict[str, Any]],
        pairs: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """
        Evaluate sentiment predictions.

        Args:
            prediction_records: Full list of records containing 'sentiment_label' and 'sentiment_score'.
            pairs: Optional list of dicts with 'true_sentiment' and 'predicted_sentiment'.

        Returns:
            Structured dictionary of distribution stats, confidence stats, and supervised metrics (if available).
        """
        # 1. Prediction distribution
        total_preds = len(prediction_records)
        label_counter: Counter = Counter()
        scores: List[float] = []

        for r in prediction_records:
            lbl = r.get("sentiment_label")
            if lbl:
                label_counter[str(lbl).strip().lower()] += 1
            else:
                label_counter["missing"] += 1

            score = r.get("sentiment_score")
            if score is not None:
                try:
                    scores.append(float(score))
                except (ValueError, TypeError):
                    pass

        # Confidence summary statistics
        if scores:
            scores.sort()
            n = len(scores)
            mean_score = sum(scores) / n
            median_score = scores[n // 2] if n % 2 != 0 else (scores[n // 2 - 1] + scores[n // 2]) / 2.0
            variance = sum((s - mean_score) ** 2 for s in scores) / n
            std_score = math.sqrt(variance)

            confidence_statistics = {
                "count": n,
                "mean": round(mean_score, 4),
                "median": round(median_score, 4),
                "std": round(std_score, 4),
                "min": round(min(scores), 4),
                "max": round(max(scores), 4),
            }
        else:
            confidence_statistics = {
                "count": 0,
                "mean": 0.0,
                "median": 0.0,
                "std": 0.0,
                "min": 0.0,
                "max": 0.0,
            }

        distribution_summary = {
            lbl: {
                "count": label_counter[lbl],
                "percentage": round((label_counter[lbl] / total_preds) * 100.0, 2) if total_preds > 0 else 0.0,
            }
            for lbl in ["positive", "negative", "neutral", "missing"]
        }

        # 2. Supervised classification evaluation (ONLY if human ground truth is provided)
        has_labeled_pairs = False
        if pairs:
            # Check if at least one record has a non-empty true_sentiment label
            valid_pairs = [
                p for p in pairs
                if str(p.get("true_sentiment") or "").strip().lower() in self.labels
            ]
            has_labeled_pairs = len(valid_pairs) > 0

        if not has_labeled_pairs:
            supervised_report = {
                "status": "unavailable_no_labeled_ground_truth",
                "message": (
                    "Supervised sentiment evaluation is unavailable because no human-annotated sentiment "
                    "labels were supplied. Showing empirical model prediction distributions and confidence statistics."
                ),
            }
        else:
            y_true = [str(p.get("true_sentiment", "")).strip().lower() for p in pairs]
            y_pred = [str(p.get("predicted_sentiment", "")).strip().lower() for p in pairs]
            supervised_metrics = compute_classification_metrics(y_true, y_pred, self.labels)
            supervised_report = {
                "status": "evaluated",
                "sample_count": len(pairs),
                "metrics": supervised_metrics,
            }

        return {
            "prediction_distribution": distribution_summary,
            "confidence_statistics": confidence_statistics,
            "supervised_evaluation": supervised_report,
        }


class NLPEvaluationPipeline:
    """
    End-to-end evaluation pipeline coordinating data quality validation,
    relevance assessment, supply-chain event extraction, and sentiment evaluation.
    """

    def __init__(self, config: Optional[EvaluationConfig] = None) -> None:
        self.config = config or EvaluationConfig()
        self.quality_validator = DataQualityValidator(self.config)
        self.relevance_evaluator = RelevanceEvaluator(self.config)
        self.event_evaluator = EventExtractionEvaluator(self.config)
        self.sentiment_evaluator = SentimentEvaluator(self.config)

    def run(
        self,
        feature_records: List[Dict[str, Any]],
        labeled_records: Optional[List[Dict[str, Any]]] = None,
        evaluation_cutoff: Optional[datetime] = None,
    ) -> Dict[str, Any]:
        """
        Execute comprehensive evaluation across data quality and classification tasks.

        Args:
            feature_records: Pipeline article feature records.
            labeled_records: Optional ground-truth records containing true_* fields.
            evaluation_cutoff: Optional timestamp cutoff for future date validation.

        Returns:
            Unified evaluation report dictionary.
        """
        # 1. Run Data Quality Validation on the feature records
        quality_report = self.quality_validator.validate(
            feature_records,
            evaluation_cutoff=evaluation_cutoff,
        )

        # 2. Check supervised evaluation availability
        has_labels = bool(labeled_records and len(labeled_records) > 0)

        if not has_labels:
            sentiment_summary = self.sentiment_evaluator.evaluate(feature_records, pairs=None)
            return {
                "evaluation_metadata": {
                    "evaluation_timestamp": datetime.now(timezone.utc).isoformat(),
                    "supervised_evaluation_available": False,
                    "reason": "No ground-truth labeled dataset provided.",
                },
                "data_quality": quality_report,
                "unsupervised_sentiment_distribution": sentiment_summary,
                "relevance_evaluation": {
                    "status": "unavailable_no_labeled_ground_truth",
                    "message": "Ground-truth relevance labels required for evaluation.",
                },
                "event_extraction_evaluation": {
                    "status": "unavailable_no_labeled_ground_truth",
                    "message": "Ground-truth event labels required for evaluation.",
                },
            }

        # 3. Match feature predictions with ground truth
        # Strict matching using (commodity, canonical_url) or deterministic article_id.
        # Fallback to positional/index matching is strictly forbidden to prevent cross-article contamination.
        pred_map: Dict[Tuple[str, str], Dict[str, Any]] = {}
        pred_id_map: Dict[str, Dict[str, Any]] = {}

        for r in feature_records:
            comm = str(r.get("commodity") or "").strip().lower()
            url = canonicalize_url(r.get("article_url") or r.get("url") or r.get("evidence_url"))
            if comm and url:
                pred_map[(comm, url)] = r
                gen_id = _generate_deterministic_id(comm, url)
                pred_id_map[gen_id] = r
            art_id = r.get("article_id")
            if art_id:
                pred_id_map[str(art_id).strip()] = r

        matched_relevance_pairs: List[Dict[str, Any]] = []
        matched_event_pairs: List[Dict[str, Any]] = []
        matched_sentiment_pairs: List[Dict[str, Any]] = []

        seen_gt_keys: Set[Tuple[str, str]] = set()
        duplicate_gt_annotations: List[Dict[str, Any]] = []
        matched_pred_keys: Set[Tuple[str, str]] = set()
        unmatched_ground_truth_count = 0

        for idx, lbl_rec in enumerate(labeled_records, start=1):
            comm = str(lbl_rec.get("commodity") or "").strip().lower()
            url = canonicalize_url(lbl_rec.get("article_url") or lbl_rec.get("url"))
            title = lbl_rec.get("article_title") or lbl_rec.get("title") or ""
            art_id = str(lbl_rec.get("article_id") or "").strip()

            # Duplicate ground-truth annotation detection
            if comm and url:
                gt_key = (comm, url)
                if gt_key in seen_gt_keys:
                    duplicate_gt_annotations.append({
                        "record_index": idx,
                        "commodity": comm,
                        "article_url": url,
                        "article_id": art_id,
                    })
                else:
                    seen_gt_keys.add(gt_key)

            # Match priority: (comm, url) -> article_id -> self-contained prediction fields in lbl_rec
            pred = None
            if comm and url and (comm, url) in pred_map:
                pred = pred_map[(comm, url)]
                matched_pred_keys.add((comm, url))
            elif art_id and art_id in pred_id_map:
                pred = pred_id_map[art_id]
                p_comm = str(pred.get("commodity") or "").strip().lower()
                p_url = canonicalize_url(pred.get("article_url") or pred.get("url") or pred.get("evidence_url"))
                if p_comm and p_url:
                    matched_pred_keys.add((p_comm, p_url))
            elif any(k in lbl_rec for k in ("relevance_class", "sentiment_label", "event_type")):
                pred = lbl_rec
            else:
                unmatched_ground_truth_count += 1

            if pred is not None:
                true_rel = str(lbl_rec.get("true_relevance_class") or lbl_rec.get("true_relevance") or "").strip().lower()
                true_ev = str(lbl_rec.get("true_event_type") or lbl_rec.get("true_event") or "").strip().lower()
                true_sent = str(lbl_rec.get("true_sentiment_label") or lbl_rec.get("true_sentiment") or "").strip().lower()

                if true_rel:
                    matched_relevance_pairs.append({
                        "commodity": comm,
                        "article_url": url,
                        "title": title,
                        "true_relevance": true_rel,
                        "predicted_relevance": str(pred.get("relevance_class") or "").strip().lower(),
                    })

                if true_ev:
                    matched_event_pairs.append({
                        "commodity": comm,
                        "article_url": url,
                        "title": title,
                        "true_event_type": true_ev,
                        "predicted_event_type": str(pred.get("event_type") or "other").strip().lower(),
                    })

                if true_sent:
                    matched_sentiment_pairs.append({
                        "commodity": comm,
                        "article_url": url,
                        "title": title,
                        "true_sentiment": true_sent,
                        "predicted_sentiment": str(pred.get("sentiment_label") or "").strip().lower(),
                    })

        num_labeled = len(labeled_records)
        unmatched_predictions_count = max(0, len(feature_records) - len(matched_pred_keys))

        # Check sample size adequacy with realistic statistical caveat
        sample_size_warning = None
        if num_labeled < self.config.min_labeled_sample_size:
            sample_size_warning = (
                f"Sample size (N={num_labeled}) is below recommended threshold of "
                f"{self.config.min_labeled_sample_size}. Note that while N >= 30 is a minimal sample heuristic, "
                f"statistical reliability depends on class support, balance, and representativeness across market regimes. "
                f"Wide confidence intervals should be expected."
            )

        evaluated_tasks = []
        if matched_relevance_pairs:
            evaluated_tasks.append("relevance")
        if matched_event_pairs:
            evaluated_tasks.append("event_extraction")
        if matched_sentiment_pairs:
            evaluated_tasks.append("sentiment")

        # Execute domain evaluations
        relevance_results = self.relevance_evaluator.evaluate(matched_relevance_pairs)
        event_results = self.event_evaluator.evaluate(matched_event_pairs)
        sentiment_results = self.sentiment_evaluator.evaluate(feature_records, pairs=matched_sentiment_pairs)

        return {
            "evaluation_metadata": {
                "evaluation_timestamp": datetime.now(timezone.utc).isoformat(),
                "supervised_evaluation_available": True,
                "total_labeled_records": num_labeled,
                "evaluated_tasks": evaluated_tasks,
                "unmatched_ground_truth_count": unmatched_ground_truth_count,
                "unmatched_predictions_count": unmatched_predictions_count,
                "duplicate_ground_truth_count": len(duplicate_gt_annotations),
                "duplicate_ground_truth_samples": duplicate_gt_annotations[:10],
                "sample_size_warning": sample_size_warning,
            },
            "data_quality": quality_report,
            "relevance_evaluation": relevance_results,
            "event_extraction_evaluation": event_results,
            "sentiment_evaluation": sentiment_results,
        }


def evaluate_nlp_pipeline(
    features_path: Path,
    ground_truth_path: Optional[Path] = None,
    output_report_path: Optional[Path] = None,
    evaluation_cutoff: Optional[datetime] = None,
) -> Dict[str, Any]:
    """
    Convenience function to run end-to-end evaluation and write report.
    """
    feature_records = load_jsonl_records(features_path)
    labeled_records = load_jsonl_records(ground_truth_path) if ground_truth_path else None

    pipeline = NLPEvaluationPipeline()
    report = pipeline.run(
        feature_records=feature_records,
        labeled_records=labeled_records,
        evaluation_cutoff=evaluation_cutoff,
    )

    if output_report_path:
        output_report_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_report_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        logger.info("Saved evaluation report to %s", output_report_path)

    return report
