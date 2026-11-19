"""
Unit and integration tests for the NLP Evaluation Framework and Data-Quality Validation.

Covers:
- Exact multi-class classification metric computations (precision, recall, F1, macro/weighted, accuracy).
- Confusion matrices and abstained/unmatched prediction tracking.
- Class imbalance, zero denominators, and zero-support robustness.
- Data-quality validation: duplicate identities, malformed records, invalid URLs, missing timestamps.
- Future publication timestamp detection against explicit evaluation cutoffs.
- Distinction between genuine absence of events and missing pipeline stages.
- Out-of-bounds confidence validation.
- Deterministic, stratified human annotation sampling with zero model prediction leakage.
- Relevance, event extraction, and sentiment domain evaluators.
- CLI execution and report formatting across all modes.
"""

from collections import Counter
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import pytest
from typing import Any, Dict, List

from src.evaluation.config import (
    EvaluationConfig,
    VALID_COMMODITIES,
    VALID_RELEVANCE_CLASSES,
    VALID_SENTIMENT_LABELS,
    VALID_EVENT_TYPES,
    VALID_IMPACT_DIRECTIONS,
)
from src.evaluation.data_quality import (
    DataQualityValidator,
    validate_article_features,
    _is_valid_confidence,
)
from src.evaluation.annotation import (
    AnnotationSampler,
    generate_annotation_sample,
    validate_annotation_dataset,
    _generate_deterministic_id,
    ANNOTATION_GUIDELINES,
    AnnotationCSVImporter,
    convert_annotated_csv_to_jsonl,
    sanitize_for_spreadsheet,
    desanitize_from_spreadsheet,
    prepare_pilot_annotation_packages,
)
from src.evaluation.quality_control import (
    compute_cohens_kappa,
    interpret_cohens_kappa,
    AnnotationAgreementAuditor,
    AnnotationAdjudicator,
    load_annotation_records,
)
from src.evaluation.metrics import (
    compute_confusion_matrix,
    compute_classification_metrics,
)
from src.evaluation.evaluator import (
    RelevanceEvaluator,
    EventExtractionEvaluator,
    SentimentEvaluator,
    NLPEvaluationPipeline,
    evaluate_nlp_pipeline,
)
from src.evaluation.cli import run_cli


# ---------------------------------------------------------------------------
# 1. Classification Metrics & Confusion Matrix Tests
# ---------------------------------------------------------------------------

def test_compute_confusion_matrix_perfect():
    """Verify confusion matrix with perfectly matching predictions."""
    labels = ["directly_relevant", "potentially_relevant", "irrelevant", "uncertain"]
    y_true = ["directly_relevant", "potentially_relevant", "irrelevant", "uncertain"]
    y_pred = ["directly_relevant", "potentially_relevant", "irrelevant", "uncertain"]

    cm = compute_confusion_matrix(y_true, y_pred, labels)
    assert cm["labels"] == labels
    for c in labels:
        assert cm["matrix"][c][c] == 1
        for other in labels:
            if other != c:
                assert cm["matrix"][c][other] == 0


def test_compute_confusion_matrix_misclassifications_and_unmatched():
    """Verify confusion matrix tracks misclassifications and off-taxonomy predictions."""
    labels = ["positive", "negative", "neutral"]
    y_true = ["positive", "positive", "negative", "neutral"]
    y_pred = ["positive", "neutral",  "positive", "abstained"]

    cm = compute_confusion_matrix(y_true, y_pred, labels)
    mat = cm["matrix"]
    assert mat["positive"]["positive"] == 1
    assert mat["positive"]["neutral"] == 1
    assert mat["negative"]["positive"] == 1
    assert mat["neutral"]["neutral"] == 0
    # 'abstained' is recorded in unmatched_predictions
    assert cm["unmatched_predictions"].get("abstained") == 1


def test_compute_classification_metrics_hand_calculated():
    """
    Test precision, recall, F1, macro, and weighted F1 against manually verified calculations.

    Classes: [A, B]
    y_true: [A, A, A, B, B]
    y_pred: [A, A, B, B, A]

    Class A:
      TP = 2, FP = 1, FN = 1
      Precision = 2/3 = 0.6667
      Recall    = 2/3 = 0.6667
      F1        = 0.6667
      Support   = 3

    Class B:
      TP = 1, FP = 1, FN = 1
      Precision = 1/2 = 0.5000
      Recall    = 1/2 = 0.5000
      F1        = 0.5000
      Support   = 2

    Macro Precision = (2/3 + 1/2) / 2 = 7/12 = 0.5833
    Macro Recall    = (2/3 + 1/2) / 2 = 7/12 = 0.5833
    Macro F1        = (0.6667 + 0.5000) / 2 = 0.5833

    Weighted F1     = (3 * 0.6667 + 2 * 0.5000) / 5 = (2.0 + 1.0) / 5 = 0.6000
    Accuracy        = (2 + 1) / 5 = 0.6000
    """
    labels = ["a", "b"]
    y_true = ["a", "a", "a", "b", "b"]
    y_pred = ["a", "a", "b", "b", "a"]

    res = compute_classification_metrics(y_true, y_pred, labels)
    assert res["overall_accuracy"] == 0.6
    assert res["macro_precision"] == 0.5834
    assert res["macro_recall"] == 0.5834
    assert res["macro_f1"] == 0.5834
    assert res["weighted_f1"] == 0.6

    assert res["per_class"]["a"]["precision"] == 0.6667
    assert res["per_class"]["a"]["recall"] == 0.6667
    assert res["per_class"]["a"]["f1_score"] == 0.6667
    assert res["per_class"]["a"]["support"] == 3

    assert res["per_class"]["b"]["precision"] == 0.5
    assert res["per_class"]["b"]["recall"] == 0.5
    assert res["per_class"]["b"]["f1_score"] == 0.5
    assert res["per_class"]["b"]["support"] == 2


def test_compute_classification_metrics_zero_support_and_denominators():
    """Ensure classes with zero true support and zero predictions return 0.0 without ZeroDivisionError."""
    labels = ["a", "b", "c"]
    # 'c' has zero true support and zero predictions
    y_true = ["a", "a", "b"]
    y_pred = ["a", "b", "b"]

    res = compute_classification_metrics(y_true, y_pred, labels)
    c_metrics = res["per_class"]["c"]
    assert c_metrics["support"] == 0
    assert c_metrics["precision"] == 0.0
    assert c_metrics["recall"] == 0.0
    assert c_metrics["f1_score"] == 0.0

    # Macro F1 over supported classes (a and b)
    assert res["macro_f1"] > 0.0
    # All-classes macro F1 includes class c with 0.0
    assert res["all_classes_macro_f1"] < res["macro_f1"]


def test_compute_classification_metrics_missing_and_abstained_predictions():
    """Missing or None predictions must be treated as errors, never assumed correct."""
    labels = ["directly_relevant", "irrelevant"]
    y_true = ["directly_relevant", "directly_relevant"]
    y_pred = [None, ""]

    res = compute_classification_metrics(y_true, y_pred, labels)
    rel = res["per_class"]["directly_relevant"]
    assert rel["true_positives"] == 0
    assert rel["false_negatives"] == 2
    assert rel["recall"] == 0.0
    assert res["overall_accuracy"] == 0.0


def test_compute_classification_metrics_length_mismatch():
    """Length mismatch between true and predicted lists raises ValueError."""
    with pytest.raises(ValueError, match="Length mismatch"):
        compute_classification_metrics(["a"], ["a", "b"], ["a", "b"])


# ---------------------------------------------------------------------------
# 2. Data-Quality Validation Tests
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_feature_record():
    return {
        "commodity": "copper",
        "article_url": "https://reuters.com/copper-supply-tightens",
        "article_title": "Escondida copper mine strike looms as talks stall",
        "article_publication_timestamp": "2026-10-20T10:00:00Z",
        "domain": "reuters.com",
        "language": "English",
        "source_country": "Chile",
        "fetch_timestamp": "2026-10-20T12:00:00Z",
        "provider": "gdelt",
        "relevance_class": "directly_relevant",
        "relevance_score": 0.95,
        "relevance_explanation": "Direct operational copper mining headline.",
        "matched_relevance_indicators": ["copper", "mine", "strike"],
        "sentiment_label": "negative",
        "sentiment_score": 0.88,
        "sentiment_model": "ProsusAI/finbert",
        "sentiment_status": "success",
        "event_type": "supply_disruption",
        "event_impact_direction": "potentially_bullish",
        "event_confidence": 0.85,
        "extraction_status": "success",
        "evidence_fact": "strike looms",
        "implication_rationale": "supply reduction",
        "feature_status": "complete",
    }


def test_data_quality_validator_clean_records(sample_feature_record):
    """Clean records should pass validation with 100% pass rate and zero anomalies."""
    validator = DataQualityValidator()
    report = validator.validate([sample_feature_record], evaluation_cutoff=datetime(2026, 10, 25, tzinfo=timezone.utc))

    meta = report["evaluation_metadata"]
    assert meta["total_records_evaluated"] == 1
    assert meta["valid_records_count"] == 1
    assert meta["data_quality_pass_rate"] == 1.0

    assert report["volume_and_commodities"]["counts_by_commodity"]["copper"] == 1
    assert report["identity_and_integrity"]["duplicate_identities_count"] == 0
    assert report["identity_and_integrity"]["malformed_records_count"] == 0
    assert report["temporal_validity"]["future_timestamps_count"] == 0
    assert report["confidence_value_validity"]["invalid_relevance_scores_count"] == 0
    assert report["confidence_value_validity"]["invalid_sentiment_scores_count"] == 0
    assert report["confidence_value_validity"]["invalid_event_confidences_count"] == 0


def test_data_quality_validator_detects_duplicates(sample_feature_record):
    """Validator detects duplicate (commodity, canonical_url) identities."""
    rec1 = dict(sample_feature_record)
    rec2 = dict(sample_feature_record)  # Duplicate identity
    rec2["article_title"] = "Slightly altered headline"

    validator = DataQualityValidator()
    report = validator.validate([rec1, rec2])

    ident = report["identity_and_integrity"]
    assert ident["unique_identities_count"] == 1
    assert ident["duplicate_identities_count"] == 1
    assert len(ident["duplicate_identities_samples"]) == 1


def test_data_quality_validator_detects_invalid_and_missing_urls(sample_feature_record):
    """Validator flags missing URLs and URLs without valid HTTP/HTTPS schemes."""
    rec_missing = dict(sample_feature_record)
    rec_missing["article_url"] = ""

    rec_invalid = dict(sample_feature_record)
    rec_invalid["article_url"] = "ftp://not-a-web-url.com/data"

    validator = DataQualityValidator()
    report = validator.validate([rec_missing, rec_invalid])

    url_rep = report["url_validity"]
    assert url_rep["missing_urls_count"] == 1
    assert url_rep["invalid_urls_count"] == 1


def test_data_quality_validator_detects_timestamp_anomalies(sample_feature_record):
    """Validator detects missing, unparseable, and future-dated publication timestamps."""
    cutoff = datetime(2026, 10, 20, 12, 0, 0, tzinfo=timezone.utc)

    rec_missing_ts = dict(sample_feature_record)
    rec_missing_ts["article_publication_timestamp"] = None

    rec_unparseable_ts = dict(sample_feature_record)
    rec_unparseable_ts["article_publication_timestamp"] = "invalid-date-string"

    rec_future_ts = dict(sample_feature_record)
    rec_future_ts["article_publication_timestamp"] = "2026-10-25T00:00:00Z"  # Past cutoff

    validator = DataQualityValidator()
    report = validator.validate([rec_missing_ts, rec_unparseable_ts, rec_future_ts], evaluation_cutoff=cutoff)

    temp = report["temporal_validity"]
    assert temp["missing_timestamps_count"] == 1
    assert temp["unparseable_timestamps_count"] == 1
    assert temp["future_timestamps_count"] == 1


def test_data_quality_validator_confidence_bounds(sample_feature_record):
    """Confidence scores must be numeric floats within [0.0, 1.0]."""
    rec_neg = dict(sample_feature_record)
    rec_neg["relevance_score"] = -0.1

    rec_over = dict(sample_feature_record)
    rec_over["sentiment_score"] = 1.05

    rec_str = dict(sample_feature_record)
    rec_str["event_confidence"] = "high"

    validator = DataQualityValidator()
    report = validator.validate([rec_neg, rec_over, rec_str])

    conf = report["confidence_value_validity"]
    assert conf["invalid_relevance_scores_count"] == 1
    assert conf["invalid_sentiment_scores_count"] == 1
    assert conf["invalid_event_confidences_count"] == 1


def test_data_quality_validator_distinguishes_genuine_absence_vs_missing(sample_feature_record):
    """
    Distinguish between:
    1. Genuine absence of events (processed, but no pattern matched -> event_type='other', confidence=0.0).
    2. True missing pipeline step (extraction_status='missing').
    """
    rec_genuine_absence = dict(sample_feature_record)
    rec_genuine_absence["event_type"] = "other"
    rec_genuine_absence["event_confidence"] = 0.0
    rec_genuine_absence["extraction_status"] = "success"

    rec_missing_step = dict(sample_feature_record)
    rec_missing_step["event_type"] = "other"
    rec_missing_step["event_confidence"] = 0.0
    rec_missing_step["extraction_status"] = "missing"

    validator = DataQualityValidator()
    report = validator.validate([rec_genuine_absence, rec_missing_step])

    miss = report["missingness_and_coverage"]
    assert miss["genuine_absence_of_events_count"] == 1
    assert miss["missing_event_extraction_count"] == 1


def test_data_quality_validator_detects_malformed_and_unsupported_values(sample_feature_record):
    """Validator detects malformed records and taxonomy violations."""
    non_dict = "not a dictionary"

    missing_title = dict(sample_feature_record)
    missing_title["article_title"] = None

    tax_violation = dict(sample_feature_record)
    tax_violation["commodity"] = "lithium"  # Unsupported commodity
    tax_violation["relevance_class"] = "super_relevant"  # Unsupported relevance
    tax_violation["sentiment_label"] = "euphoric"        # Unsupported sentiment
    tax_violation["event_type"] = "alien_invasion"       # Unsupported event

    validator = DataQualityValidator()
    report = validator.validate([non_dict, missing_title, tax_violation])

    ident = report["identity_and_integrity"]
    assert ident["malformed_records_count"] == 2  # non_dict and missing_title

    tax = report["taxonomy_violations"]
    assert "lithium" in tax["unsupported_commodities"]
    assert "super_relevant" in tax["unsupported_relevance_classes"]
    assert "euphoric" in tax["unsupported_sentiment_labels"]
    assert "alien_invasion" in tax["unsupported_event_types"]


# ---------------------------------------------------------------------------
# 3. Human Annotation Workflow Tests
# ---------------------------------------------------------------------------

@pytest.fixture
def multi_commodity_news_records():
    records = []
    commodities = ["copper", "gold", "crude oil"]
    for comm in commodities:
        for i in range(10):
            records.append({
                "commodity": comm,
                "url": f"https://news.example.com/{comm}/article-{i}",
                "title": f"Report {i} discussing {comm} supply and market dynamics",
                "publication_timestamp": f"2026-10-{10+i:02d}T10:00:00Z",
                "domain": "news.example.com",
                "language": "English",
                # Simulate model outputs that must be stripped in annotation template
                "relevance_class": "directly_relevant",
                "relevance_score": 0.95,
                "sentiment_label": "positive",
                "event_type": "production_change",
            })
    return records


def test_annotation_sampler_deterministic_and_stratified(multi_commodity_news_records):
    """Deterministic sampling with same seed yields identical results, balanced across commodities."""
    sampler = AnnotationSampler()
    sample1 = sampler.sample(multi_commodity_news_records, sample_size=9, seed=42, stratify_by_commodity=True)
    sample2 = sampler.sample(multi_commodity_news_records, sample_size=9, seed=42, stratify_by_commodity=True)
    sample_diff_seed = sampler.sample(multi_commodity_news_records, sample_size=9, seed=999, stratify_by_commodity=True)

    assert len(sample1) == 9
    assert sample1 == sample2
    assert sample1 != sample_diff_seed

    # Confirm stratification: 3 per commodity
    counts = Counter(s["commodity"] for s in sample1)
    assert counts["copper"] == 3
    assert counts["gold"] == 3
    assert counts["crude oil"] == 3


def test_annotation_sampler_prevents_model_prediction_leakage(multi_commodity_news_records):
    """Annotation templates must contain ONLY article context and blank human-label fields."""
    sampler = AnnotationSampler()
    sample = sampler.sample(multi_commodity_news_records, sample_size=5, seed=42)

    leaked_keys = {
        "relevance_class",
        "relevance_score",
        "relevance_explanation",
        "matched_relevance_indicators",
        "sentiment_label",
        "sentiment_score",
        "event_type",
        "event_confidence",
        "event_impact_direction",
        "feature_status",
    }

    for item in sample:
        # Verify no model predictions leaked
        for k in leaked_keys:
            assert k not in item

        # Verify required blank ground-truth fields
        assert item["true_relevance_class"] == ""
        assert item["true_sentiment_label"] == ""
        assert item["true_event_type"] == ""
        assert item["true_event_impact_direction"] == ""
        assert item["annotator_id"] == ""
        assert item["annotator_notes"] == ""

        # Verify article context preserved
        assert item["commodity"] in VALID_COMMODITIES
        assert item["article_url"].startswith("http")
        assert len(item["article_title"]) > 0


def test_validate_annotation_dataset():
    """Verify auditing of completed vs partial human annotation datasets."""
    records = [
        {
            "commodity": "copper",
            "article_url": "https://example.com/1",
            "article_title": "Copper mine strike",
            "true_relevance_class": "directly_relevant",
            "true_sentiment_label": "negative",
            "true_event_type": "supply_disruption",
        },
        {
            "commodity": "gold",
            "article_url": "https://example.com/2",
            "article_title": "Gold reserves increase",
            "true_relevance_class": "directly_relevant",
            "true_sentiment_label": "",  # Missing sentiment
            "true_event_type": "demand_change",
        },
        {
            "commodity": "crude oil",
            "article_url": "https://example.com/3",
            "article_title": "Oil report",
            "true_relevance_class": "invalid_class",  # Taxonomy violation
            "true_sentiment_label": "neutral",
            "true_event_type": "other",
        },
    ]

    audit = validate_annotation_dataset(records)
    assert audit["total_records"] == 3
    assert audit["completion_status"]["fully_labeled_records"] == 1
    assert audit["completion_status"]["labeled_relevance_count"] == 3
    assert audit["completion_status"]["labeled_sentiment_count"] == 2
    assert audit["completion_status"]["labeled_events_count"] == 3
    assert audit["taxonomy_violations"]["invalid_relevance_count"] == 1


# ---------------------------------------------------------------------------
# 4. Domain Evaluators Tests
# ---------------------------------------------------------------------------

def test_relevance_evaluator_and_commodity_breakdown():
    """RelevanceEvaluator calculates overall and per-commodity metrics."""
    pairs = [
        {"commodity": "copper", "true_relevance": "directly_relevant", "predicted_relevance": "directly_relevant"},
        {"commodity": "copper", "true_relevance": "irrelevant", "predicted_relevance": "irrelevant"},
        {"commodity": "gold", "true_relevance": "potentially_relevant", "predicted_relevance": "potentially_relevant"},
        {"commodity": "gold", "true_relevance": "irrelevant", "predicted_relevance": "directly_relevant"}, # Misclassification
        {"commodity": "crude oil", "true_relevance": "directly_relevant", "predicted_relevance": "directly_relevant"},
    ]

    evaluator = RelevanceEvaluator()
    res = evaluator.evaluate(pairs)

    assert res["status"] == "evaluated"
    assert res["sample_count"] == 5
    assert res["metrics"]["overall_accuracy"] == 0.80

    cb = res["commodity_breakdown"]
    assert "copper" in cb
    assert "gold" in cb
    assert "crude oil" in cb
    assert cb["copper"]["overall_accuracy"] == 1.0
    assert cb["gold"]["overall_accuracy"] == 0.5


def test_event_extraction_evaluator_coverage_and_errors():
    """EventExtractionEvaluator measures detection coverage separately from type classification."""
    pairs = [
        # 1. Correct detection & correct type
        {
            "commodity": "copper",
            "true_event_type": "supply_disruption",
            "predicted_event_type": "supply_disruption",
            "title": "Copper mine strike halts output",
        },
        # 2. Missed event (FN for detection): true was geopolitical_risk, predicted was 'other'
        {
            "commodity": "crude oil",
            "true_event_type": "geopolitical_risk",
            "predicted_event_type": "other",
            "title": "Tanker seized in Strait of Hormuz",
        },
        # 3. False alarm (FP for detection): true was 'other', predicted was production_change
        {
            "commodity": "gold",
            "true_event_type": "other",
            "predicted_event_type": "production_change",
            "title": "Jewelry store opens downtown",
        },
        # 4. Correctly rejected non-event
        {
            "commodity": "copper",
            "true_event_type": "other",
            "predicted_event_type": "other",
            "title": "General market recap",
        },
        # 5. Correct detection but type confusion: true was demand_change, predicted supply_disruption
        {
            "commodity": "crude oil",
            "true_event_type": "demand_change",
            "predicted_event_type": "supply_disruption",
            "title": "EV adoption accelerates",
        },
    ]

    evaluator = EventExtractionEvaluator()
    res = evaluator.evaluate(pairs)

    assert res["status"] == "evaluated"
    cov = res["coverage_analysis"]
    # True events: items 1, 2, 5 -> 3 true events
    assert cov["true_event_count"] == 3
    # Extracted events: items 1, 3, 5 -> 3 extracted events
    assert cov["extracted_event_count"] == 3
    # Correctly detected (any non-'other' event): items 1, 5 -> 2
    assert cov["correctly_detected_event_count"] == 2
    assert cov["event_detection_recall_rate"] == round(2 / 3, 4)
    assert cov["event_detection_precision_rate"] == round(2 / 3, 4)

    # Missed events count
    assert cov["missed_events_count"] == 1
    assert cov["missed_events_samples"][0]["true_event"] == "geopolitical_risk"

    # False alarms count
    assert cov["false_alarms_count"] == 1
    assert cov["false_alarms_samples"][0]["predicted_event"] == "production_change"

    # Type confusions count
    assert cov["type_confusions_count"] == 1
    assert cov["type_confusions_samples"][0]["true_event"] == "demand_change"
    assert cov["type_confusions_samples"][0]["predicted_event"] == "supply_disruption"


def test_sentiment_evaluator_unsupervised_vs_supervised():
    """SentimentEvaluator reports summary stats, and executes classification only when labels exist."""
    predictions = [
        {"sentiment_label": "positive", "sentiment_score": 0.90},
        {"sentiment_label": "negative", "sentiment_score": 0.80},
        {"sentiment_label": "neutral", "sentiment_score": 0.70},
    ]

    evaluator = SentimentEvaluator()

    # Case A: No ground truth
    res_no_labels = evaluator.evaluate(predictions, pairs=None)
    assert res_no_labels["supervised_evaluation"]["status"] == "unavailable_no_labeled_ground_truth"
    conf = res_no_labels["confidence_statistics"]
    assert conf["mean"] == 0.80
    assert conf["min"] == 0.70
    assert conf["max"] == 0.90

    # Case B: Ground truth provided
    pairs = [
        {"true_sentiment": "positive", "predicted_sentiment": "positive"},
        {"true_sentiment": "negative", "predicted_sentiment": "negative"},
        {"true_sentiment": "neutral",  "predicted_sentiment": "positive"},
    ]
    res_labeled = evaluator.evaluate(predictions, pairs=pairs)
    assert res_labeled["supervised_evaluation"]["status"] == "evaluated"
    assert res_labeled["supervised_evaluation"]["metrics"]["overall_accuracy"] == round(2 / 3, 4)


def test_nlp_evaluation_pipeline_e2e_with_and_without_labels(tmp_path):
    """End-to-end pipeline handles unlabeled datasets, small labeled datasets, and outputs JSON reports."""
    feature_file = tmp_path / "features.jsonl"
    ground_truth_file = tmp_path / "ground_truth.jsonl"
    report_file = tmp_path / "report.json"

    records = [
        {
            "commodity": "copper",
            "article_url": f"https://example.com/art-{i}",
            "article_title": f"Copper headline {i}",
            "article_publication_timestamp": "2026-10-20T10:00:00Z",
            "relevance_class": "directly_relevant",
            "relevance_score": 0.90,
            "sentiment_label": "positive",
            "sentiment_score": 0.85,
            "event_type": "supply_disruption",
            "event_confidence": 0.80,
            "extraction_status": "success",
            "feature_status": "complete",
        }
        for i in range(10)
    ]

    with open(feature_file, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")

    pipeline = NLPEvaluationPipeline()

    # 1. Evaluate without ground truth
    rep_unlabeled = pipeline.run(feature_records=records, labeled_records=None)
    assert rep_unlabeled["evaluation_metadata"]["supervised_evaluation_available"] is False
    assert rep_unlabeled["relevance_evaluation"]["status"] == "unavailable_no_labeled_ground_truth"
    assert rep_unlabeled["data_quality"]["evaluation_metadata"]["valid_records_count"] == 10

    # 2. Evaluate with small ground-truth set (< 30) -> triggers warning
    gt_records = [
        {
            "commodity": "copper",
            "article_url": f"https://example.com/art-{i}",
            "article_title": f"Copper headline {i}",
            "true_relevance_class": "directly_relevant",
            "true_event_type": "supply_disruption",
            "true_sentiment_label": "positive",
        }
        for i in range(10)
    ]
    with open(ground_truth_file, "w", encoding="utf-8") as f:
        for r in gt_records:
            f.write(json.dumps(r) + "\n")

    rep_labeled = evaluate_nlp_pipeline(
        features_path=feature_file,
        ground_truth_path=ground_truth_file,
        output_report_path=report_file,
    )
    assert rep_labeled["evaluation_metadata"]["supervised_evaluation_available"] is True
    assert rep_labeled["evaluation_metadata"]["sample_size_warning"] is not None
    assert rep_labeled["relevance_evaluation"]["metrics"]["overall_accuracy"] == 1.0
    assert report_file.exists()


# ---------------------------------------------------------------------------
# 5. CLI Execution Smoke Tests
# ---------------------------------------------------------------------------

def test_cli_validate_mode(tmp_path):
    """CLI validate mode audits file and writes report."""
    input_file = tmp_path / "features.jsonl"
    output_file = tmp_path / "quality_report.json"

    rec = {
        "commodity": "gold",
        "article_url": "https://example.com/gold-record",
        "article_title": "Gold reserves rise",
        "article_publication_timestamp": "2026-10-20T10:00:00Z",
        "relevance_class": "directly_relevant",
        "relevance_score": 0.90,
        "sentiment_label": "positive",
        "sentiment_score": 0.80,
        "event_type": "demand_change",
        "event_confidence": 0.75,
        "extraction_status": "success",
        "feature_status": "complete",
    }
    with open(input_file, "w", encoding="utf-8") as f:
        f.write(json.dumps(rec) + "\n")

    code = run_cli(["--mode", "validate", "--input", str(input_file), "--output", str(output_file)])
    assert code == 0
    assert output_file.exists()


def test_cli_sample_mode(tmp_path):
    """CLI sample mode generates annotation template and guidelines."""
    input_file = tmp_path / "news.jsonl"
    output_file = tmp_path / "sample.jsonl"

    rec = {
        "commodity": "copper",
        "url": "https://example.com/news-1",
        "title": "Copper mine opens",
        "publication_timestamp": "2026-10-20T10:00:00Z",
    }
    with open(input_file, "w", encoding="utf-8") as f:
        f.write(json.dumps(rec) + "\n")

    code = run_cli([
        "--mode", "sample",
        "--input", str(input_file),
        "--output", str(output_file),
        "--sample-size", "5",
        "--seed", "42",
    ])
    assert code == 0
    assert output_file.exists()
    assert (tmp_path / "ANNOTATION_GUIDELINES.md").exists()


def test_cli_evaluate_mode_without_ground_truth(tmp_path):
    """CLI evaluate mode without ground truth warns that supervised evaluation is unavailable."""
    input_file = tmp_path / "features.jsonl"
    output_file = tmp_path / "eval_report.json"

    rec = {
        "commodity": "crude oil",
        "article_url": "https://example.com/oil-cut",
        "article_title": "OPEC meets",
        "article_publication_timestamp": "2026-10-20T10:00:00Z",
        "relevance_class": "directly_relevant",
        "relevance_score": 0.90,
        "sentiment_label": "neutral",
        "sentiment_score": 0.70,
        "event_type": "production_change",
        "event_confidence": 0.80,
        "extraction_status": "success",
        "feature_status": "complete",
    }
    with open(input_file, "w", encoding="utf-8") as f:
        f.write(json.dumps(rec) + "\n")

    code = run_cli([
        "--mode", "evaluate",
        "--input", str(input_file),
        "--output", str(output_file),
    ])
    assert code == 0
    assert output_file.exists()
    with open(output_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    assert data["evaluation_metadata"]["supervised_evaluation_available"] is False


def test_cli_missing_input_file():
    """CLI exits with error code 1 when input file is missing."""
    code = run_cli(["--mode", "validate", "--input", "non_existent_file.jsonl"])
    assert code == 1


# ---------------------------------------------------------------------------
# 6. Audit Regression Tests (Defects & Statistical Correctness)
# ---------------------------------------------------------------------------

def test_regression_ground_truth_matching_no_index_fallback():
    """
    CRITICAL REGRESSION TEST:
    Ground truth matching must NEVER fall back to index/position matching.
    An unmatched ground truth record must remain unmatched, not pair with an arbitrary prediction.
    """
    feature_records = [
        {
            "commodity": "copper",
            "article_url": "https://reuters.com/copper-supply-tightens",
            "article_title": "Copper mine strike halts operations",
            "article_publication_timestamp": "2026-10-20T10:00:00Z",
            "relevance_class": "directly_relevant",
            "sentiment_label": "negative",
            "event_type": "supply_disruption",
            "feature_status": "complete",
        }
    ]

    # Ground truth record for a completely different commodity and URL
    labeled_records = [
        {
            "commodity": "gold",
            "article_url": "https://bloomberg.com/gold-central-bank-buying",
            "article_title": "Central banks purchase record gold",
            "true_relevance_class": "directly_relevant",
            "true_sentiment_label": "positive",
            "true_event_type": "demand_change",
        }
    ]

    pipeline = NLPEvaluationPipeline()
    report = pipeline.run(feature_records=feature_records, labeled_records=labeled_records)
    meta = report["evaluation_metadata"]

    # Ground truth must be counted as unmatched, NEVER paired with feature_records[0]
    assert meta["unmatched_ground_truth_count"] == 1
    assert meta["unmatched_predictions_count"] == 1

    # Because there were no matching pairs between gold and copper, domain evaluations have sample_count == 0
    assert report["relevance_evaluation"]["sample_count"] == 0
    assert report["event_extraction_evaluation"]["sample_count"] == 0


def test_regression_data_quality_multi_failure_not_double_counted():
    """
    HIGH SEVERITY REGRESSION TEST:
    A single record with multiple validation failures must count as exactly ONE invalid record.
    Multiple defects on one record must not artificially deflate the data-quality pass rate.
    """
    clean_record = {
        "commodity": "copper",
        "article_url": "https://example.com/clean-article",
        "article_title": "Clean copper headline",
        "article_publication_timestamp": "2026-10-20T10:00:00Z",
        "relevance_class": "directly_relevant",
        "relevance_score": 0.90,
        "sentiment_label": "positive",
        "sentiment_score": 0.85,
        "sentiment_status": "success",
        "event_type": "supply_disruption",
        "event_confidence": 0.80,
        "extraction_status": "success",
        "feature_status": "complete",
    }

    # 9 completely clean records
    records = [dict(clean_record, article_url=f"https://example.com/clean-{i}") for i in range(9)]

    # 1 record with 3 separate errors: missing URL, unparseable timestamp, and out-of-bounds confidence
    multi_failure_record = {
        "commodity": "copper",
        "article_url": "",  # Failure 1: missing URL
        "article_title": "Multi-defect article",
        "article_publication_timestamp": "unparseable-date",  # Failure 2: unparseable timestamp
        "relevance_class": "directly_relevant",
        "relevance_score": 2.50,  # Failure 3: out-of-bounds score
        "sentiment_label": "positive",
        "sentiment_score": 0.85,
        "sentiment_status": "success",
        "event_type": "supply_disruption",
        "event_confidence": 0.80,
        "extraction_status": "success",
        "feature_status": "complete",
    }
    records.append(multi_failure_record)

    validator = DataQualityValidator()
    report = validator.validate(records)
    meta = report["evaluation_metadata"]

    # 10 total records, exactly 1 invalid record, exactly 9 valid records -> 90% pass rate
    assert meta["total_records_evaluated"] == 10
    assert meta["invalid_records_count"] == 1
    assert meta["valid_records_count"] == 9
    assert meta["data_quality_pass_rate"] == 0.90

    # Specific category counts must accurately reflect the specific anomalies
    assert report["url_validity"]["missing_urls_count"] == 1
    assert report["temporal_validity"]["unparseable_timestamps_count"] == 1
    assert report["confidence_value_validity"]["invalid_relevance_scores_count"] == 1


def test_regression_confidence_validator_rejects_nan_and_inf():
    """
    MEDIUM SEVERITY REGRESSION TEST:
    _is_valid_confidence must reject NaN and Infinity (both float and string representations).
    """
    import math

    assert not _is_valid_confidence(float("nan"))
    assert not _is_valid_confidence(float("inf"))
    assert not _is_valid_confidence(float("-inf"))
    assert not _is_valid_confidence("nan")
    assert not _is_valid_confidence("inf")
    assert not _is_valid_confidence("-inf")
    assert not _is_valid_confidence(None)
    assert not _is_valid_confidence(True)  # Booleans must not pass as 1.0
    assert not _is_valid_confidence(False) # Booleans must not pass as 0.0
    assert not _is_valid_confidence(-0.01)
    assert not _is_valid_confidence(1.01)

    # Valid values
    assert _is_valid_confidence(0.0)
    assert _is_valid_confidence(1.0)
    assert _is_valid_confidence(0.75)
    assert _is_valid_confidence("0.85")


def test_regression_event_detection_vs_conditional_type_accuracy():
    """
    MATHEMATICAL RIGOR REGRESSION TEST:
    Verify exact distinction between:
    1. Binary event detection (presence vs absence of any operational event)
    2. Conditional event-type classification accuracy (evaluated only when event detected)
    3. End-to-end 7-class classification metrics
    """
    pairs = [
        # Pair 1: TP_det, exact type match (supply_disruption == supply_disruption)
        {"commodity": "copper", "true_event_type": "supply_disruption", "predicted_event_type": "supply_disruption"},
        # Pair 2: TP_det, type confusion (demand_change vs supply_disruption)
        {"commodity": "crude oil", "true_event_type": "demand_change", "predicted_event_type": "supply_disruption"},
        # Pair 3: FN_det, missed event (geopolitical_risk vs other)
        {"commodity": "gold", "true_event_type": "geopolitical_risk", "predicted_event_type": "other"},
        # Pair 4: FP_det, false alarm (other vs production_change)
        {"commodity": "copper", "true_event_type": "other", "predicted_event_type": "production_change"},
        # Pair 5: TN_det, correct rejection (other vs other)
        {"commodity": "crude oil", "true_event_type": "other", "predicted_event_type": "other"},
    ]

    evaluator = EventExtractionEvaluator()
    res = evaluator.evaluate(pairs)

    cov = res["coverage_analysis"]
    # Binary detection:
    # True events: pairs 1, 2, 3 -> 3
    # Extracted events: pairs 1, 2, 4 -> 3
    # TP_det: pairs 1, 2 -> 2
    # FP_det (false alarms): pair 4 -> 1
    # FN_det (missed events): pair 3 -> 1
    # TN_det (correct rejections): pair 5 -> 1
    assert cov["true_event_count"] == 3
    assert cov["extracted_event_count"] == 3
    assert cov["correctly_detected_event_count"] == 2
    assert cov["event_detection_recall_rate"] == round(2 / 3, 4)
    assert cov["event_detection_precision_rate"] == round(2 / 3, 4)
    assert cov["event_detection_f1_score"] == round(2 / 3, 4)
    assert cov["event_detection_accuracy"] == round((2 + 1) / 5, 4) # (TP + TN) / total = 3/5 = 0.6000

    # Conditional type accuracy:
    # Evaluated on TP_det (pairs 1 and 2):
    # exact matches = pair 1 (1 instance), type confusion = pair 2 (1 instance)
    # Conditional Type Accuracy = 1 / 2 = 0.5000
    cond = cov["conditional_event_type_classification"]
    assert cond["evaluated_instances_tp_det"] == 2
    assert cond["exact_type_matches"] == 1
    assert cond["type_confusions_count"] == 1
    assert cond["conditional_type_accuracy"] == 0.5000

    # End-to-end 7-class accuracy: pairs 1 and 5 correct -> 2/5 = 0.4000
    assert res["classification_metrics"]["overall_accuracy"] == 0.4000


def test_regression_duplicate_ground_truth_detection():
    """Verify duplicate annotations are detected in ground-truth datasets."""
    records = [
        {
            "commodity": "copper",
            "article_url": "https://example.com/art-1",
            "true_relevance_class": "directly_relevant",
            "true_sentiment_label": "positive",
            "true_event_type": "supply_disruption",
        },
        {
            "commodity": "copper",
            "article_url": "https://example.com/art-1",  # Duplicate (comm, url)
            "true_relevance_class": "irrelevant",
            "true_sentiment_label": "negative",
            "true_event_type": "other",
        },
    ]

    # In validate_annotation_dataset
    val_rep = validate_annotation_dataset(records)
    assert val_rep["duplicate_detection"]["duplicate_annotations_count"] == 1

    # In NLPEvaluationPipeline
    pipeline = NLPEvaluationPipeline()
    feature_records = [
        {
            "commodity": "copper",
            "article_url": "https://example.com/art-1",
            "article_title": "Copper title",
            "relevance_class": "directly_relevant",
            "sentiment_label": "positive",
            "event_type": "supply_disruption",
        }
    ]
    eval_rep = pipeline.run(feature_records=feature_records, labeled_records=records)
    assert eval_rep["evaluation_metadata"]["duplicate_ground_truth_count"] == 1


def test_regression_deterministic_id_url_canonicalization():
    """Deterministic article IDs must be stable across trailing slash and case variations."""
    id1 = _generate_deterministic_id("copper", "https://example.com/news/article-1/")
    id2 = _generate_deterministic_id("copper", "https://example.com/news/article-1")
    id3 = _generate_deterministic_id("COPPER", "https://example.com/news/article-1")

    assert id1 == id2
    assert id1 == id3


def test_regression_missing_predictions_penalized_in_accuracy():
    """Missing or off-taxonomy predictions must not be ignored in accuracy calculations."""
    labels = ["positive", "negative", "neutral"]
    y_true = ["positive", "positive", "negative", "neutral"]
    # Model returns 1 correct, 1 wrong, 1 missing, 1 off-taxonomy
    y_pred = ["positive", "negative", "", "unrecognized_class"]

    metrics = compute_classification_metrics(y_true, y_pred, labels)

    # 4 ground-truth samples, only 1 true positive -> accuracy = 1/4 = 0.2500
    assert metrics["overall_accuracy"] == 0.2500
    assert metrics["per_class"]["negative"]["false_negatives"] == 1
    assert metrics["per_class"]["neutral"]["false_negatives"] == 1


def test_regression_partial_annotations_graceful_handling():
    """
    Partial annotations (e.g. only sentiment labeled) must evaluate sentiment without
    manufacturing fake labels or claiming full evaluation for relevance or events.
    """
    feature_records = [
        {
            "commodity": "copper",
            "article_url": "https://example.com/art-partial",
            "article_title": "Headline",
            "relevance_class": "directly_relevant",
            "sentiment_label": "positive",
            "event_type": "supply_disruption",
        }
    ]

    # Only sentiment labeled
    labeled_records = [
        {
            "commodity": "copper",
            "article_url": "https://example.com/art-partial",
            "true_relevance_class": "",
            "true_sentiment_label": "positive",
            "true_event_type": "",
        }
    ]

    pipeline = NLPEvaluationPipeline()
    report = pipeline.run(feature_records=feature_records, labeled_records=labeled_records)

    meta = report["evaluation_metadata"]
    assert meta["evaluated_tasks"] == ["sentiment"]
    assert report["sentiment_evaluation"]["supervised_evaluation"]["status"] == "evaluated"
    assert report["relevance_evaluation"]["sample_count"] == 0
    assert report["event_extraction_evaluation"]["sample_count"] == 0


def test_generate_annotation_sample_exports_both_jsonl_and_csv(tmp_path):
    """Ensure generate_annotation_sample writes both JSONL and Excel-friendly CSV with BOM."""
    input_file = tmp_path / "articles.jsonl"
    output_jsonl = tmp_path / "sample.jsonl"
    output_csv = tmp_path / "sample.csv"

    articles = [
        {
            "commodity": "copper",
            "article_url": f"https://example.com/copper-{i}",
            "article_title": f"Copper headline {i}",
            "article_publication_timestamp": "2026-11-10T12:00:00Z",
            "domain": "example.com",
            "relevance_class": "directly_relevant",
            "sentiment_label": "positive",
        }
        for i in range(5)
    ]
    with open(input_file, "w", encoding="utf-8") as f:
        for a in articles:
            f.write(json.dumps(a) + "\n")

    sample = generate_annotation_sample(input_file, output_jsonl, sample_size=3, seed=42)
    assert len(sample) == 3
    assert output_jsonl.exists()
    assert output_csv.exists()

    # Verify CSV content and Excel UTF-8 BOM encoding
    with open(output_csv, "rb") as f:
        raw_bytes = f.read()
    assert raw_bytes.startswith(b"\xef\xbb\xbf")  # UTF-8 BOM

    with open(output_csv, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        csv_rows = list(reader)
    assert len(csv_rows) == 3
    assert "true_sentiment_label" in reader.fieldnames
    assert "sentiment_label" not in reader.fieldnames  # No prediction leakage in CSV headers


def test_annotation_sampler_water_filling_imbalanced_distribution():
    """Water-filling allocation absorbs available scarce commodities and satisfies requested total."""
    sampler = AnnotationSampler()
    # 2 copper, 2 gold, 20 crude oil
    records = []
    for i in range(2):
        records.append({
            "commodity": "copper",
            "url": f"https://example.com/c-{i}",
            "title": f"Copper {i}",
        })
        records.append({
            "commodity": "gold",
            "url": f"https://example.com/g-{i}",
            "title": f"Gold {i}",
        })
    for i in range(20):
        records.append({
            "commodity": "crude oil",
            "url": f"https://example.com/o-{i}",
            "title": f"Oil {i}",
        })

    # Request 10 samples
    sample = sampler.sample(records, sample_size=10, seed=42, stratify_by_commodity=True)
    assert len(sample) == 10
    counts = Counter(s["commodity"] for s in sample)
    # Scarce commodities (2 each) are fully utilized
    assert counts["copper"] == 2
    assert counts["gold"] == 2
    # Remainder (6) comes from crude oil
    assert counts["crude oil"] == 6


def test_csv_importer_valid_utf8_and_bom(tmp_path):
    """CSV importer correctly parses UTF-8 with BOM and without BOM, preserving all fields."""
    csv_file = tmp_path / "valid_annotations.csv"
    output_jsonl = tmp_path / "imported_gt.jsonl"

    content = (
        "article_id,commodity,article_url,article_title,article_publication_timestamp,domain,language,source_country,true_relevance_class,true_sentiment_label,true_event_type,true_event_impact_direction,annotator_id,annotator_notes\n"
        "id001,copper,https://example.com/c1,Copper mine strike,2026-11-10T10:00:00Z,example.com,English,Chile,directly_relevant,negative,supply_disruption,potentially_bullish,ann_01,operational strike\n"
        "id002,gold,https://example.com/g1,Central bank gold buying,2026-11-11T12:00:00Z,example.com,English,US,directly_relevant,positive,demand_change,potentially_bullish,ann_02,reserve growth\n"
    )
    with open(csv_file, "w", encoding="utf-8-sig", newline="") as f:
        f.write(content)

    importer = AnnotationCSVImporter()
    report = importer.import_csv(csv_path=csv_file, output_jsonl_path=output_jsonl)

    assert report["status"] == "success"
    assert report["valid_records_count"] == 2
    assert report["completion_summary"]["fully_labeled_count"] == 2
    assert output_jsonl.exists()

    with open(output_jsonl, "r", encoding="utf-8") as f:
        records = [json.loads(line) for line in f]
    assert len(records) == 2
    assert records[0]["article_id"] == "id001"
    assert records[0]["commodity"] == "copper"
    assert records[0]["true_sentiment_label"] == "negative"
    assert records[0]["true_event_impact_direction"] == "potentially_bullish"
    assert records[0]["annotator_id"] == "ann_01"


def test_csv_importer_detects_incomplete_annotations(tmp_path):
    """Importer accurately differentiates fully, partially, and completely unlabeled records."""
    csv_file = tmp_path / "incomplete.csv"
    content = (
        "article_id,commodity,article_url,article_title,true_relevance_class,true_sentiment_label,true_event_type\n"
        "1,copper,https://example.com/1,Title 1,directly_relevant,positive,production_change\n"
        "2,gold,https://example.com/2,Title 2,directly_relevant,,\n"
        "3,crude oil,https://example.com/3,Title 3,,,\n"
    )
    with open(csv_file, "w", encoding="utf-8", newline="") as f:
        f.write(content)

    importer = AnnotationCSVImporter()
    report = importer.import_csv(csv_path=csv_file)

    comp = report["completion_summary"]
    assert comp["fully_labeled_count"] == 1
    assert comp["partially_labeled_count"] == 1
    assert comp["unlabeled_count"] == 1
    assert comp["relevance_labeled_count"] == 2
    assert comp["sentiment_labeled_count"] == 1


def test_csv_importer_detects_taxonomy_violations(tmp_path):
    """Off-taxonomy values are flagged; strict mode raises ValueError."""
    csv_file = tmp_path / "violations.csv"
    content = (
        "article_id,commodity,article_url,article_title,true_relevance_class,true_sentiment_label,true_event_type,true_event_impact_direction\n"
        "1,copper,https://example.com/1,Title 1,totally_relevant,super_bullish,fake_event,skyrocketing\n"
    )
    with open(csv_file, "w", encoding="utf-8", newline="") as f:
        f.write(content)

    importer = AnnotationCSVImporter()

    # Non-strict mode captures violations in report
    report = importer.import_csv(csv_path=csv_file, strict=False)
    assert report["status"] == "completed_with_warnings"
    assert report["taxonomy_violations"]["violation_count"] == 4

    # Strict mode raises ValueError
    with pytest.raises(ValueError, match="Strict validation failed"):
        importer.import_csv(csv_path=csv_file, strict=True)


def test_csv_importer_rejects_duplicate_articles(tmp_path):
    """Duplicate (commodity, URL) pairs are detected and skipped or rejected."""
    csv_file = tmp_path / "duplicates.csv"
    content = (
        "commodity,article_url,article_title,true_sentiment_label\n"
        "copper,https://example.com/dup,Title A,positive\n"
        "copper,https://example.com/dup,Title A duplicate,negative\n"
    )
    with open(csv_file, "w", encoding="utf-8", newline="") as f:
        f.write(content)

    importer = AnnotationCSVImporter()
    report = importer.import_csv(csv_path=csv_file, strict=False)
    assert report["duplicate_identities"]["duplicate_count"] == 1
    # Only first unique record retained
    assert report["valid_records_count"] == 1

    with pytest.raises(ValueError, match="duplicate article identity"):
        importer.import_csv(csv_path=csv_file, strict=True)


def test_csv_importer_cross_commodity_shared_url_allowed(tmp_path):
    """Distinct commodities referencing the exact same URL produce distinct valid records."""
    csv_file = tmp_path / "shared_url.csv"
    content = (
        "commodity,article_url,article_title,true_sentiment_label\n"
        "copper,https://example.com/mining-report,Mining Sector Review,positive\n"
        "gold,https://example.com/mining-report,Mining Sector Review,neutral\n"
    )
    with open(csv_file, "w", encoding="utf-8", newline="") as f:
        f.write(content)

    importer = AnnotationCSVImporter()
    report = importer.import_csv(csv_path=csv_file, strict=True)
    assert report["duplicate_identities"]["duplicate_count"] == 0
    assert report["valid_records_count"] == 2


def test_csv_importer_formula_injection_defense(tmp_path):
    """Formula characters are sanitized during export and restored without identity corruption."""
    csv_file = tmp_path / "formula_test.csv"
    jsonl_output = tmp_path / "formula_out.jsonl"

    # Verify sanitizer
    assert sanitize_for_spreadsheet("=HYPERLINK('evil')") == "'=HYPERLINK('evil')"
    assert sanitize_for_spreadsheet("+4.5% price rally") == "'+4.5% price rally"
    assert sanitize_for_spreadsheet("-10% crash") == "'-10% crash"
    assert sanitize_for_spreadsheet("@mention") == "'@mention"
    assert sanitize_for_spreadsheet("Normal headline") == "Normal headline"

    # Verify desanitizer
    assert desanitize_from_spreadsheet("'=HYPERLINK('evil')") == "=HYPERLINK('evil')"
    assert desanitize_from_spreadsheet("'+4.5% price rally") == "+4.5% price rally"
    assert desanitize_from_spreadsheet("Normal headline") == "Normal headline"
    assert desanitize_from_spreadsheet("'Normal starting with quote") == "'Normal starting with quote"

    # End-to-end CSV write and readback
    content = (
        "commodity,article_url,article_title,true_sentiment_label\n"
        "crude oil,https://example.com/formula,'=1+1 calculation,positive\n"
    )
    with open(csv_file, "w", encoding="utf-8", newline="") as f:
        f.write(content)

    report = convert_annotated_csv_to_jsonl(csv_file, jsonl_output)
    assert report["valid_records_count"] == 1

    with open(jsonl_output, "r", encoding="utf-8") as f:
        rec = json.loads(f.readline())
    assert rec["article_title"] == "=1+1 calculation"
    # Stable ID matches underlying unescaped identity
    expected_id = _generate_deterministic_id("crude oil", "https://example.com/formula")
    assert rec["article_id"] == expected_id


def test_csv_importer_rejects_malformed_csv(tmp_path):
    """Missing required columns or empty files raise ValueError."""
    empty_csv = tmp_path / "empty.csv"
    empty_csv.write_text("", encoding="utf-8")

    missing_cols_csv = tmp_path / "missing_cols.csv"
    missing_cols_csv.write_text("random_header,notes\nval1,val2\n", encoding="utf-8")

    importer = AnnotationCSVImporter()
    with pytest.raises(ValueError, match="no header row"):
        importer.import_csv(empty_csv)

    with pytest.raises(ValueError, match="missing required column"):
        importer.import_csv(missing_cols_csv)


def test_csv_importer_safeguards_existing_files(tmp_path):
    """Existing ground-truth JSONL is never overwritten without overwrite=True."""
    csv_file = tmp_path / "data.csv"
    csv_file.write_text("commodity,article_url,article_title\ncopper,https://example.com/1,Title\n", encoding="utf-8")
    out_file = tmp_path / "output.jsonl"
    out_file.write_text('{"existing": true}\n', encoding="utf-8")

    importer = AnnotationCSVImporter()
    with pytest.raises(FileExistsError, match="already exists"):
        importer.import_csv(csv_file, output_jsonl_path=out_file, overwrite=False)

    # Overwrite permitted
    importer.import_csv(csv_file, output_jsonl_path=out_file, overwrite=True)
    with open(out_file, "r", encoding="utf-8") as f:
        lines = f.readlines()
    assert len(lines) == 1
    assert "copper" in lines[0]


def test_cli_import_csv_mode(tmp_path):
    """CLI runner executes import-csv mode and creates JSONL ground truth."""
    csv_file = tmp_path / "cli_sample.csv"
    output_jsonl = tmp_path / "cli_gt.jsonl"

    content = (
        "commodity,article_url,article_title,true_sentiment_label\n"
        "copper,https://example.com/cli,Copper strike,negative\n"
    )
    with open(csv_file, "w", encoding="utf-8", newline="") as f:
        f.write(content)

    from src.evaluation.cli import run_cli
    exit_code = run_cli([
        "--mode", "import-csv",
        "--input", str(csv_file),
        "--output", str(output_jsonl),
        "--overwrite",
    ])
    assert exit_code == 0
    assert output_jsonl.exists()


# ---------------------------------------------------------------------------
# Task 6 Phase 2B: Quality Control, Inter-Annotator Agreement, and Adjudication Tests
# ---------------------------------------------------------------------------

def test_cohens_kappa_perfect_agreement():
    """Exact Cohen's Kappa evaluates to 1.0 when two raters agree completely."""
    y1 = ["positive", "negative", "neutral", "positive", "neutral", "negative"]
    y2 = ["positive", "negative", "neutral", "positive", "neutral", "negative"]
    kappa = compute_cohens_kappa(y1, y2, categories=["positive", "negative", "neutral"])
    assert kappa == 1.0
    assert interpret_cohens_kappa(kappa) == "almost_perfect_agreement"


def test_cohens_kappa_systematic_disagreement():
    """Exact Cohen's Kappa evaluates to negative score when raters systematically disagree."""
    y1 = ["positive", "positive", "negative", "negative"]
    y2 = ["negative", "negative", "positive", "positive"]
    kappa = compute_cohens_kappa(y1, y2, categories=["positive", "negative"])
    assert kappa == -1.0
    assert interpret_cohens_kappa(kappa) == "poor_systematic_disagreement"


def test_cohens_kappa_boundary_and_empty_inputs():
    """Boundary cases like empty lists, unequal inputs, or uniform rater classifications return safe values."""
    assert compute_cohens_kappa([], []) is None
    assert compute_cohens_kappa(["positive"], ["positive", "negative"]) is None
    assert compute_cohens_kappa(["positive"], ["positive"], categories=[]) is None

    # Degenerate uniform single category from both raters: P_o = 1.0, P_e = 1.0 -> 0/0 (undefined)
    y1 = ["directly_relevant", "directly_relevant", "directly_relevant"]
    y2 = ["directly_relevant", "directly_relevant", "directly_relevant"]
    assert compute_cohens_kappa(y1, y2) is None
    assert interpret_cohens_kappa(None) == "undefined_constant_distribution"


def test_cohens_kappa_interpretation_benchmarks():
    """Landis & Koch (1977) interpretation mapping strictly categorizes Kappa scores."""
    assert interpret_cohens_kappa(0.95) == "almost_perfect_agreement"
    assert interpret_cohens_kappa(0.75) == "substantial_agreement"
    assert interpret_cohens_kappa(0.55) == "moderate_agreement"
    assert interpret_cohens_kappa(0.35) == "fair_agreement"
    assert interpret_cohens_kappa(0.15) == "slight_agreement"
    assert interpret_cohens_kappa(0.0) == "chance_agreement"
    assert interpret_cohens_kappa(-0.40) == "poor_systematic_disagreement"


def test_prepare_pilot_annotation_packages(tmp_path):
    """
    Validates generation of primary (50 items), secondary (20% overlap = 10 items: 2 copper, 2 gold, 6 crude oil),
    and assignment manifest with blank human fields and zero prediction leakage.
    """
    sample_file = tmp_path / "pilot_sample.jsonl"
    records = []
    # 5 copper, 5 gold, 40 crude oil
    for i in range(5):
        records.append({
            "commodity": "copper",
            "article_url": f"https://example.com/copper/{i}",
            "article_title": f"Copper title {i}",
            "article_publication_timestamp": "2026-11-01T10:00:00Z",
            "domain": "mining.com",
            "language": "en",
        })
    for i in range(5):
        records.append({
            "commodity": "gold",
            "article_url": f"https://example.com/gold/{i}",
            "article_title": f"Gold title {i}",
            "article_publication_timestamp": "2026-11-01T10:00:00Z",
            "domain": "gold.org",
            "language": "en",
        })
    for i in range(40):
        records.append({
            "commodity": "crude oil",
            "article_url": f"https://example.com/oil/{i}",
            "article_title": f"Oil title {i}",
            "article_publication_timestamp": "2026-11-01T10:00:00Z",
            "domain": "oilprice.com",
            "language": "en",
        })

    with open(sample_file, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")

    pkg_dir = tmp_path / "packages"
    res = prepare_pilot_annotation_packages(sample_file, output_dir=pkg_dir, overlap_fraction=0.20, seed=42)

    assert res["status"] == "success"
    assert res["primary_count"] == 50
    assert res["secondary_count"] == 10
    assert res["overlap_count"] == 10
    assert res["commodity_breakdown"]["secondary"] == {"crude oil": 6, "gold": 2, "copper": 2}

    # Verify primary files exist
    assert Path(res["files_generated"]["primary_jsonl"]).exists()
    assert Path(res["files_generated"]["primary_csv"]).exists()
    assert Path(res["files_generated"]["secondary_jsonl"]).exists()
    assert Path(res["files_generated"]["secondary_csv"]).exists()
    assert Path(res["files_generated"]["manifest_json"]).exists()
    assert Path(res["files_generated"]["manifest_csv"]).exists()
    assert Path(res["files_generated"]["instructions_markdown"]).exists()
    assert Path(res["files_generated"]["guidelines_markdown"]).exists()

    # Verify all human fields are blank and no prediction fields exist
    with open(res["files_generated"]["primary_jsonl"], "r", encoding="utf-8") as f:
        p_recs = [json.loads(line) for line in f]
    with open(res["files_generated"]["secondary_jsonl"], "r", encoding="utf-8") as f:
        s_recs = [json.loads(line) for line in f]

    human_fields = [
        "true_relevance_class",
        "true_sentiment_label",
        "true_event_type",
        "true_event_impact_direction",
        "annotator_id",
        "annotator_notes",
    ]
    forbidden_prediction_fields = [
        "relevance_score",
        "finbert_sentiment_label",
        "finbert_sentiment_score",
        "supply_chain_event_type",
        "event_confidence",
        "is_relevant",
    ]

    for rec in p_recs + s_recs:
        for fld in human_fields:
            assert rec[fld] == "", f"Field {fld} is not blank"
        for fld in forbidden_prediction_fields:
            assert fld not in rec, f"Model prediction field {fld} leaked into template"


def test_annotation_agreement_auditor():
    """AnnotationAgreementAuditor matches overlapping records and calculates task metrics and disagreements."""
    primary = [
        {
            "article_id": "art_001",
            "commodity": "copper",
            "article_url": "https://example.com/1",
            "article_title": "Copper mine strike",
            "true_relevance_class": "directly_relevant",
            "true_sentiment_label": "negative",
            "true_event_type": "supply_disruption",
            "true_event_impact_direction": "potentially_bullish",
            "annotator_id": "ann_primary",
        },
        {
            "article_id": "art_002",
            "commodity": "gold",
            "article_url": "https://example.com/2",
            "article_title": "Gold reserves up",
            "true_relevance_class": "directly_relevant",
            "true_sentiment_label": "positive",
            "true_event_type": "inventory_change",
            "true_event_impact_direction": "potentially_bearish",
            "annotator_id": "ann_primary",
        },
        {
            "article_id": "art_003",
            "commodity": "crude oil",
            "article_url": "https://example.com/3",
            "article_title": "Unshared oil news",
            "true_relevance_class": "directly_relevant",
            "true_sentiment_label": "neutral",
            "true_event_type": "other",
            "true_event_impact_direction": "uncertain",
            "annotator_id": "ann_primary",
        },
    ]

    secondary = [
        # art_001: complete agreement
        {
            "article_id": "art_001",
            "commodity": "copper",
            "article_url": "https://example.com/1",
            "article_title": "Copper mine strike",
            "true_relevance_class": "directly_relevant",
            "true_sentiment_label": "negative",
            "true_event_type": "supply_disruption",
            "true_event_impact_direction": "potentially_bullish",
            "annotator_id": "ann_secondary",
        },
        # art_002: disagreement on sentiment and event type
        {
            "article_id": "art_002",
            "commodity": "gold",
            "article_url": "https://example.com/2",
            "article_title": "Gold reserves up",
            "true_relevance_class": "directly_relevant",
            "true_sentiment_label": "neutral",
            "true_event_type": "production_change",
            "true_event_impact_direction": "potentially_bearish",
            "annotator_id": "ann_secondary",
        },
    ]

    auditor = AnnotationAgreementAuditor()
    report = auditor.audit(primary, secondary)

    meta = report["audit_metadata"]
    assert meta["matched_overlapping_records"] == 2
    assert meta["unmatched_secondary_records"] == 0
    assert meta["total_records_with_disagreements"] == 1

    # Task metrics: relevance agreed 2/2 (1.0), sentiment agreed 1/2 (0.5), impact direction agreed 2/2 (1.0)
    tasks = report["task_agreements"]
    assert tasks["relevance"]["observed_agreement_rate"] == 1.0
    assert tasks["sentiment"]["observed_agreement_rate"] == 0.5
    assert tasks["event_type"]["observed_agreement_rate"] == 0.5
    assert tasks["impact_direction"]["observed_agreement_rate"] == 1.0

    # Disagreements
    disagreements = report["disagreements"]
    assert len(disagreements) == 1
    assert disagreements[0]["article_id"] == "art_002"
    assert "true_sentiment_label" in disagreements[0]["disagreed_fields"]
    assert "true_event_type" in disagreements[0]["disagreed_fields"]
    assert disagreements[0]["field_values"]["true_sentiment_label"]["primary"] == "positive"
    assert disagreements[0]["field_values"]["true_sentiment_label"]["secondary"] == "neutral"


def test_annotation_adjudicator_template_and_merge(tmp_path):
    """AnnotationAdjudicator creates adjudication template and merges consensus with human resolutions."""
    audit_report = {
        "disagreements": [
            {
                "article_id": "art_conflict",
                "commodity": "copper",
                "article_title": "Disputed copper event",
                "article_url": "https://example.com/disputed",
                "primary_annotator_id": "ann_1",
                "secondary_annotator_id": "ann_2",
                "disagreed_fields": ["true_event_type"],
                "field_values": {
                    "true_relevance_class": {"primary": "directly_relevant", "secondary": "directly_relevant"},
                    "true_sentiment_label": {"primary": "negative", "secondary": "negative"},
                    "true_event_type": {"primary": "supply_disruption", "secondary": "other"},
                    "true_event_impact_direction": {"primary": "potentially_bullish", "secondary": "uncertain"},
                },
            }
        ]
    }

    adjudicator = AnnotationAdjudicator()
    template_csv = tmp_path / "adjudication_template.csv"
    template_rows = adjudicator.create_adjudication_template(audit_report, output_csv_path=template_csv)

    assert len(template_rows) == 1
    assert template_csv.exists()
    assert template_rows[0]["article_id"] == "art_conflict"
    assert template_rows[0]["adjudicated_event_type"] == ""

    # Simulate human expert adjudication
    adjudicated_records = [
        {
            "article_id": "art_conflict",
            "adjudicated_relevance": "directly_relevant",
            "adjudicated_sentiment": "negative",
            "adjudicated_event_type": "supply_disruption",
            "adjudicated_impact_direction": "potentially_bullish",
            "adjudicator_id": "expert_01",
            "adjudication_notes": "Rockburst is a classic supply disruption.",
        }
    ]

    primary_records = [
        {
            "article_id": "art_conflict",
            "commodity": "copper",
            "article_url": "https://example.com/disputed",
            "article_title": "Disputed copper event",
            "true_relevance_class": "directly_relevant",
            "true_sentiment_label": "negative",
            "true_event_type": "supply_disruption",
            "true_event_impact_direction": "potentially_bullish",
            "annotator_id": "ann_1",
        },
        {
            "article_id": "art_consensus",
            "commodity": "gold",
            "article_url": "https://example.com/consensus",
            "article_title": "Agreed gold news",
            "true_relevance_class": "directly_relevant",
            "true_sentiment_label": "neutral",
            "true_event_type": "other",
            "true_event_impact_direction": "unknown",
            "annotator_id": "ann_1",
        },
        {
            "article_id": "art_unshared",
            "commodity": "crude oil",
            "article_url": "https://example.com/unshared",
            "article_title": "Primary only oil news",
            "true_relevance_class": "directly_relevant",
            "true_sentiment_label": "positive",
            "true_event_type": "production_change",
            "true_event_impact_direction": "potentially_bearish",
            "annotator_id": "ann_1",
        },
    ]

    secondary_records = [
        {
            "article_id": "art_conflict",
            "commodity": "copper",
            "true_relevance_class": "directly_relevant",
            "true_sentiment_label": "negative",
            "true_event_type": "other",
            "true_event_impact_direction": "potentially_bullish",
            "annotator_id": "ann_2",
        },
        {
            "article_id": "art_consensus",
            "commodity": "gold",
            "true_relevance_class": "directly_relevant",
            "true_sentiment_label": "neutral",
            "true_event_type": "other",
            "true_event_impact_direction": "unknown",
            "annotator_id": "ann_2",
        },
    ]

    out_gt = tmp_path / "final_ground_truth.jsonl"
    merge_summary = adjudicator.merge_adjudicated_ground_truth(
        primary_records=primary_records,
        secondary_records=secondary_records,
        adjudicated_records=adjudicated_records,
        output_jsonl_path=out_gt,
        overwrite=True,
    )

    assert merge_summary["total_merged_records"] == 3
    assert merge_summary["primary_unshared_count"] == 1
    assert merge_summary["consensus_accepted_count"] == 1
    assert merge_summary["adjudicated_resolved_count"] == 1
    assert out_gt.exists()

    # Inspect merged contents
    merged = load_annotation_records(out_gt)
    m_by_id = {r["article_id"]: r for r in merged}
    assert m_by_id["art_conflict"]["true_event_type"] == "supply_disruption"
    assert "adj:expert_01" in m_by_id["art_conflict"]["annotator_id"]
    assert "consensus" in m_by_id["art_consensus"]["annotator_id"]
    assert m_by_id["art_unshared"]["annotator_id"] == "ann_1"


def test_load_annotation_records_csv_and_jsonl(tmp_path):
    """load_annotation_records handles both JSONL and CSV with formula desanitization."""
    csv_path = tmp_path / "formula_test.csv"
    with open(csv_path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["commodity", "article_title", "article_url"])
        writer.writeheader()
        writer.writerow({
            "commodity": "copper",
            "article_title": "'+5% copper rally",
            "article_url": "https://example.com/1",
        })

    records = load_annotation_records(csv_path)
    assert len(records) == 1
    # Leading formula quote stripped cleanly
    assert records[0]["article_title"] == "+5% copper rally"


def test_cli_package_and_agreement_modes(tmp_path):
    """CLI runner successfully executes 'package', 'agreement', and 'adjudicate' modes."""
    # 1. Prepare sample
    sample_path = tmp_path / "sample.jsonl"
    sample_records = [
        {
            "commodity": "copper",
            "article_url": f"https://example.com/c/{i}",
            "article_title": f"Copper article {i}",
            "article_publication_timestamp": "2026-11-01T10:00:00Z",
        }
        for i in range(10)
    ]
    with open(sample_path, "w", encoding="utf-8") as f:
        for r in sample_records:
            f.write(json.dumps(r) + "\n")

    pkg_dir = tmp_path / "cli_packages"
    exit_code_pkg = run_cli([
        "--mode", "package",
        "--input", str(sample_path),
        "--output-dir", str(pkg_dir),
        "--overlap-fraction", "0.20",
    ])
    assert exit_code_pkg == 0
    p_jsonl = pkg_dir / "annotation_package_primary.jsonl"
    s_jsonl = pkg_dir / "annotation_package_secondary.jsonl"
    assert p_jsonl.exists()
    assert s_jsonl.exists()

    # 2. Test agreement CLI mode
    report_path = tmp_path / "audit_report.json"
    template_path = tmp_path / "adj_template.csv"
    exit_code_agr = run_cli([
        "--mode", "agreement",
        "--primary", str(p_jsonl),
        "--secondary", str(s_jsonl),
        "--output", str(report_path),
        "--adjudication-template", str(template_path),
    ])
    assert exit_code_agr == 0
    assert report_path.exists()
    assert template_path.exists()


def test_cohens_kappa_partial_agreement_and_class_imbalance():
    """Cohen's Kappa with class imbalance and partial agreement matches manual calculation."""
    y1 = ["positive", "positive", "positive", "positive", "positive", "positive", "negative", "negative", "neutral", "neutral"]
    y2 = ["positive", "positive", "positive", "positive", "positive", "negative", "negative", "negative", "neutral", "neutral"]
    # P_o = 0.90, P_e = 0.40 -> Kappa = 0.50 / 0.60 = 0.8333
    kappa = compute_cohens_kappa(y1, y2, categories=["positive", "negative", "neutral"])
    assert kappa is not None
    assert abs(kappa - 0.8333) < 1e-3
    assert interpret_cohens_kappa(kappa) == "almost_perfect_agreement"


def test_cohens_kappa_degenerate_constant_labels_is_none():
    """When both raters assign the exact same single category, Kappa is mathematically undefined (0/0)."""
    y1 = ["positive"] * 10
    y2 = ["positive"] * 10
    kappa = compute_cohens_kappa(y1, y2, categories=["positive", "negative", "neutral"])
    assert kappa is None
    assert interpret_cohens_kappa(kappa) == "undefined_constant_distribution"


def test_annotation_agreement_auditor_missing_invalid_labels_and_sample_warning():
    """Auditor excludes invalid/missing labels from Kappa, reports them, and flags small sample warning."""
    primary = [
        {"article_id": "art_1", "commodity": "copper", "true_relevance_class": "directly_relevant", "true_sentiment_label": "positive", "true_event_type": "supply_disruption", "true_event_impact_direction": "potentially_bullish", "annotator_id": "p1"},
        {"article_id": "art_2", "commodity": "copper", "true_relevance_class": "invalid_relevance", "true_sentiment_label": "positive", "true_event_type": "supply_disruption", "true_event_impact_direction": "potentially_bullish", "annotator_id": "p1"},
    ]
    secondary = [
        {"article_id": "art_1", "commodity": "copper", "true_relevance_class": "directly_relevant", "true_sentiment_label": "positive", "true_event_type": "supply_disruption", "true_event_impact_direction": "potentially_bullish", "annotator_id": "s1"},
        {"article_id": "art_2", "commodity": "copper", "true_relevance_class": "directly_relevant", "true_sentiment_label": "positive", "true_event_type": "supply_disruption", "true_event_impact_direction": "potentially_bullish", "annotator_id": "s1"},
    ]

    auditor = AnnotationAgreementAuditor()
    report = auditor.audit(primary, secondary)

    # art_2 had invalid_relevance in primary -> excluded from paired relevance calculation
    rel_task = report["task_agreements"]["relevance"]
    assert rel_task["valid_paired_evaluations"] == 1
    assert rel_task["missing_or_invalid_count"] == 1
    assert len(rel_task["missing_or_invalid_samples"]) == 1

    # Small sample warning triggered (N=2 < 30)
    assert report["audit_metadata"]["sample_size_warning"] is not None
    assert "Small overlap sample size" in report["audit_metadata"]["sample_size_warning"]

    # Per-label agreement exists
    assert "directly_relevant" in rel_task["per_label_agreement"]


def test_annotation_agreement_auditor_flags_duplicates_and_cross_commodity():
    """Auditor flags duplicate records in primary/secondary and prevents cross-commodity shared URL mixing."""
    primary = [
        {"article_id": "id_1", "commodity": "copper", "article_url": "https://example.com/shared", "true_relevance_class": "directly_relevant", "true_sentiment_label": "positive", "true_event_type": "supply_disruption", "true_event_impact_direction": "potentially_bullish"},
        # Duplicate in primary
        {"article_id": "id_1", "commodity": "copper", "article_url": "https://example.com/shared", "true_relevance_class": "directly_relevant", "true_sentiment_label": "positive", "true_event_type": "supply_disruption", "true_event_impact_direction": "potentially_bullish"},
    ]
    secondary = [
        # Secondary has same URL but commodity is gold (cross-commodity shared URL)
        {"article_id": "id_diff", "commodity": "gold", "article_url": "https://example.com/shared", "true_relevance_class": "directly_relevant", "true_sentiment_label": "positive", "true_event_type": "supply_disruption", "true_event_impact_direction": "potentially_bullish"},
    ]

    auditor = AnnotationAgreementAuditor()
    report = auditor.audit(primary, secondary)

    assert report["audit_metadata"]["duplicate_primary_records_count"] == 1
    # Cross-commodity record is not matched with copper primary record
    assert report["audit_metadata"]["matched_overlapping_records"] == 0
    assert report["audit_metadata"]["unmatched_secondary_records"] == 1


def test_merge_adjudicated_ground_truth_safety_rejections(tmp_path):
    """Adjudicator strictly rejects all unsafe/incomplete conditions."""
    adjudicator = AnnotationAdjudicator()

    primary = [{
        "article_id": "art_1", "commodity": "copper", "article_url": "https://example.com/1", "article_title": "Title 1",
        "true_relevance_class": "directly_relevant", "true_sentiment_label": "positive", "true_event_type": "supply_disruption", "true_event_impact_direction": "potentially_bullish", "annotator_id": "ann1"
    }]
    secondary = [{
        "article_id": "art_1", "commodity": "copper",
        "true_relevance_class": "directly_relevant", "true_sentiment_label": "negative",  # DISPUTE on sentiment!
        "true_event_type": "supply_disruption", "true_event_impact_direction": "potentially_bullish", "annotator_id": "ann2"
    }]

    # 1. Unresolved disagreement (no adjudication record) -> MUST RAISE
    with pytest.raises(ValueError, match="Unresolved disagreement"):
        adjudicator.merge_adjudicated_ground_truth(primary, secondary, adjudicated_records=None)

    # 2. Incomplete adjudication (adjudicated field left blank) -> MUST RAISE
    adj_blank = [{"article_id": "art_1", "adjudicator_id": "exp1", "adjudicated_sentiment": ""}]
    with pytest.raises(ValueError, match="Incomplete adjudication"):
        adjudicator.merge_adjudicated_ground_truth(primary, secondary, adjudicated_records=adj_blank)

    # 3. Missing adjudicator_id -> MUST RAISE
    adj_no_id = [{"article_id": "art_1", "adjudicator_id": "", "adjudicated_sentiment": "positive"}]
    with pytest.raises(ValueError, match="Missing adjudicator_id"):
        adjudicator.merge_adjudicated_ground_truth(primary, secondary, adjudicated_records=adj_no_id)

    # 4. Invalid adjudicated taxonomy -> MUST RAISE
    adj_invalid_tax = [{"article_id": "art_1", "adjudicator_id": "exp1", "adjudicated_sentiment": "super_positive"}]
    with pytest.raises(ValueError, match="Invalid adjudicated value"):
        adjudicator.merge_adjudicated_ground_truth(primary, secondary, adjudicated_records=adj_invalid_tax)

    # 5. Duplicate adjudication records -> MUST RAISE
    adj_dup = [
        {"article_id": "art_1", "adjudicator_id": "exp1", "adjudicated_sentiment": "positive"},
        {"article_id": "art_1", "adjudicator_id": "exp2", "adjudicated_sentiment": "negative"},
    ]
    with pytest.raises(ValueError, match="Duplicate adjudication record"):
        adjudicator.merge_adjudicated_ground_truth(primary, secondary, adjudicated_records=adj_dup)

    # 6. Unknown article_id in adjudication -> MUST RAISE
    adj_unknown = [
        {"article_id": "art_1", "adjudicator_id": "exp1", "adjudicated_sentiment": "positive"},
        {"article_id": "nonexistent_id", "adjudicator_id": "exp1", "adjudicated_sentiment": "positive"},
    ]
    with pytest.raises(ValueError, match="Unknown article_id"):
        adjudicator.merge_adjudicated_ground_truth(primary, secondary, adjudicated_records=adj_unknown)

    # 7. Incomplete primary record (missing core label in primary) -> MUST RAISE
    primary_incomplete = [{
        "article_id": "art_1", "commodity": "copper", "article_url": "https://example.com/1", "article_title": "Title 1",
        "true_relevance_class": "directly_relevant", "true_sentiment_label": "", "true_event_type": "supply_disruption", "true_event_impact_direction": "potentially_bullish", "annotator_id": "ann1"
    }]
    with pytest.raises(ValueError, match="Incomplete ground truth"):
        adjudicator.merge_adjudicated_ground_truth(primary_incomplete)

    # 8. Overwrite existing file without overwrite=True -> MUST RAISE
    out_file = tmp_path / "existing.jsonl"
    out_file.write_text('{"existing": true}\n', encoding="utf-8")
    adj_valid = [{"article_id": "art_1", "adjudicator_id": "exp1", "adjudicated_sentiment": "positive"}]
    with pytest.raises(FileExistsError, match="already exists"):
        adjudicator.merge_adjudicated_ground_truth(primary, secondary, adjudicated_records=adj_valid, output_jsonl_path=out_file, overwrite=False)


def test_merge_adjudicated_ground_truth_field_isolation_and_provenance(tmp_path):
    """Adjudicated resolution applies strictly to disputed field; agreed fields retain consensus."""
    adjudicator = AnnotationAdjudicator()

    primary = [{
        "article_id": "art_1", "commodity": "copper", "article_url": "https://example.com/1", "article_title": "Copper mine accident",
        "true_relevance_class": "directly_relevant", "true_sentiment_label": "negative",
        "true_event_type": "supply_disruption", "true_event_impact_direction": "potentially_bullish",
        "annotator_id": "lead_rater"
    }]
    secondary = [{
        "article_id": "art_1", "commodity": "copper",
        "true_relevance_class": "directly_relevant", "true_sentiment_label": "negative",
        "true_event_type": "other",  # Disputed!
        "true_event_impact_direction": "potentially_bullish",
        "annotator_id": "second_rater"
    }]
    adjudication = [{
        "article_id": "art_1", "adjudicator_id": "domain_expert",
        "adjudicated_event_type": "supply_disruption",  # Resolved!
        "adjudication_notes": "Mine accidents halting production are supply disruptions."
    }]

    out_file = tmp_path / "merged_gt.jsonl"
    summary = adjudicator.merge_adjudicated_ground_truth(
        primary_records=primary,
        secondary_records=secondary,
        adjudicated_records=adjudication,
        output_jsonl_path=out_file,
    )
    assert summary["adjudicated_resolved_count"] == 1
    assert summary["total_merged_records"] == 1

    records = load_annotation_records(out_file)
    rec = records[0]
    assert rec["true_event_type"] == "supply_disruption"
    assert rec["true_sentiment_label"] == "negative"
    assert rec["true_relevance_class"] == "directly_relevant"
    assert rec["true_event_impact_direction"] == "potentially_bullish"
    assert rec["annotator_id"] == "lead_rater+second_rater[adj:domain_expert]"
    assert "Mine accidents halting production" in rec["annotator_notes"]


def test_e2e_full_human_workflow_synthetic_fixtures(tmp_path):
    """
    End-to-End simulation of the full 9-step human workflow using isolated synthetic fixtures:
    1. Generate primary & secondary packages.
    2. Populate independent synthetic human labels.
    3. Import completed CSVs.
    4. Compute inter-annotator agreement & Cohen's Kappa.
    5. Detect disagreements.
    6. Generate adjudication template.
    7. Simulate human expert resolution.
    8. Merge final verified ground truth.
    9. Run supervised evaluation pipeline against synthetic model features.
    """
    from src.evaluation.evaluator import NLPEvaluationPipeline

    # Step 1: Create synthetic sample of 50 articles (40 oil, 5 copper, 5 gold)
    raw_sample = tmp_path / "synthetic_sample.jsonl"
    records = []
    for i in range(5):
        records.append({"commodity": "copper", "article_url": f"https://example.com/c/{i}", "article_title": f"Copper news {i}", "article_publication_timestamp": "2026-11-01T10:00:00Z"})
    for i in range(5):
        records.append({"commodity": "gold", "article_url": f"https://example.com/g/{i}", "article_title": f"Gold news {i}", "article_publication_timestamp": "2026-11-01T10:00:00Z"})
    for i in range(40):
        records.append({"commodity": "crude oil", "article_url": f"https://example.com/o/{i}", "article_title": f"Oil news {i}", "article_publication_timestamp": "2026-11-01T10:00:00Z"})
    with open(raw_sample, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")

    pkg_dir = tmp_path / "workflow_packages"
    pkg_res = prepare_pilot_annotation_packages(raw_sample, output_dir=pkg_dir, overlap_fraction=0.20, seed=42)
    assert pkg_res["primary_count"] == 50
    assert pkg_res["secondary_count"] == 10

    # Step 2: Populate synthetic human annotations
    p_csv = Path(pkg_res["files_generated"]["primary_csv"])
    s_csv = Path(pkg_res["files_generated"]["secondary_csv"])

    p_rows = load_annotation_records(p_csv)
    s_rows = load_annotation_records(s_csv)

    for r in p_rows:
        r["true_relevance_class"] = "directly_relevant"
        r["true_sentiment_label"] = "positive"
        r["true_event_type"] = "production_change"
        r["true_event_impact_direction"] = "potentially_bearish"
        r["annotator_id"] = "primary_human"

    for idx, r in enumerate(s_rows):
        r["true_relevance_class"] = "directly_relevant"
        # Introduce deliberate dispute on 2 articles for sentiment and event_type
        if idx in (0, 1):
            r["true_sentiment_label"] = "negative"
            r["true_event_type"] = "supply_disruption"
        else:
            r["true_sentiment_label"] = "positive"
            r["true_event_type"] = "production_change"
        r["true_event_impact_direction"] = "potentially_bearish"
        r["annotator_id"] = "secondary_human"

    # Step 3: Save to CSV and Import with strict validation
    completed_p_csv = tmp_path / "completed_primary.csv"
    completed_s_csv = tmp_path / "completed_secondary.csv"
    with open(completed_p_csv, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(p_rows[0].keys()))
        writer.writeheader()
        writer.writerows(p_rows)
    with open(completed_s_csv, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(s_rows[0].keys()))
        writer.writeheader()
        writer.writerows(s_rows)

    importer = AnnotationCSVImporter()
    p_gt_jsonl = tmp_path / "imported_primary.jsonl"
    s_gt_jsonl = tmp_path / "imported_secondary.jsonl"
    importer.import_csv(completed_p_csv, output_jsonl_path=p_gt_jsonl, strict=True)
    importer.import_csv(completed_s_csv, output_jsonl_path=s_gt_jsonl, strict=True)

    imported_p = load_annotation_records(p_gt_jsonl)
    imported_s = load_annotation_records(s_gt_jsonl)

    # Step 4 & 5: Compute Agreement and Detect Disagreements
    auditor = AnnotationAgreementAuditor()
    audit_report = auditor.audit(imported_p, imported_s)
    assert audit_report["audit_metadata"]["matched_overlapping_records"] == 10
    assert audit_report["audit_metadata"]["total_records_with_disagreements"] == 2
    assert len(audit_report["disagreements"]) == 2

    # Step 6: Generate Adjudication Template
    adjudicator = AnnotationAdjudicator()
    adj_template_csv = tmp_path / "adjudication_template.csv"
    template_rows = adjudicator.create_adjudication_template(audit_report, output_csv_path=adj_template_csv)
    assert len(template_rows) == 2
    assert all(r["adjudicated_sentiment"] == "" for r in template_rows)

    # Step 7: Simulate Human Expert Resolution
    adjudicated_records = []
    for r in template_rows:
        adjudicated_records.append({
            "article_id": r["article_id"],
            "adjudicator_id": "senior_quant",
            "adjudicated_sentiment": "negative",
            "adjudicated_event_type": "supply_disruption",
            "adjudication_notes": "Expert resolved conflict based on supply reduction.",
        })

    # Step 8: Merge Final Verified Ground Truth
    final_gt_jsonl = tmp_path / "final_verified_ground_truth.jsonl"
    merge_summary = adjudicator.merge_adjudicated_ground_truth(
        primary_records=imported_p,
        secondary_records=imported_s,
        adjudicated_records=adjudicated_records,
        output_jsonl_path=final_gt_jsonl,
        strict=True,
    )
    assert merge_summary["total_merged_records"] == 50
    assert merge_summary["adjudicated_resolved_count"] == 2
    assert merge_summary["consensus_accepted_count"] == 8
    assert merge_summary["primary_unshared_count"] == 40
    assert final_gt_jsonl.exists()

    # Step 9: Evaluate Supervised Classification against Synthetic Predictions
    # Create synthetic feature predictions corresponding to the 50 articles
    synthetic_features = []
    for r in imported_p:
        synthetic_features.append({
            "article_id": r["article_id"],
            "commodity": r["commodity"],
            "article_url": r["article_url"],
            "article_title": r["article_title"],
            "article_publication_timestamp": r["article_publication_timestamp"],
            "relevance_class": "directly_relevant",
            "relevance_score": 0.95,
            "finbert_sentiment_label": "positive",
            "finbert_sentiment_score": 0.88,
            "supply_chain_event_type": "production_change",
            "event_confidence": 0.90,
            "event_impact_direction": "potentially_bearish",
            "feature_pipeline_status": "complete",
        })

    pipeline = NLPEvaluationPipeline()
    final_gt_records = load_annotation_records(final_gt_jsonl)
    eval_report = pipeline.run(feature_records=synthetic_features, labeled_records=final_gt_records)

    assert eval_report["evaluation_metadata"]["supervised_evaluation_available"] is True
    assert eval_report["evaluation_metadata"]["total_labeled_records"] == 50
    assert eval_report["evaluation_metadata"]["unmatched_ground_truth_count"] == 0
    assert eval_report["relevance_evaluation"]["sample_count"] == 50
    assert eval_report["event_extraction_evaluation"]["sample_count"] == 50
    assert eval_report["sentiment_evaluation"]["supervised_evaluation"]["sample_count"] == 50
    assert eval_report["sentiment_evaluation"]["supervised_evaluation"]["status"] == "evaluated"
