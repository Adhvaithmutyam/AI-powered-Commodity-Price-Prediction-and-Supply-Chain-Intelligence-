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
