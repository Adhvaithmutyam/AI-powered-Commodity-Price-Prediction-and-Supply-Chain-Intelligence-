"""
Command-Line Interface for the NLP Evaluation Framework.

Provides subcommands for data quality auditing, human annotation sampling,
and supervised classification evaluation.
"""

import argparse
from datetime import datetime, timezone
import json
import logging
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional

from src.features.builder import load_jsonl_records
from src.features.aggregator import parse_timestamp_utc
from src.evaluation.config import EvaluationConfig
from src.evaluation.data_quality import DataQualityValidator
from src.evaluation.annotation import generate_annotation_sample, AnnotationCSVImporter
from src.evaluation.evaluator import NLPEvaluationPipeline

logger = logging.getLogger("nlp_evaluation")


def _print_banner(title: str) -> None:
    print("\n" + "=" * 70)
    print(f" {title}")
    print("=" * 70)


def _format_quality_summary(report: Dict[str, Any]) -> str:
    meta = report.get("evaluation_metadata", {})
    vol = report.get("volume_and_commodities", {})
    ident = report.get("identity_and_integrity", {})
    temp = report.get("temporal_validity", {})
    conf = report.get("confidence_value_validity", {})
    dist = report.get("distributions", {})
    miss = report.get("missingness_and_coverage", {})
    tax = report.get("taxonomy_violations", {})

    lines = []
    lines.append(f"Total Records Evaluated:      {meta.get('total_records_evaluated', 0)}")
    lines.append(f"Valid Records Count:          {meta.get('valid_records_count', 0)}")
    lines.append(f"Invalid Records Count:        {meta.get('invalid_records_count', 0)}")
    lines.append(f"Data-Quality Pass Rate:       {meta.get('data_quality_pass_rate', 0.0) * 100:.2f}%")
    lines.append(f"Evaluation Cutoff Applied:    {meta.get('evaluation_cutoff_applied', 'None')}")
    lines.append("-" * 70)
    lines.append("Commodity Distribution:")
    for comm, cnt in vol.get("counts_by_commodity", {}).items():
        lines.append(f"  • {comm:12s}: {cnt:5d}")
    if vol.get("unsupported_commodity_counts"):
        lines.append(f"  • Unsupported commodities: {vol.get('unsupported_commodity_counts')}")

    lines.append("-" * 70)
    lines.append("Integrity & Temporal Checks:")
    lines.append(f"  • Duplicate Identities:     {ident.get('duplicate_identities_count', 0)}")
    lines.append(f"  • Malformed Records:        {ident.get('malformed_records_count', 0)}")
    lines.append(f"  • Missing URLs:             {report.get('url_validity', {}).get('missing_urls_count', 0)}")
    lines.append(f"  • Invalid URLs:             {report.get('url_validity', {}).get('invalid_urls_count', 0)}")
    lines.append(f"  • Missing Timestamps:       {temp.get('missing_timestamps_count', 0)}")
    lines.append(f"  • Unparseable Timestamps:   {temp.get('unparseable_timestamps_count', 0)}")
    lines.append(f"  • Future Timestamps:        {temp.get('future_timestamps_count', 0)}")
    lines.append(f"  • Invalid Relevance Scores: {conf.get('invalid_relevance_scores_count', 0)}")
    lines.append(f"  • Invalid Sentiment Scores: {conf.get('invalid_sentiment_scores_count', 0)}")
    lines.append(f"  • Invalid Event Confidences:{conf.get('invalid_event_confidences_count', 0)}")

    lines.append("-" * 70)
    lines.append("Feature Completeness & Missingness:")
    lines.append(f"  • Missing Sentiment Steps:  {miss.get('missing_sentiment_count', 0)} ({miss.get('missing_sentiment_rate', 0.0) * 100:.1f}%)")
    lines.append(f"  • Missing Event Steps:      {miss.get('missing_event_extraction_count', 0)} ({miss.get('missing_event_extraction_rate', 0.0) * 100:.1f}%)")
    lines.append(f"  • Rule Non-Match (event='other'): {miss.get('no_event_rule_match_count', 0)} ({miss.get('no_event_rule_match_rate', 0.0) * 100:.1f}%)")

    lines.append("-" * 70)
    lines.append("Relevance Class Distribution:")
    for rel, data in dist.get("relevance_classes", {}).items():
        lines.append(f"  • {rel:22s}: {data.get('count', 0):5d} ({data.get('percentage', 0.0):5.1f}%)")

    lines.append("-" * 70)
    lines.append("Sentiment Label Distribution:")
    for sent, data in dist.get("sentiment_labels", {}).items():
        lines.append(f"  • {sent:22s}: {data.get('count', 0):5d} ({data.get('percentage', 0.0):5.1f}%)")

    lines.append("-" * 70)
    lines.append("Event Type Distribution:")
    for ev, data in dist.get("event_types", {}).items():
        lines.append(f"  • {ev:22s}: {data.get('count', 0):5d} ({data.get('percentage', 0.0):5.1f}%)")

    return "\n".join(lines)


def _format_evaluation_summary(report: Dict[str, Any]) -> str:
    lines = []
    meta = report.get("evaluation_metadata", {})
    avail = meta.get("supervised_evaluation_available", False)

    if not avail:
        lines.append("SUPERVISED EVALUATION STATUS: UNAVAILABLE")
        lines.append(f"Reason: {meta.get('reason', 'No ground-truth dataset supplied.')}")
        lines.append("\nNote: Data-quality validation and empirical prediction distributions are included below.")
        lines.append(_format_quality_summary(report.get("data_quality", {})))
        return "\n".join(lines)

    lines.append(f"Total Labeled Records:        {meta.get('total_labeled_records', 0)}")
    if meta.get("evaluated_tasks"):
        lines.append(f"Evaluated Pipeline Tasks:     {', '.join(meta.get('evaluated_tasks', []))}")
    if meta.get("unmatched_ground_truth_count", 0) > 0:
        lines.append(f"Unmatched Ground Truth:       {meta.get('unmatched_ground_truth_count')}")
    if meta.get("unmatched_predictions_count", 0) > 0:
        lines.append(f"Unmatched Predictions:        {meta.get('unmatched_predictions_count')}")
    if meta.get("duplicate_ground_truth_count", 0) > 0:
        lines.append(f"Duplicate Ground Truth Items: {meta.get('duplicate_ground_truth_count')}")
    if meta.get("sample_size_warning"):
        lines.append(f"WARNING: {meta.get('sample_size_warning')}")

    # Relevance Summary
    rel = report.get("relevance_evaluation", {})
    if rel.get("status") == "evaluated":
        rm = rel.get("metrics", {})
        lines.append("\n" + "-" * 70)
        lines.append("RELEVANCE CLASSIFICATION PERFORMANCE:")
        lines.append(f"  • Overall Accuracy:         {rm.get('overall_accuracy', 0.0) * 100:.2f}%")
        lines.append(f"  • Macro F1 Score:           {rm.get('macro_f1', 0.0) * 100:.2f}%")
        lines.append(f"  • Weighted F1 Score:        {rm.get('weighted_f1', 0.0) * 100:.2f}%")
        lines.append("\n  Per-Class Relevance Metrics:")
        lines.append(f"    {'Class':22s} | {'Precision':9s} | {'Recall':9s} | {'F1':9s} | {'Support':7s}")
        lines.append("    " + "-" * 64)
        for c, m in rm.get("per_class", {}).items():
            lines.append(
                f"    {c:22s} | {m.get('precision', 0.0)*100:8.2f}% | "
                f"{m.get('recall', 0.0)*100:8.2f}% | {m.get('f1_score', 0.0)*100:8.2f}% | "
                f"{m.get('support', 0):7d}"
            )

    # Event Extraction Summary
    ev = report.get("event_extraction_evaluation", {})
    if ev.get("status") == "evaluated":
        em = ev.get("classification_metrics", {})
        cov = ev.get("coverage_analysis", {})
        cond = cov.get("conditional_event_type_classification", {})
        lines.append("\n" + "-" * 70)
        lines.append("SUPPLY-CHAIN EVENT EXTRACTION PERFORMANCE:")
        lines.append(f"  • Overall 7-Class Acc:      {em.get('overall_accuracy', 0.0) * 100:.2f}%")
        lines.append(f"  • Macro F1 (all types):     {em.get('macro_f1', 0.0) * 100:.2f}%")
        lines.append(f"  • Binary Event Detection Recall:   {cov.get('event_detection_recall_rate', 0.0) * 100:.2f}%")
        lines.append(f"  • Binary Event Detection Precision:{cov.get('event_detection_precision_rate', 0.0) * 100:.2f}%")
        lines.append(f"  • Binary Event Detection F1:       {cov.get('event_detection_f1_score', 0.0) * 100:.2f}%")
        lines.append(f"  • Conditional Event-Type Accuracy: {cond.get('conditional_type_accuracy', 0.0) * 100:.2f}% (on {cond.get('evaluated_instances_tp_det', 0)} detected events)")
        lines.append(f"  • True Events in Sample:    {cov.get('true_event_count', 0)}")
        lines.append(f"  • Events Extracted by Model:{cov.get('extracted_event_count', 0)}")
        lines.append(f"  • Missed Events (FN):       {cov.get('missed_events_count', 0)}")
        lines.append(f"  • False Alarms (FP):        {cov.get('false_alarms_count', 0)}")
        lines.append(f"  • Type Confusions:          {cov.get('type_confusions_count', 0)}")

    # Sentiment Summary
    sent = report.get("sentiment_evaluation", {})
    lines.append("\n" + "-" * 70)
    lines.append("FINANCIAL SENTIMENT ANALYSIS:")
    conf = sent.get("confidence_statistics", {})
    lines.append(f"  • Mean Model Confidence:    {conf.get('mean', 0.0):.4f}")
    lines.append(f"  • Confidence Std Deviation: {conf.get('std', 0.0):.4f}")
    lines.append(f"  • Confidence Range:         [{conf.get('min', 0.0):.4f}, {conf.get('max', 0.0):.4f}]")

    sup_sent = sent.get("supervised_evaluation", {})
    if sup_sent.get("status") == "evaluated":
        sm = sup_sent.get("metrics", {})
        lines.append(f"  • Supervised Accuracy:      {sm.get('overall_accuracy', 0.0) * 100:.2f}%")
        lines.append(f"  • Supervised Macro F1:      {sm.get('macro_f1', 0.0) * 100:.2f}%")
    else:
        lines.append(f"  • Supervised Evaluation:    {sup_sent.get('status', 'skipped')}")
        lines.append(f"    ({sup_sent.get('message', '')})")

    return "\n".join(lines)


def _format_csv_import_summary(report: Dict[str, Any]) -> str:
    lines = []
    comp = report.get("completion_summary", {})
    dups = report.get("duplicate_identities", {})
    tax = report.get("taxonomy_violations", {})
    mal = report.get("malformed_rows", {})

    lines.append(f"Source CSV Path:              {report.get('csv_path')}")
    lines.append(f"Target Ground-Truth JSONL:    {report.get('output_jsonl_path')}")
    lines.append(f"Total Rows Read:              {report.get('total_rows_read', 0)}")
    lines.append(f"Valid Records Converted:      {report.get('valid_records_count', 0)}")
    lines.append(f"Import Status:                {report.get('status')}")
    lines.append("-" * 70)
    lines.append("Annotation Completion Status:")
    lines.append(f"  • Fully Labeled Records:    {comp.get('fully_labeled_count', 0)} ({comp.get('fully_labeled_rate', 0.0)*100:.1f}%)")
    lines.append(f"  • Partially Labeled Records:{comp.get('partially_labeled_count', 0)} ({comp.get('partially_labeled_rate', 0.0)*100:.1f}%)")
    lines.append(f"  • Unlabeled Records:        {comp.get('unlabeled_count', 0)} ({comp.get('unlabeled_rate', 0.0)*100:.1f}%)")
    lines.append(f"  • Relevance Labeled:        {comp.get('relevance_labeled_count', 0)}")
    lines.append(f"  • Sentiment Labeled:        {comp.get('sentiment_labeled_count', 0)}")
    lines.append(f"  • Events Labeled:           {comp.get('events_labeled_count', 0)}")
    lines.append(f"  • Impact Direction Labeled: {comp.get('impact_direction_labeled_count', 0)}")
    lines.append("-" * 70)
    lines.append("Quality & Integrity Audits:")
    lines.append(f"  • Duplicate Identities:     {dups.get('duplicate_count', 0)}")
    lines.append(f"  • Taxonomy Violations:      {tax.get('violation_count', 0)}")
    lines.append(f"  • Malformed CSV Rows:       {mal.get('malformed_count', 0)}")

    if tax.get("violation_samples"):
        lines.append("  • Taxonomy Violation Samples:")
        for s in tax.get("violation_samples", [])[:5]:
            lines.append(f"      Row {s.get('row_number')}: {s.get('field')}='{s.get('value')}' (allowed: {s.get('allowed')})")

    if dups.get("duplicate_samples"):
        lines.append("  • Duplicate Identity Samples:")
        for d in dups.get("duplicate_samples", [])[:5]:
            lines.append(f"      Row {d.get('row_number')}: {d.get('commodity')} - {d.get('article_url')}")

    return "\n".join(lines)


def run_cli(args: Optional[List[str]] = None) -> int:
    """Main CLI runner parsing arguments and delegating to evaluation components."""
    parser = argparse.ArgumentParser(
        description="NLP Evaluation Framework and Data-Quality Validation CLI.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--mode",
        choices=["validate", "sample", "evaluate", "import-csv"],
        default="validate",
        help="Operational mode: 'validate' data quality, 'sample' for annotation, 'evaluate' labeled data, 'import-csv' to convert annotated CSV.",
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/processed/nlp_article_features.jsonl"),
        help="Path to input feature, news JSONL, or annotated CSV file.",
    )
    parser.add_argument(
        "--ground-truth",
        type=Path,
        default=None,
        help="Path to human-labeled JSONL file (used in 'evaluate' mode).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Path to output report (JSON), sample template (JSONL), or converted ground truth (JSONL).",
    )
    parser.add_argument(
        "--cutoff",
        type=str,
        default=None,
        help="Optional ISO timestamp cutoff for detecting future publication dates.",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=50,
        help="Sample size for human annotation template generation.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for deterministic sampling.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        default=False,
        help="Allow overwriting existing output files without prompting.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        default=False,
        help="Strict mode: reject CSV conversion if taxonomy violations or duplicates exist.",
    )

    parsed = parser.parse_args(args)

    # 1. Mode: Data-Quality Validation
    if parsed.mode == "validate":
        _print_banner(f"NLP Data Quality Validation: {parsed.input}")
        if not parsed.input.exists():
            print(f"Error: Input file '{parsed.input}' does not exist.")
            return 1

        cutoff_dt = parse_timestamp_utc(parsed.cutoff) if parsed.cutoff else None
        records = load_jsonl_records(parsed.input)
        validator = DataQualityValidator()
        report = validator.validate(records, evaluation_cutoff=cutoff_dt)

        output_path = parsed.output or Path("data/evaluation/data_quality_report.json")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)

        print(_format_quality_summary(report))
        print("-" * 70)
        print(f"Full structured data quality report saved to: {output_path}")
        return 0

    # 2. Mode: Annotation Sample Generation
    elif parsed.mode == "sample":
        _print_banner(f"Human Annotation Sample Generation: {parsed.input}")
        if not parsed.input.exists():
            print(f"Error: Input file '{parsed.input}' does not exist.")
            return 1

        output_path = parsed.output or Path("data/evaluation/annotation_sample.jsonl")
        samples = generate_annotation_sample(
            input_path=parsed.input,
            output_path=output_path,
            sample_size=parsed.sample_size,
            seed=parsed.seed,
            stratify=True,
            include_guidelines=True,
        )

        print(f"Generated {len(samples)} unbiased annotation records.")
        print(f"Output JSONL: {output_path if output_path.suffix.lower() != '.csv' else output_path.with_suffix('.jsonl')}")
        print(f"Output CSV:   {output_path.with_suffix('.csv') if output_path.suffix.lower() != '.csv' else output_path}")
        print(f"Guidelines:   {output_path.parent / 'ANNOTATION_GUIDELINES.md'}")
        print("\nArticle distribution by commodity:")
        counts = {}
        for s in samples:
            counts[s['commodity']] = counts.get(s['commodity'], 0) + 1
        for comm, cnt in counts.items():
            print(f"  • {comm:12s}: {cnt:4d}")
        return 0

    # 3. Mode: Supervised Evaluation
    elif parsed.mode == "evaluate":
        _print_banner(f"NLP Classification Performance Evaluation: {parsed.input}")
        if not parsed.input.exists():
            print(f"Error: Input feature file '{parsed.input}' does not exist.")
            return 1

        feature_records = load_jsonl_records(parsed.input)
        labeled_records = None
        if parsed.ground_truth:
            if not parsed.ground_truth.exists():
                print(f"Error: Ground-truth file '{parsed.ground_truth}' does not exist.")
                return 1
            labeled_records = load_jsonl_records(parsed.ground_truth)

        cutoff_dt = parse_timestamp_utc(parsed.cutoff) if parsed.cutoff else None
        pipeline = NLPEvaluationPipeline()
        report = pipeline.run(
            feature_records=feature_records,
            labeled_records=labeled_records,
            evaluation_cutoff=cutoff_dt,
        )

        output_path = parsed.output or Path("data/evaluation/classification_report.json")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)

        print(_format_evaluation_summary(report))
        print("-" * 70)
        print(f"Full structured evaluation report saved to: {output_path}")
        return 0

    # 4. Mode: Import & Validate Human-Annotated CSV
    elif parsed.mode == "import-csv":
        _print_banner(f"Import Human-Annotated CSV to JSONL: {parsed.input}")
        if not parsed.input.exists():
            print(f"Error: Input CSV file '{parsed.input}' does not exist.")
            return 1

        output_path = parsed.output or Path("data/evaluation/human_ground_truth.jsonl")
        importer = AnnotationCSVImporter()
        try:
            report = importer.import_csv(
                csv_path=parsed.input,
                output_jsonl_path=output_path,
                overwrite=parsed.overwrite,
                strict=parsed.strict,
            )
        except Exception as e:
            print(f"Error: CSV import failed: {e}")
            return 1

        print(_format_csv_import_summary(report))
        print("-" * 70)
        print(f"Converted {report['valid_records_count']} ground-truth records saved to: {output_path}")
        return 0

    return 0
