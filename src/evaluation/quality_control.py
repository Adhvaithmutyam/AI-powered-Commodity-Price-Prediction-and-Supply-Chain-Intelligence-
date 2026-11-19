"""
Quality Control, Inter-Annotator Agreement, and Adjudication Framework.

Provides mathematical and procedural tools to:
1. Audit independent multi-annotator datasets on overlapping articles.
2. Compute exact Cohen's Kappa (kappa) and observed agreement percentages with
   strict degenerate zero-variance handling (0/0 indeterminate protection).
3. Detect, categorize, and extract annotation disagreements without AI intervention.
4. Generate human adjudication templates for senior domain experts.
5. Merge consensus annotations and human-adjudicated resolutions into final ground truth
   with strict validation against silent adoption of primary annotations on unresolved disputes.
"""

from collections import Counter, defaultdict
from datetime import datetime, timezone
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple
import csv

from src.features.builder import canonicalize_url, load_jsonl_records
from src.evaluation.annotation import (
    sanitize_for_spreadsheet,
    desanitize_from_spreadsheet,
    _generate_deterministic_id,
    AnnotationCSVImporter,
)
from src.evaluation.config import (
    EvaluationConfig,
    VALID_COMMODITIES,
    VALID_RELEVANCE_CLASSES,
    VALID_SENTIMENT_LABELS,
    VALID_EVENT_TYPES,
    VALID_IMPACT_DIRECTIONS,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Exact Mathematical Computation of Cohen's Kappa
# ---------------------------------------------------------------------------

def compute_cohens_kappa(
    y1: List[str],
    y2: List[str],
    categories: Optional[List[str]] = None,
) -> Optional[float]:
    """
    Compute exact Cohen's Kappa coefficient between two independent raters.

    Formula:
        kappa = (P_o - P_e) / (1 - P_e)
    where:
        P_o = observed proportional agreement
        P_e = hypothetical chance agreement based on marginal totals

    Args:
        y1: Labels assigned by rater 1.
        y2: Labels assigned by rater 2.
        categories: Optional fixed category list. If None, derived from union of observed labels.

    Returns:
        Cohen's Kappa score bounded in [-1.0, 1.0], rounded to 4 decimal places,
        or None when mathematically undefined (e.g. 0/0 when P_o = P_e = 1.0 due to zero variance).
    """
    if len(y1) != len(y2) or len(y1) == 0:
        return None

    n = len(y1)
    cats = list(categories) if categories is not None else sorted(list(set(y1) | set(y2)))
    if not cats:
        return None

    # Build contingency matrix M[i][j]: count where rater 1 chose cats[i] and rater 2 chose cats[j]
    cat_to_idx = {c: i for i, c in enumerate(cats)}
    matrix = [[0] * len(cats) for _ in range(len(cats))]

    for a, b in zip(y1, y2):
        if a in cat_to_idx and b in cat_to_idx:
            matrix[cat_to_idx[a]][cat_to_idx[b]] += 1

    # Observed agreement: sum of diagonal entries / n
    agreed = sum(matrix[i][i] for i in range(len(cats)))
    p_o = agreed / n

    # Expected chance agreement: sum of (marginal_1 * marginal_2) / n^2
    marginal_1 = [sum(matrix[i][j] for j in range(len(cats))) for i in range(len(cats))]
    marginal_2 = [sum(matrix[i][j] for i in range(len(cats))) for j in range(len(cats))]

    p_e = sum(m1 * m2 for m1, m2 in zip(marginal_1, marginal_2)) / (n * n)

    # Degenerate case: when expected chance agreement equals 1.0, denominator (1 - P_e) is 0.
    # When both raters assign the exact same single category to every record,
    # P_o = 1.0 and P_e = 1.0 -> (1 - 1) / (1 - 1) = 0/0, which is mathematically undefined.
    # We return None to distinguish this degenerate zero-variance scenario.
    if abs(1.0 - p_e) < 1e-9:
        return None

    kappa = (p_o - p_e) / (1.0 - p_e)
    # Clamp to theoretical bounds
    kappa = max(-1.0, min(1.0, kappa))
    return round(kappa, 4)


def interpret_cohens_kappa(kappa: Optional[float]) -> str:
    """
    Return Landis & Koch (1977) benchmark interpretation for a Cohen's Kappa score.
    Note: Landis & Koch benchmarks are purely descriptive conventions and heuristics,
    not statistical proof of annotation reliability or significance.
    """
    if kappa is None:
        return "undefined_constant_distribution"
    elif kappa >= 0.81:
        return "almost_perfect_agreement"
    elif kappa >= 0.61:
        return "substantial_agreement"
    elif kappa >= 0.41:
        return "moderate_agreement"
    elif kappa >= 0.21:
        return "fair_agreement"
    elif kappa >= 0.01:
        return "slight_agreement"
    elif kappa == 0.0:
        return "chance_agreement"
    else:
        return "poor_systematic_disagreement"


# ---------------------------------------------------------------------------
# Inter-Annotator Agreement Auditor
# ---------------------------------------------------------------------------

class AnnotationAgreementAuditor:
    """
    Audits pair-wise agreement across multiple independent annotators on an overlapping subset.
    Flags duplicate identities, missing/invalid labels, computes Cohen's Kappa and per-label
    specific agreements, and extracts itemized disagreements without AI intervention.
    """

    TASKS = (
        ("relevance", "true_relevance_class", VALID_RELEVANCE_CLASSES),
        ("sentiment", "true_sentiment_label", VALID_SENTIMENT_LABELS),
        ("event_type", "true_event_type", VALID_EVENT_TYPES),
        ("impact_direction", "true_event_impact_direction", VALID_IMPACT_DIRECTIONS),
    )

    def __init__(self, config: Optional[EvaluationConfig] = None) -> None:
        self.config = config or EvaluationConfig()

    def audit(
        self,
        primary_records: List[Dict[str, Any]],
        secondary_records: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        Audit agreement between primary and secondary annotation datasets.

        Args:
            primary_records: Full list of primary annotations.
            secondary_records: Overlapping subset annotated by secondary rater.

        Returns:
            Structured dictionary of agreement metrics, confusion matrices, and disagreement cases.
        """
        # 1. Audit Primary Dataset Identities & Duplicates
        p_by_id: Dict[str, Dict[str, Any]] = {}
        p_by_key: Dict[Tuple[str, str], Dict[str, Any]] = {}
        duplicate_primary: List[Dict[str, Any]] = []
        seen_p_ids: Set[str] = set()
        seen_p_keys: Set[Tuple[str, str]] = set()

        for idx, r in enumerate(primary_records):
            art_id = str(r.get("article_id") or "").strip()
            comm = str(r.get("commodity") or "").strip().lower()
            url = canonicalize_url(r.get("article_url") or r.get("url") or "")

            key = (comm, url)
            if art_id in seen_p_ids or (comm and url and key in seen_p_keys):
                duplicate_primary.append({
                    "record_index": idx,
                    "article_id": art_id,
                    "commodity": comm,
                    "article_url": url,
                })
            else:
                if art_id:
                    seen_p_ids.add(art_id)
                    p_by_id[art_id] = r
                if comm and url:
                    seen_p_keys.add(key)
                    p_by_key[key] = r

        # 2. Audit Secondary Dataset Identities & Duplicates
        seen_s_ids: Set[str] = set()
        seen_s_keys: Set[Tuple[str, str]] = set()
        duplicate_secondary: List[Dict[str, Any]] = []
        matched_pairs: List[Tuple[Dict[str, Any], Dict[str, Any]]] = []
        unmatched_secondary: List[Dict[str, Any]] = []

        for idx, sec in enumerate(secondary_records):
            s_id = str(sec.get("article_id") or "").strip()
            s_comm = str(sec.get("commodity") or "").strip().lower()
            s_url = canonicalize_url(sec.get("article_url") or sec.get("url") or "")
            s_key = (s_comm, s_url)

            if s_id in seen_s_ids or (s_comm and s_url and s_key in seen_s_keys):
                duplicate_secondary.append({
                    "record_index": idx,
                    "article_id": s_id,
                    "commodity": s_comm,
                    "article_url": s_url,
                })
                continue
            if s_id:
                seen_s_ids.add(s_id)
            if s_comm and s_url:
                seen_s_keys.add(s_key)

            # Match primary record by ID or composite key, ensuring commodity matches strictly
            match = None
            if s_id and s_id in p_by_id:
                candidate = p_by_id[s_id]
                # Enforce commodity alignment so cross-commodity shared URLs are never mixed
                if str(candidate.get("commodity") or "").strip().lower() == s_comm:
                    match = candidate
            elif s_key in p_by_key:
                match = p_by_key[s_key]

            if match:
                matched_pairs.append((match, sec))
            else:
                unmatched_secondary.append(sec)

        num_matched = len(matched_pairs)
        task_metrics: Dict[str, Any] = {}
        disagreements_by_record: List[Dict[str, Any]] = []

        # 3. Compute Per-Task Agreement & Quality
        for task_name, field_key, valid_cats in self.TASKS:
            valid_cats_sorted = sorted(list(valid_cats))
            y1_valid: List[str] = []
            y2_valid: List[str] = []
            missing_or_invalid: List[Dict[str, Any]] = []

            for p_rec, s_rec in matched_pairs:
                art_id = p_rec.get("article_id") or s_rec.get("article_id") or ""
                v1 = str(p_rec.get(field_key) or "").strip().lower()
                v2 = str(s_rec.get(field_key) or "").strip().lower()

                # Validate paired human labels strictly
                is_valid_1 = v1 in valid_cats
                is_valid_2 = v2 in valid_cats

                if is_valid_1 and is_valid_2:
                    y1_valid.append(v1)
                    y2_valid.append(v2)
                else:
                    missing_or_invalid.append({
                        "article_id": art_id,
                        "primary_value": v1,
                        "secondary_value": v2,
                        "primary_valid": is_valid_1,
                        "secondary_valid": is_valid_2,
                    })

            valid_paired_count = len(y1_valid)
            if valid_paired_count > 0:
                agreed = sum(1 for a, b in zip(y1_valid, y2_valid) if a == b)
                disagreed = valid_paired_count - agreed
                obs_rate = round(agreed / valid_paired_count, 4)
                kappa = compute_cohens_kappa(y1_valid, y2_valid, categories=valid_cats_sorted)
                interp = interpret_cohens_kappa(kappa)

                # Compute per-label specific agreement: P_pos(c) = 2 * A_c / (R1_c + R2_c)
                per_label: Dict[str, Any] = {}
                c1_counts = Counter(y1_valid)
                c2_counts = Counter(y2_valid)
                for cat in valid_cats_sorted:
                    a_c = sum(1 for a, b in zip(y1_valid, y2_valid) if a == cat and b == cat)
                    r1_c = c1_counts.get(cat, 0)
                    r2_c = c2_counts.get(cat, 0)
                    spec_rate = round((2.0 * a_c) / (r1_c + r2_c), 4) if (r1_c + r2_c) > 0 else None
                    per_label[cat] = {
                        "rater1_count": r1_c,
                        "rater2_count": r2_c,
                        "agreed_count": a_c,
                        "specific_agreement_rate": spec_rate,
                    }

                task_metrics[task_name] = {
                    "valid_paired_evaluations": valid_paired_count,
                    "agreed_records": agreed,
                    "disagreed_records": disagreed,
                    "observed_agreement_rate": obs_rate,
                    "cohens_kappa": kappa,
                    "interpretation": interp,
                    "interpretation_caveat": "Landis-Koch benchmark is a descriptive heuristic, not statistical proof of model or annotation reliability.",
                    "missing_or_invalid_count": len(missing_or_invalid),
                    "missing_or_invalid_samples": missing_or_invalid[:5],
                    "per_label_agreement": per_label,
                }
            else:
                task_metrics[task_name] = {
                    "valid_paired_evaluations": 0,
                    "agreed_records": 0,
                    "disagreed_records": 0,
                    "observed_agreement_rate": 0.0,
                    "cohens_kappa": None,
                    "interpretation": "no_valid_paired_annotations",
                    "interpretation_caveat": "No complete paired annotations exist for this task.",
                    "missing_or_invalid_count": len(missing_or_invalid),
                    "missing_or_invalid_samples": missing_or_invalid[:5],
                    "per_label_agreement": {},
                }

        # 4. Identify Detailed Itemized Disagreements
        for p_rec, s_rec in matched_pairs:
            art_id = p_rec.get("article_id") or s_rec.get("article_id") or ""
            comm = p_rec.get("commodity") or s_rec.get("commodity") or ""
            title = p_rec.get("article_title") or s_rec.get("article_title") or ""
            url = p_rec.get("article_url") or s_rec.get("article_url") or ""

            disagreed_fields: List[str] = []
            field_values: Dict[str, Dict[str, str]] = {}

            for task_name, field_key, _ in self.TASKS:
                v1 = str(p_rec.get(field_key) or "").strip().lower()
                v2 = str(s_rec.get(field_key) or "").strip().lower()
                if v1 != v2:
                    disagreed_fields.append(field_key)
                    field_values[field_key] = {
                        "primary": v1,
                        "secondary": v2,
                    }

            if disagreed_fields:
                disagreements_by_record.append({
                    "article_id": art_id,
                    "commodity": comm,
                    "article_title": title,
                    "article_url": url,
                    "primary_annotator_id": p_rec.get("annotator_id", ""),
                    "secondary_annotator_id": s_rec.get("annotator_id", ""),
                    "disagreed_fields": disagreed_fields,
                    "field_values": field_values,
                })

        # 5. Sample Size Disclosure
        sample_warning = None
        if num_matched < 30:
            sample_warning = (
                f"Small overlap sample size (N={num_matched}). This overlap is a preliminary "
                f"calibration pilot and is not statistically conclusive. Binomial variance and "
                f"confidence intervals for Kappa are wide."
            )

        return {
            "audit_metadata": {
                "audit_timestamp": datetime.now(timezone.utc).isoformat(),
                "total_primary_records": len(primary_records),
                "total_secondary_records": len(secondary_records),
                "matched_overlapping_records": num_matched,
                "unmatched_secondary_records": len(unmatched_secondary),
                "total_records_with_disagreements": len(disagreements_by_record),
                "duplicate_primary_records_count": len(duplicate_primary),
                "duplicate_secondary_records_count": len(duplicate_secondary),
                "sample_size_warning": sample_warning,
            },
            "task_agreements": task_metrics,
            "disagreements": disagreements_by_record,
        }


# ---------------------------------------------------------------------------
# Human Adjudication Template Generator and Dataset Merger
# ---------------------------------------------------------------------------

class AnnotationAdjudicator:
    """
    Produces human adjudication spreadsheets for resolving annotator disagreements,
    and merges consensus and adjudicated resolutions into final ground truth.
    Strictly prevents silent adoption of primary annotations on unresolved disputes.
    """

    ADJUDICATION_FIELDS = [
        "article_id",
        "commodity",
        "article_title",
        "article_url",
        "disagreed_fields",
        "primary_annotator_id",
        "secondary_annotator_id",
        "primary_relevance",
        "secondary_relevance",
        "adjudicated_relevance",
        "primary_sentiment",
        "secondary_sentiment",
        "adjudicated_sentiment",
        "primary_event_type",
        "secondary_event_type",
        "adjudicated_event_type",
        "primary_impact_direction",
        "secondary_impact_direction",
        "adjudicated_impact_direction",
        "adjudicator_id",
        "adjudication_notes",
    ]

    TASK_MAPPING = [
        ("true_relevance_class", "adjudicated_relevance", VALID_RELEVANCE_CLASSES),
        ("true_sentiment_label", "adjudicated_sentiment", VALID_SENTIMENT_LABELS),
        ("true_event_type", "adjudicated_event_type", VALID_EVENT_TYPES),
        ("true_event_impact_direction", "adjudicated_impact_direction", VALID_IMPACT_DIRECTIONS),
    ]

    def create_adjudication_template(
        self,
        agreement_audit: Dict[str, Any],
        output_csv_path: Optional[Path] = None,
    ) -> List[Dict[str, Any]]:
        """
        Generate a spreadsheet template containing exclusively the items where
        annotators had conflicting judgments, with blank adjudicated columns for human review.

        Args:
            agreement_audit: Audit report from AnnotationAgreementAuditor.audit.
            output_csv_path: Optional path to write Excel-friendly CSV.

        Returns:
            List of adjudication template row dictionaries.
        """
        disagreements = agreement_audit.get("disagreements", [])
        template_rows: List[Dict[str, Any]] = []

        for item in disagreements:
            fvals = item.get("field_values", {})
            row = {
                "article_id": item.get("article_id", ""),
                "commodity": item.get("commodity", ""),
                "article_title": item.get("article_title", ""),
                "article_url": item.get("article_url", ""),
                "disagreed_fields": ", ".join(item.get("disagreed_fields", [])),
                "primary_annotator_id": item.get("primary_annotator_id", ""),
                "secondary_annotator_id": item.get("secondary_annotator_id", ""),
                "primary_relevance": fvals.get("true_relevance_class", {}).get("primary", ""),
                "secondary_relevance": fvals.get("true_relevance_class", {}).get("secondary", ""),
                "adjudicated_relevance": "",
                "primary_sentiment": fvals.get("true_sentiment_label", {}).get("primary", ""),
                "secondary_sentiment": fvals.get("true_sentiment_label", {}).get("secondary", ""),
                "adjudicated_sentiment": "",
                "primary_event_type": fvals.get("true_event_type", {}).get("primary", ""),
                "secondary_event_type": fvals.get("true_event_type", {}).get("secondary", ""),
                "adjudicated_event_type": "",
                "primary_impact_direction": fvals.get("true_event_impact_direction", {}).get("primary", ""),
                "secondary_impact_direction": fvals.get("true_event_impact_direction", {}).get("secondary", ""),
                "adjudicated_impact_direction": "",
                "adjudicator_id": "",
                "adjudication_notes": "",
            }
            template_rows.append(row)

        if output_csv_path is not None:
            out_p = Path(output_csv_path)
            out_p.parent.mkdir(parents=True, exist_ok=True)
            with open(out_p, "w", encoding="utf-8-sig", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=self.ADJUDICATION_FIELDS, extrasaction="ignore")
                writer.writeheader()
                for r in template_rows:
                    sanitized_r = {k: sanitize_for_spreadsheet(v) for k, v in r.items()}
                    writer.writerow(sanitized_r)
            logger.info("Saved %d adjudication items to %s", len(template_rows), out_p)

        return template_rows

    def merge_adjudicated_ground_truth(
        self,
        primary_records: List[Dict[str, Any]],
        secondary_records: Optional[List[Dict[str, Any]]] = None,
        adjudicated_records: Optional[List[Dict[str, Any]]] = None,
        output_jsonl_path: Optional[Path] = None,
        overwrite: bool = False,
        strict: bool = True,
    ) -> Dict[str, Any]:
        """
        Merge primary consensus annotations and human-adjudicated conflict resolutions into
        a verified ground-truth JSONL dataset for downstream model evaluation.

        Strictly enforces:
        1. No unresolved disagreements can silently adopt primary annotator labels.
        2. All adjudicated values are validated against established taxonomies.
        3. Adjudicator IDs are required for audit trail.
        4. Duplicate or unknown adjudication records are detected and rejected.
        5. Final dataset retains all intended primary articles with complete valid annotations.

        Args:
            primary_records: Full set of primary annotations.
            secondary_records: Optional overlapping secondary annotations.
            adjudicated_records: Completed human adjudication records for conflicts.
            output_jsonl_path: Optional output JSONL destination.
            overwrite: Whether to overwrite existing destination file.
            strict: If True, raises ValueError on any unresolved dispute or invalid data.

        Returns:
            Structured summary of merged ground-truth dataset.
        """
        if output_jsonl_path is not None:
            out_p = Path(output_jsonl_path)
            if out_p.exists() and not overwrite:
                raise FileExistsError(
                    f"Destination ground truth file '{out_p}' already exists. Pass overwrite=True to replace."
                )

        # 1. Audit Primary Records and Build Lookup
        primary_by_id: Dict[str, Dict[str, Any]] = {}
        for idx, prec in enumerate(primary_records):
            aid = str(prec.get("article_id") or "").strip()
            if not aid:
                raise ValueError(f"Primary record at index {idx} has missing or empty 'article_id'.")
            if aid in primary_by_id:
                raise ValueError(f"Duplicate article_id '{aid}' detected in primary records.")
            primary_by_id[aid] = prec

        # 2. Audit Secondary Records and Build Lookup
        sec_by_id: Dict[str, Dict[str, Any]] = {}
        if secondary_records:
            for idx, sr in enumerate(secondary_records):
                aid = str(sr.get("article_id") or "").strip()
                if not aid:
                    raise ValueError(f"Secondary record at index {idx} has missing or empty 'article_id'.")
                if aid in sec_by_id:
                    raise ValueError(f"Duplicate article_id '{aid}' detected in secondary records.")
                if aid not in primary_by_id:
                    raise ValueError(f"Secondary article_id '{aid}' does not exist in primary dataset.")
                sec_by_id[aid] = sr

        # 3. Audit Adjudication Records and Validate Taxonomies
        adj_by_id: Dict[str, Dict[str, Any]] = {}
        if adjudicated_records:
            for idx, ar in enumerate(adjudicated_records):
                aid = str(ar.get("article_id") or "").strip()
                if not aid:
                    raise ValueError(f"Adjudication record at index {idx} has missing or empty 'article_id'.")
                if aid in adj_by_id:
                    raise ValueError(f"Duplicate adjudication record detected for article_id: '{aid}'.")
                if aid not in primary_by_id:
                    raise ValueError(f"Unknown article_id '{aid}' in adjudication records does not match any primary article.")

                # Validate adjudicator ID is present
                adj_id = str(ar.get("adjudicator_id") or "").strip()
                if not adj_id:
                    raise ValueError(f"Missing adjudicator_id for article '{aid}'. Adjudicator ID is required for audit trail.")

                # Validate non-empty adjudicated values against official taxonomies
                for raw_field, adj_field, valid_cats in self.TASK_MAPPING:
                    val = str(ar.get(adj_field) or "").strip().lower()
                    if val and val not in valid_cats:
                        raise ValueError(
                            f"Invalid adjudicated value '{val}' for field '{adj_field}' in article '{aid}'. "
                            f"Allowed: {sorted(list(valid_cats))}."
                        )

                adj_by_id[aid] = ar

        # 4. Process Each Primary Record and Resolve Overlaps
        merged_records: List[Dict[str, Any]] = []
        auto_accepted_consensus = 0
        adjudicated_resolved = 0
        primary_unshared = 0

        for prec in primary_records:
            aid = str(prec.get("article_id") or "").strip()
            comm = str(prec.get("commodity") or "").strip().lower()
            url = canonicalize_url(prec.get("article_url") or prec.get("url") or "")
            title = str(prec.get("article_title") or prec.get("title") or "").strip()

            final_values: Dict[str, str] = {}
            final_annotator = str(prec.get("annotator_id") or "").strip()
            final_notes = str(prec.get("annotator_notes") or "").strip()

            # Check if this article was in the secondary overlap
            if aid in sec_by_id:
                srec = sec_by_id[aid]
                disputed_in_article: List[str] = []

                # Audit each classification task between primary and secondary
                for raw_field, adj_field, valid_cats in self.TASK_MAPPING:
                    p_val = str(prec.get(raw_field) or "").strip().lower()
                    s_val = str(srec.get(raw_field) or "").strip().lower()

                    if p_val != s_val:
                        disputed_in_article.append(raw_field)
                        # Ensure adjudication record exists for this disputed article
                        if aid not in adj_by_id:
                            raise ValueError(
                                f"Unresolved disagreement on article '{aid}' field '{raw_field}': "
                                f"primary '{p_val}' vs secondary '{s_val}'. Adjudication is required "
                                f"and cannot silently adopt the primary annotator's label."
                            )

                        adj = adj_by_id[aid]
                        resolved_val = str(adj.get(adj_field) or "").strip().lower()
                        if not resolved_val:
                            raise ValueError(
                                f"Incomplete adjudication for article '{aid}': field '{raw_field}' "
                                f"was disputed ('{p_val}' vs '{s_val}') but adjudicated column '{adj_field}' is blank."
                            )
                        if resolved_val not in valid_cats:
                            raise ValueError(
                                f"Invalid adjudicated value '{resolved_val}' for field '{adj_field}' on article '{aid}'."
                            )
                        final_values[raw_field] = resolved_val
                    else:
                        # Raters agreed: accept consensus value
                        final_values[raw_field] = p_val

                if disputed_in_article:
                    adj = adj_by_id[aid]
                    adj_id = str(adj.get("adjudicator_id") or "expert").strip()
                    p_id = str(prec.get("annotator_id") or "ann1").strip()
                    s_id = str(srec.get("annotator_id") or "ann2").strip()
                    final_annotator = f"{p_id}+{s_id}[adj:{adj_id}]"
                    adj_notes = str(adj.get("adjudication_notes") or "").strip()
                    final_notes = f"Adjudicated. Notes: {adj_notes}" if adj_notes else "Adjudicated."
                    adjudicated_resolved += 1
                else:
                    # Raters completely agreed across all tasks
                    p_id = str(prec.get("annotator_id") or "ann1").strip()
                    s_id = str(srec.get("annotator_id") or "ann2").strip()
                    final_annotator = f"{p_id}+{s_id}[consensus]"
                    auto_accepted_consensus += 1
            else:
                # Article not part of the overlap sample: accept primary labels
                for raw_field, _, _ in self.TASK_MAPPING:
                    final_values[raw_field] = str(prec.get(raw_field) or "").strip().lower()
                primary_unshared += 1

            # 5. Strict Completeness and Taxonomy Validation of Final Ground Truth
            if strict:
                if not final_annotator:
                    raise ValueError(f"Article '{aid}' has empty annotator_id in final ground truth.")
                for raw_field, _, valid_cats in self.TASK_MAPPING:
                    val = final_values.get(raw_field, "")
                    if not val:
                        raise ValueError(f"Incomplete ground truth for article '{aid}': missing required label '{raw_field}'.")
                    if val not in valid_cats:
                        raise ValueError(f"Taxonomy violation for article '{aid}' field '{raw_field}': '{val}' not in {sorted(list(valid_cats))}.")

            clean_item = {
                "article_id": aid,
                "commodity": comm,
                "article_url": url,
                "article_title": title,
                "article_publication_timestamp": prec.get("article_publication_timestamp"),
                "domain": prec.get("domain", ""),
                "language": prec.get("language", "unknown"),
                "source_country": prec.get("source_country"),
                "true_relevance_class": final_values.get("true_relevance_class", ""),
                "true_sentiment_label": final_values.get("true_sentiment_label", ""),
                "true_event_type": final_values.get("true_event_type", ""),
                "true_event_impact_direction": final_values.get("true_event_impact_direction", ""),
                "annotator_id": final_annotator,
                "annotator_notes": final_notes,
            }
            merged_records.append(clean_item)

        # 6. Verify Total Article Retention
        if len(merged_records) != len(primary_records):
            raise ValueError(
                f"Merged dataset count mismatch: expected {len(primary_records)} records, got {len(merged_records)}."
            )

        # 7. Write Output JSONL
        if output_jsonl_path is not None:
            out_p = Path(output_jsonl_path)
            out_p.parent.mkdir(parents=True, exist_ok=True)
            with open(out_p, "w", encoding="utf-8") as f:
                for r in merged_records:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
            logger.info("Saved %d merged ground truth records to %s", len(merged_records), out_p)

        return {
            "total_merged_records": len(merged_records),
            "primary_unshared_count": primary_unshared,
            "consensus_accepted_count": auto_accepted_consensus,
            "adjudicated_resolved_count": adjudicated_resolved,
            "output_path": str(output_jsonl_path) if output_jsonl_path else None,
        }


def load_annotation_records(path: Path) -> List[Dict[str, Any]]:
    """
    Convenience loader supporting both JSONL and CSV annotated files with
    automatic spreadsheet formula desanitization.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Annotation file not found: {path}")

    if path.suffix.lower() == ".csv":
        with open(path, "r", encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)
            records = []
            for row in reader:
                cleaned = {
                    (k.strip() if k else ""): str(desanitize_from_spreadsheet((v or "").strip()))
                    for k, v in row.items()
                }
                records.append(cleaned)
        return records
    else:
        return load_jsonl_records(path)
