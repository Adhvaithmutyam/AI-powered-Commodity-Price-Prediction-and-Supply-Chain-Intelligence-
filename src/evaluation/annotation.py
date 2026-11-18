"""
Human Annotation Workflow and Evaluation Sample Generation.

Generates reproducible, stratified, unbiased annotation templates from ingested
news or feature datasets. Eliminates model prediction leakage to ensure objective
human ground-truth evaluation.
"""

from collections import Counter
import csv
from datetime import datetime, timezone
import hashlib
import json
import logging
from pathlib import Path
import random
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

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Comprehensive Annotation Guidelines
# ---------------------------------------------------------------------------
ANNOTATION_GUIDELINES: str = """# Commodity News Intelligence Human Annotation Guidelines

## 0. Dataset Scope & Characterization: Real-World Pilot Dataset

> [!IMPORTANT]
> **Real-World Pilot Status**: The active annotation dataset (50 articles: 40 crude oil, 5 copper, 5 gold)
> is formally designated as a **real-world pilot annotation dataset**, NOT a statistically representative
> or balanced benchmark across all three commodity markets.

### Key Characteristics & Known Limitations:
1. **Severe Commodity Imbalance**: Due to repeated HTTP 429 rate limits encountered on GDELT public servers for complex copper and gold queries, genuine articles available are skewed (40 crude oil vs 5 copper and 5 gold). Never fabricate fake articles to artificially balance the dataset.
2. **Short Temporal Window**: Current articles span October 20 to November 12, 2026 (~3 weeks), reflecting immediate market reporting rather than multiple macro cycles, price regimes, or long-term structural supply shocks.
3. **Publisher Concentration**: Primarily composed of financial portals, corporate news aggregators, and energy reporting wires.
4. **Event Type Sparsity**: Rule-based event extraction classified 48 of the 50 sampled articles as `other` (96%), with only 1 `inventory_change` and 1 `geopolitical_risk`. Rare event categories (`supply_disruption`, `demand_change`, `policy_action`, `production_change`) have zero automated detections in this pilot.
5. **Statistical Consequences**:
   - Class-specific metrics for copper ($N=5$) and gold ($N=5$) will have high binomial variance and wide confidence intervals.
   - Macro F1 across commodities will be disproportionately affected by individual errors in small strata.
   - Event detection recall cannot be established without complete human annotation of the large `other` category to uncover missed true events (false negatives).
   - This pilot serves to calibrate the human annotation protocol and verify pipeline integration before large-scale scheduled collection.

---

## Overview and Ground-Truth Integrity Principle

This document establishes the official criteria for human annotators creating unbiased ground-truth
evaluation datasets for:
1. **News Relevance Assessment** (`true_relevance_class`)
2. **Financial Sentiment Analysis** (`true_sentiment_label`)
3. **Physical Supply-Chain Event Extraction** (`true_event_type`)
4. **Expected Commodity Price Impact Direction** (`true_event_impact_direction`)

### Strict Evaluation Safeguards
- **Zero Model Leakage**: Annotators must evaluate articles blind. Never consult model outputs, automated rule tags, or model confidence scores during annotation.
- **Evidence-Based Labeling**: Decisions must be justified strictly by the verifiable evidence available in the article headline and source metadata. Do not speculate beyond what is written.
- **Empirical Ground-Truth Standard**: Synthetic test labels establish test suite correctness, not real-world NLP accuracy. Only verified human labels produce genuine empirical precision, recall, and F1 metrics. Model accuracy must never be claimed without complete human ground truth.

---

## CRITICAL CONCEPTUAL DISTINCTION: Financial Sentiment vs. Commodity Price Impact

A fundamental challenge in commodity NLP is that **financial sentiment (tone/polarity)** is frequently the **inverse of expected physical commodity price impact**.

- **Financial Sentiment** reflects the emotional or business tone of the reported event itself (e.g. disaster, industrial accident, labor dispute, cost inflation, or corporate distress is **negative**; operational success, record extraction, or corporate profit is **positive**).
- **Commodity Price Impact** reflects microeconomic supply and demand balance:
  - $\\text{Supply Disruptions} \\implies \\text{Supply Deficit} \\implies \\text{Price Rallies (Bullish)}$
  - $\\text{Production Booms / Inventory Accumulation} \\implies \\text{Supply Surplus} \\implies \\text{Price Drops (Bearish)}$

### Direct Comparison Matrix:

| Event Headline | Operational / News Sentiment (`true_sentiment_label`) | Supply-Chain Event Type (`true_event_type`) | Expected Commodity Price Impact (`true_event_impact_direction`) | Economic Rationale |
| :--- | :--- | :--- | :--- | :--- |
| **"Escondida copper mine halts operations after fatal rockburst accident"** | `negative` | `supply_disruption` | `potentially_bullish` | Operational tragedy/distress (negative sentiment) curtails global copper supply, driving prices upward (bullish price impact). |
| **"Major new crude oil field discovered in Guyana exceeding 2 billion barrels"** | `positive` | `production_change` | `potentially_bearish` | Commercial success (positive sentiment) expands future supply capacity, exerting downward pressure on crude prices (bearish price impact). |
| **"OPEC+ ministers agree on surprise 1.5 million bpd production cut"** | `neutral` / `negative` | `production_change` | `potentially_bullish` | Cartel intervention restricts supply; oil prices surge (bullish). |
| **"EIA reports massive 8.4 million barrel draw in US commercial crude oil stockpiles"** | `positive` | `inventory_change` | `potentially_bullish` | Large stockpile depletion signals robust demand/tight supply (bullish price impact). |
| **"EIA reports unexpected 5.2 million barrel build in US crude stockpiles as refineries stall"** | `negative` | `inventory_change` | `potentially_bearish` | Accumulating inventory signals excess supply or sluggish refining demand (bearish price impact). |
| **"Gold slides 1.8% as Fed signals aggressive rate hike path and strong dollar"** | `negative` | `other` | `potentially_bearish` | Market reporting of price drop and rising yields; non-yielding gold loses attractiveness (bearish price impact). |
| **"Middle East conflict escalates; oil tankers rerouted away from Red Sea strait"** | `negative` | `geopolitical_risk` | `potentially_bullish` | Geopolitical crisis and transit disruption raises transport costs and supply risk (bullish price impact). |

**Rule for Annotators:** Never automatically copy `true_sentiment_label` into `true_event_impact_direction`. Evaluate them independently using the criteria below.

---

## 1. News Relevance (`true_relevance_class`)

Evaluates whether the article provides actionable intelligence regarding commodity supply, physical logistics, consumption demand, pricing, or relevant macro drivers.

### Allowed Taxonomy:
- `directly_relevant`
- `potentially_relevant`
- `irrelevant`
- `uncertain`

### Taxonomy Definitions & Criteria:

1. **`directly_relevant`**:
   - The article directly names the commodity (copper, gold, crude oil, petroleum, gasoline, diesel, Brent, WTI) or a primary market asset/institution (LME, COMEX, OPEC, EIA, Codelco, Aramco, Newmont).
   - The headline discusses physical mining, drilling, refining, smelting, commercial stockpiles, trade tariffs, physical shipping/transit, pipeline operations, or physical trading prices.
   - *Copper Examples*: "Freeport Indonesia copper concentrate export license extended", "Codelco copper output drops to 25-year low", "LME copper warehouse inventories surge 14%".
   - *Gold Examples*: "Central banks purchase record 290 tonnes of gold in Q3", "Barrick Gold quarterly bullion production tops analyst estimates", "Gold prices touch all-time high of $2,700/oz".
   - *Crude Oil Examples*: "OPEC+ members extend voluntary oil production cuts of 2.2 mb/d", "US crude exports jump to 4.5 million barrels a day", "Pipeline blast disrupts 200,000 bpd of crude delivery".

2. **`potentially_relevant`**:
   - Macroeconomic, monetary, cross-asset, geopolitical, or broad industrial trends that strongly influence commodity prices or industrial consumption, even if the specific commodity is not named in the headline.
   - Includes: Federal Reserve / central bank interest rate decisions, global inflation reports (CPI/PPI), manufacturing PMI surveys (especially China, US, Eurozone), US Dollar index (DXY) moves, and broad environmental/trade policy.
   - *Examples*: "Fed cuts interest rates by 50 basis points amidst softening labor market", "China manufacturing activity shrinks for fifth consecutive month", "US Dollar rallies to two-year peak following robust retail sales", "Global shipping freight rates spike on Panama Canal drought restrictions".

3. **`irrelevant`**:
   - False positives generated by search homonyms, figurative expressions, sports achievements, entertainment, consumer retail products, or unrelated corporate entities.
   - *Copper*: "Copper Mountain ski resort opens new alpine chairlift", "Police arrest suspect in copper pipe burglary from residential construction site", "Chef recommends hammered copper cookware for confectionary".
   - *Gold*: "Sprinter wins Olympic gold in 100-meter dash", "Broadway musical strikes box office gold in opening weekend", "Retiring director receives commemorative gold watch", "Golden Retrievers gather for annual kennel club show".
   - *Crude Oil*: "Comedian's crude monologue draws criticism at awards gala", "Company uses crude estimation technique for employee surveys".

4. **`uncertain`**:
   - Vague, truncated, ambiguous, or boilerplate headlines where subject matter cannot be determined without full body text, or headlines consisting of pure navigation/clickbait artifacts.
   - *Examples*: "Market perspectives: what to expect this week", "Top stories in business and world news today", "Exclusive: Key industry leaders share their outlook", "Editorial: Navigating current global turbulence".

---

## 2. Financial Sentiment Polarity (`true_sentiment_label`)

Measures the financial sentiment/tone of the headline from the standpoint of market participants and financial economics (aligned with FinBERT conventions).

### Allowed Taxonomy:
- `positive`
- `negative`
- `neutral`

### Taxonomy Definitions & Criteria:

1. **`positive`**:
   - Reports financial gains, revenue growth, upward price movement, robust demand, successful exploration, project expansion, or constructive commercial agreements.
   - *Examples*: "Copper prices rally to three-month peak on supply deficit concerns", "ExxonMobil delivers record upstream earnings", "Gold shines as safe-haven inflows surge".

2. **`negative`**:
   - Reports financial losses, downward price movement, production shutdowns, accidents, labor strikes, regulatory sanctions, credit downgrades, or demand contraction.
   - *Examples*: "Gold slides on surging bond yields and hawkish Fed stance", "Crude oil plunges 4% as demand growth outlook dims", "Smelter fire halts operations indefinitely at major copper plant".

3. **`neutral`**:
   - Factual, informational, scheduled, or administrative announcements without distinct directional sentiment or emotion.
   - *Examples*: "EIA scheduled to release weekly petroleum status report at 10:30 AM EST", "LME releases revised warehouse storage rules for base metals", "Mining industry delegates gather for annual Latin American summit in Lima".

---

## 3. Physical Supply-Chain Event Types (`true_event_type`)

Classifies physical, operational, and macroeconomic supply-chain events impacting physical commodity availability and logistics.

### Allowed Taxonomy:
- `supply_disruption`
- `demand_change`
- `geopolitical_risk`
- `inventory_change`
- `policy_action`
- `production_change`
- `other`

### Taxonomy Definitions & Criteria:

1. **`supply_disruption`**:
   - Physical impediments directly halting or curtailing extraction, refining, or transport: unplanned mine closures, cave-ins, labor strikes, equipment failures, pipeline leaks/explosions, railway derailments, or maritime shipping bottlenecks.
   - *Examples*: "Escondida union workers vote to strike after wage negotiations collapse", "Colonial Pipeline shut down following diesel leak in Virginia", "Flooding in Queensland halts coking coal and copper rail shipments".

2. **`demand_change`**:
   - Explicit shifts in physical consumption channels, commercial purchasing patterns, strategic reserve purchases, or macroeconomic end-use demand (e.g. EV battery production, construction, grid infrastructure).
   - *Examples*: "China state grid copper procurement rises 18% year-over-year", "Global electric vehicle sales reach record quarterly volume, boosting copper and battery metals", "European industrial gas and diesel demand declines 6% amidst factory closures".

3. **`geopolitical_risk`**:
   - Armed conflict, military actions, terrorism, rebel attacks, maritime piracy, direct economic sanctions, embargoes, or heightened national security tensions affecting commodity producing or transit regions.
   - *Examples*: "Drone strike damages key Russian oil export terminal on the Black Sea", "Houthi missile attacks force container ships and oil tankers to bypass Bab el-Mandeb strait", "US imposes new sanctions targeting Iranian petroleum fleet and illicit bunkering network".

4. **`inventory_change`**:
   - Explicit announcements, official releases, or exchange data regarding physical stockpiles in warehouses, storage hubs, or reserves (LME, COMEX, SHFE, EIA, API).
   - *Examples*: "LME copper inventories drop to lowest level since August", "API reports 4.1 million barrel draw in US crude stocks", "Cushing crude oil hub storage levels fall to near operational minimums".

5. **`policy_action`**:
   - Government, legislative, or regulatory actions: export bans, import tariffs, mining concessions, environmental restrictions, carbon taxes, or strategic reserve releases.
   - *Examples*: "Chile passes landmark mining royalty tax increase on large copper producers", "US government announces release of 15 million barrels from Strategic Petroleum Reserve", "Indonesia introduces new export ban on unrefined mineral ores".

6. **`production_change`**:
   - Deliberate corporate or cartel decisions adjusting output capacity, quotas, exploration development, or smelter utilization.
   - *Examples*: "OPEC+ delegates agree to prolong voluntary output cut of 2.2 million bpd through Q1", "Anglo American curtails production targets at Los Bronces copper mine to manage water constraints", "Newmont achieves commercial production at Ahafo North gold project ahead of schedule".

7. **`other`**:
   - Any article that does not describe a concrete physical supply-chain event (e.g. pure financial price commentary, macroeconomic interest rate updates, general opinion pieces, or irrelevant articles).
   - *Examples*: "Why copper is considered the world's economic bellwether", "Gold price analysis: key support and resistance levels for November", "Oil market perspectives following US election results".

---

## 4. Expected Impact Direction (`true_event_impact_direction`)

Evaluates the theoretical directional pressure exerted by the event on physical commodity prices based on supply and demand microeconomics.

### Allowed Taxonomy:
- `potentially_bullish`
- `potentially_bearish`
- `mixed`
- `unknown`

### Decision Rules:
- **`potentially_bullish`**: Event reduces physical supply (mine strike, export ban, pipeline outage, production cut) or increases physical demand (grid expansion, reserve purchases, stock draws).
- **`potentially_bearish`**: Event increases physical supply (production quotas raised, new mine opens, inventory build) or reduces physical demand (industrial shutdowns, recessionary contraction).
- **`mixed`**: Conflicting economic forces present in the headline (e.g. "OPEC raises quotas but Russian pipeline offline; markets weigh supply balance").
- **`unknown`**: Used when event type is `other`, article is `irrelevant` or `uncertain`, or no directional supply/demand mechanism exists.

---

## 5. Handling Uncertainty, Missing Evidence, and Ambiguity

- **Missing or Truncated Text**: If a headline is incomplete (e.g. ends with "..." or "Read more") and its meaning cannot be deduced with high confidence, set `true_relevance_class = "uncertain"`, `true_sentiment_label = "neutral"`, `true_event_type = "other"`, and `true_event_impact_direction = "unknown"`. Provide a note in `annotator_notes`.
- **Dual Commodity Mentions**: If an article discusses multiple commodities (e.g. "Copper and gold rally as dollar slides"), annotate the record specifically with respect to the designated `commodity` field for that row.
- **Corporate Earnings / Financial Results**: If an oil company reports high earnings, this is `positive` financial sentiment. However, unless specific production or reserve changes are cited, event type should be `other` and impact direction `unknown`.

---

## 6. Quality Control and Disagreement Protocol

To guarantee reliable ground-truth datasets for scientific and operational evaluation:
1. **Dual Independent Annotation**: A random 20% validation split of all records must be annotated independently by two annotators without consultation.
2. **Inter-Annotator Agreement (IAA)**:
   - Compute Cohen's Kappa ($\\kappa$) for categorical fields (`relevance`, `sentiment`, `event_type`).
   - Acceptable benchmark: $\\kappa \\ge 0.75$ (substantial agreement). If $\\kappa < 0.70$, hold a calibration session to clarify guideline ambiguities.
3. **Adjudication Procedure**:
   - Records with conflicting annotations are flagged for adjudication.
   - A designated domain expert (e.g. senior commodity quantitative analyst) makes the final binding determination.
   - The original individual labels and resolution notes must be recorded in audit logs.

---

## 7. Human Annotation & Spreadsheet Workflow

### Step-by-Step Procedure:
1. **Open the CSV**: Open `data/evaluation/annotation_sample.csv` in Microsoft Excel, LibreOffice Calc, or Google Sheets.
2. **Inspect Context Columns**: Read `commodity`, `article_title`, `article_publication_timestamp`, `article_url`, and `domain`.
3. **Populate Blank Label Columns**: Enter authorized values (lowercase) into:
   - `true_relevance_class`: `directly_relevant`, `potentially_relevant`, `irrelevant`, or `uncertain`
   - `true_sentiment_label`: `positive`, `negative`, or `neutral`
   - `true_event_type`: `supply_disruption`, `demand_change`, `geopolitical_risk`, `inventory_change`, `policy_action`, `production_change`, or `other`
   - `true_event_impact_direction`: `potentially_bullish`, `potentially_bearish`, `mixed`, or `unknown`
   - `annotator_id`: Your initials or assigned ID (e.g. `annotator_01`)
   - `annotator_notes`: Optional comments regarding ambiguity, dual commodity mentions, or reasoning.
4. **Save File**: Save the file as CSV (UTF-8 format).
5. **Convert & Validate to JSONL**:
   ```bash
   python run_evaluation.py --mode import-csv --input data/evaluation/annotation_sample.csv --output data/evaluation/human_ground_truth.jsonl
   ```
6. **Run Supervised Evaluation**:
   ```bash
   python run_evaluation.py --mode evaluate --ground-truth data/evaluation/human_ground_truth.jsonl
   ```

### Formula Injection Safety Notice:
When CSV files are exported, any text starting with formula execution characters (`=`, `+`, `-`, `@`, `\t`, `\r`) is automatically escaped with a leading single quote (`'`) to protect spreadsheet software against DDE/CSV formula injection attacks. The import pipeline automatically strips this escaping prefix when converting back to JSONL, ensuring the underlying article title, URL, and deterministic ID match 100% without data corruption.
"""


FORMULA_PREFIXES: Tuple[str, ...] = ("=", "+", "-", "@", "\t", "\r")


def sanitize_for_spreadsheet(value: Any) -> Any:
    """
    Protect string values against formula injection (CSV/DDE injection) in spreadsheets.
    If a string value begins with formula characters (=, +, -, @, \\t, \\r), prefix with a single quote.
    """
    if isinstance(value, str) and value.startswith(FORMULA_PREFIXES):
        return "'" + value
    return value


def desanitize_from_spreadsheet(value: Any) -> Any:
    """
    Restore string values escaped with leading single-quote for spreadsheet formula protection.
    Leaves non-escaped strings untouched.
    """
    if isinstance(value, str) and value.startswith("'") and len(value) > 1 and value[1] in FORMULA_PREFIXES:
        return value[1:]
    return value


def _generate_deterministic_id(commodity: str, url: str) -> str:
    """Generate a stable, unique 12-character article identifier using canonicalized URL."""
    c_url = canonicalize_url(url) or str(url or "").strip()
    raw = f"{commodity.strip().lower()}::{c_url}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:12]


class AnnotationSampler:
    """
    Constructs deterministic, stratified evaluation sample datasets for human annotation.
    Guarantees no leakage of model predictions into labeling templates.
    """

    def __init__(self, config: Optional[EvaluationConfig] = None) -> None:
        self.config = config or EvaluationConfig()

    def sample(
        self,
        records: List[Dict[str, Any]],
        sample_size: int = 50,
        seed: int = 42,
        stratify_by_commodity: bool = True,
    ) -> List[Dict[str, Any]]:
        """
        Extract a deduplicated, stratified, bias-free sample for human labeling.

        Args:
            records: List of article or feature records.
            sample_size: Total number of records to sample.
            seed: Fixed random seed for complete reproducibility.
            stratify_by_commodity: Whether to balance sample across target commodities.

        Returns:
            List of blank annotation templates containing only article context.
        """
        if not records or sample_size <= 0:
            return []

        # 1. Deduplicate by (commodity, canonical_url) and filter to supported commodities
        seen: Set[Tuple[str, str]] = set()
        by_commodity: Dict[str, List[Dict[str, Any]]] = {comm: [] for comm in self.config.commodities}

        for rec in records:
            if not isinstance(rec, dict):
                continue
            comm = str(rec.get("commodity") or "").strip().lower()
            if comm not in self.config.commodities:
                continue

            url_raw = rec.get("article_url") or rec.get("url") or rec.get("evidence_url")
            url = canonicalize_url(url_raw)
            if not url:
                continue

            title = str(rec.get("article_title") or rec.get("title") or rec.get("event_summary") or "").strip()
            if not title:
                continue

            key = (comm, url)
            if key in seen:
                continue
            seen.add(key)

            pub_ts = rec.get("article_publication_timestamp") or rec.get("publication_timestamp")

            clean_item = {
                "article_id": _generate_deterministic_id(comm, url),
                "commodity": comm,
                "article_url": url,
                "article_title": title,
                "article_publication_timestamp": pub_ts,
                "domain": rec.get("domain") or "",
                "language": rec.get("language") or "unknown",
                "source_country": rec.get("source_country"),
                # Blank human ground-truth fields to be populated by annotators
                "true_relevance_class": "",
                "true_sentiment_label": "",
                "true_event_type": "",
                "true_event_impact_direction": "",
                "annotator_id": "",
                "annotator_notes": "",
            }
            by_commodity[comm].append(clean_item)

        rng = random.Random(seed)

        # 2. Perform stratified or uniform deterministic sampling
        sampled_records: List[Dict[str, Any]] = []

        if stratify_by_commodity:
            active_commodities = [c for c in self.config.commodities if by_commodity[c]]
            if not active_commodities:
                return []

            total_available = sum(len(by_commodity[c]) for c in active_commodities)
            target_total = min(sample_size, total_available)

            # Water-filling quota allocation to balance commodities fairly while fulfilling target_total
            allocated_quotas: Dict[str, int] = {c: 0 for c in active_commodities}
            remaining_commodities = list(active_commodities)
            needed = target_total

            while remaining_commodities and needed > 0:
                fair_share = needed // len(remaining_commodities)
                remainder = needed % len(remaining_commodities)
                progress = False

                for idx, comm in enumerate(list(remaining_commodities)):
                    tentative_quota = fair_share + (1 if idx < remainder else 0)
                    avail = len(by_commodity[comm]) - allocated_quotas[comm]
                    if avail <= tentative_quota:
                        allocated_quotas[comm] += avail
                        needed -= avail
                        remaining_commodities.remove(comm)
                        progress = True
                    elif len(remaining_commodities) == 1:
                        allocated_quotas[comm] += needed
                        needed = 0
                        remaining_commodities.remove(comm)
                        progress = True

                if not progress:
                    # All remaining commodities have plenty of items
                    for idx, comm in enumerate(remaining_commodities):
                        tentative_quota = fair_share + (1 if idx < remainder else 0)
                        allocated_quotas[comm] += tentative_quota
                        needed -= tentative_quota
                    break

            # Deterministically shuffle each commodity pool and extract quota
            for comm in active_commodities:
                pool = by_commodity[comm]
                rng.shuffle(pool)
                quota = allocated_quotas[comm]
                sampled_records.extend(pool[:quota])

            # Re-shuffle combined sample deterministically
            rng.shuffle(sampled_records)
        else:
            flat_pool = [item for pool in by_commodity.values() for item in pool]
            rng.shuffle(flat_pool)
            sampled_records = flat_pool[:sample_size]

        return sampled_records


def generate_annotation_sample(
    input_path: Path,
    output_path: Path,
    sample_size: int = 50,
    seed: int = 42,
    stratify: bool = True,
    include_guidelines: bool = True,
) -> List[Dict[str, Any]]:
    """
    Convenience function to load records, sample, and save annotation templates.
    Generates both JSONL and Excel-friendly CSV formats, and optionally writes
    an accompanying ANNOTATION_GUIDELINES.md file.
    """
    records = load_jsonl_records(input_path)
    sampler = AnnotationSampler()
    sample = sampler.sample(records, sample_size=sample_size, seed=seed, stratify_by_commodity=stratify)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_path.suffix.lower() == ".csv":
        csv_path = output_path
        jsonl_path = output_path.with_suffix(".jsonl")
    else:
        jsonl_path = output_path
        csv_path = output_path.with_suffix(".csv")

    # 1. Write JSONL template
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for rec in sample:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    logger.info("Saved %d annotation templates to JSONL: %s", len(sample), jsonl_path)

    # 2. Write Excel-friendly CSV template (UTF-8 with BOM)
    if sample:
        fieldnames = list(sample[0].keys())
    else:
        fieldnames = [
            "article_id",
            "commodity",
            "article_url",
            "article_title",
            "article_publication_timestamp",
            "domain",
            "language",
            "source_country",
            "true_relevance_class",
            "true_sentiment_label",
            "true_event_type",
            "true_event_impact_direction",
            "annotator_id",
            "annotator_notes",
        ]

    with open(csv_path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for rec in sample:
            sanitized_rec = {k: sanitize_for_spreadsheet(v) for k, v in rec.items()}
            writer.writerow(sanitized_rec)
    logger.info("Saved %d annotation templates to CSV: %s", len(sample), csv_path)

    # 3. Write accompanying guidelines
    if include_guidelines:
        guide_path = output_path.parent / "ANNOTATION_GUIDELINES.md"
        with open(guide_path, "w", encoding="utf-8") as f:
            f.write(ANNOTATION_GUIDELINES)
        logger.info("Saved annotation guidelines to %s", guide_path)

    return sample


def validate_annotation_dataset(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Audit an annotated dataset to verify label formatting, completion rate,
    duplicate annotations, and taxonomy compliance prior to downstream evaluation.
    """
    total = len(records)
    labeled_relevance = 0
    labeled_sentiment = 0
    labeled_events = 0
    fully_labeled = 0

    seen_annotations: Set[Tuple[str, str]] = set()
    duplicate_annotations: List[Dict[str, Any]] = []

    relevance_counts: Counter = Counter()
    sentiment_counts: Counter = Counter()
    event_counts: Counter = Counter()

    invalid_relevance: List[Dict[str, Any]] = []
    invalid_sentiment: List[Dict[str, Any]] = []
    invalid_events: List[Dict[str, Any]] = []

    for idx, rec in enumerate(records, start=1):
        if not isinstance(rec, dict):
            continue

        comm = str(rec.get("commodity") or "").strip().lower()
        url = canonicalize_url(rec.get("article_url") or rec.get("url"))
        art_id = rec.get("article_id")

        if comm and url:
            dedup_key = (comm, url)
            if dedup_key in seen_annotations:
                duplicate_annotations.append({
                    "record_index": idx,
                    "commodity": comm,
                    "article_url": url,
                    "article_id": art_id,
                })
            else:
                seen_annotations.add(dedup_key)

        rel = str(rec.get("true_relevance_class") or "").strip().lower()
        sent = str(rec.get("true_sentiment_label") or "").strip().lower()
        ev = str(rec.get("true_event_type") or "").strip().lower()

        rel_valid = False
        sent_valid = False
        ev_valid = False

        if rel:
            labeled_relevance += 1
            relevance_counts[rel] += 1
            if rel not in VALID_RELEVANCE_CLASSES:
                invalid_relevance.append({"record_index": idx, "value": rel})
            else:
                rel_valid = True

        if sent:
            labeled_sentiment += 1
            sentiment_counts[sent] += 1
            if sent not in VALID_SENTIMENT_LABELS:
                invalid_sentiment.append({"record_index": idx, "value": sent})
            else:
                sent_valid = True

        if ev:
            labeled_events += 1
            event_counts[ev] += 1
            if ev not in VALID_EVENT_TYPES:
                invalid_events.append({"record_index": idx, "value": ev})
            else:
                ev_valid = True

        if rel_valid and sent_valid and ev_valid:
            fully_labeled += 1

    return {
        "total_records": total,
        "completion_status": {
            "fully_labeled_records": fully_labeled,
            "fully_labeled_rate": round(fully_labeled / total, 4) if total > 0 else 0.0,
            "labeled_relevance_count": labeled_relevance,
            "labeled_relevance_rate": round(labeled_relevance / total, 4) if total > 0 else 0.0,
            "labeled_sentiment_count": labeled_sentiment,
            "labeled_sentiment_rate": round(labeled_sentiment / total, 4) if total > 0 else 0.0,
            "labeled_events_count": labeled_events,
            "labeled_events_rate": round(labeled_events / total, 4) if total > 0 else 0.0,
        },
        "duplicate_detection": {
            "unique_annotations_count": len(seen_annotations),
            "duplicate_annotations_count": len(duplicate_annotations),
            "duplicate_annotations_samples": duplicate_annotations[:10],
        },
        "label_distributions": {
            "relevance": dict(relevance_counts),
            "sentiment": dict(sentiment_counts),
            "events": dict(event_counts),
        },
        "taxonomy_violations": {
            "invalid_relevance_count": len(invalid_relevance),
            "invalid_relevance_samples": invalid_relevance[:10],
            "invalid_sentiment_count": len(invalid_sentiment),
            "invalid_sentiment_samples": invalid_sentiment[:10],
            "invalid_events_count": len(invalid_events),
            "invalid_events_samples": invalid_events[:10],
        },
    }


class AnnotationCSVImporter:
    """
    Imports human-annotated CSV files (saved from Excel or spreadsheet software),
    validates taxonomies, detects incomplete annotations, catches duplicate identities,
    sanitizes spreadsheet formula injections, and exports validated JSONL ground truth.
    """

    REQUIRED_COLUMNS: Set[str] = {"commodity", "article_url", "article_title"}

    def __init__(self, config: Optional[EvaluationConfig] = None) -> None:
        self.config = config or EvaluationConfig()

    def import_csv(
        self,
        csv_path: Path,
        output_jsonl_path: Optional[Path] = None,
        overwrite: bool = False,
        strict: bool = False,
    ) -> Dict[str, Any]:
        """
        Import and validate an annotated CSV file.

        Args:
            csv_path: Path to input CSV.
            output_jsonl_path: Optional path to write converted JSONL.
            overwrite: Whether to overwrite existing output JSONL file.
            strict: If True, raise ValueError on taxonomy violations, duplicates, or malformed rows.

        Returns:
            Structured audit dictionary of conversion and validation results.
        """
        csv_path = Path(csv_path)
        if not csv_path.exists():
            raise FileNotFoundError(f"Input CSV file not found: '{csv_path}'")

        if output_jsonl_path is not None:
            out_p = Path(output_jsonl_path)
            if out_p.exists() and not overwrite:
                raise FileExistsError(
                    f"Target ground-truth file '{out_p}' already exists. "
                    f"Set overwrite=True (or pass --overwrite) to replace."
                )

        # 1. Read CSV using utf-8-sig to handle Excel UTF-8 with BOM and plain UTF-8
        try:
            with open(csv_path, "r", encoding="utf-8-sig", newline="") as f:
                reader = csv.DictReader(f)
                if not reader.fieldnames:
                    raise ValueError(f"Malformed CSV: '{csv_path}' has no header row.")

                fieldnames_set = {fn.strip() for fn in reader.fieldnames if fn}
                missing_req = self.REQUIRED_COLUMNS - fieldnames_set
                if missing_req:
                    raise ValueError(
                        f"Malformed annotation CSV: missing required column(s): {sorted(list(missing_req))}. "
                        f"Found columns: {reader.fieldnames}"
                    )

                raw_rows = list(reader)
        except UnicodeDecodeError as exc:
            raise ValueError(f"Failed to decode CSV '{csv_path}': file is not valid UTF-8/UTF-8-BOM.") from exc

        total_rows = len(raw_rows)
        if total_rows == 0:
            raise ValueError(f"CSV file '{csv_path}' contains a header but no data rows.")

        converted_records: List[Dict[str, Any]] = []
        seen_keys: Set[Tuple[str, str]] = set()
        seen_ids: Set[str] = set()

        duplicate_annotations: List[Dict[str, Any]] = []
        taxonomy_violations: List[Dict[str, Any]] = []
        malformed_rows: List[Dict[str, Any]] = []

        labeled_relevance = 0
        labeled_sentiment = 0
        labeled_events = 0
        labeled_impact = 0
        fully_labeled = 0
        partially_labeled = 0
        unlabeled = 0

        for row_idx, row in enumerate(raw_rows, start=2):
            # Clean keys/values and desanitize spreadsheet formula escaping
            cleaned_row: Dict[str, str] = {
                (k.strip() if k else ""): str(desanitize_from_spreadsheet((v or "").strip()))
                for k, v in row.items()
            }

            comm = cleaned_row.get("commodity", "").lower()
            raw_url = cleaned_row.get("article_url") or cleaned_row.get("url") or ""
            c_url = canonicalize_url(raw_url)
            title = cleaned_row.get("article_title") or cleaned_row.get("title") or ""
            art_id = cleaned_row.get("article_id") or ""

            # Check essential fields
            if not comm or not c_url or not title:
                malformed_rows.append({
                    "row_number": row_idx,
                    "commodity": comm,
                    "url": raw_url,
                    "title": title,
                    "error": "Missing commodity, URL, or title.",
                })
                continue

            if comm not in self.config.commodities:
                taxonomy_violations.append({
                    "row_number": row_idx,
                    "field": "commodity",
                    "value": comm,
                    "allowed": list(self.config.commodities),
                })

            # Derive/preserve deterministic article ID
            expected_id = _generate_deterministic_id(comm, c_url)
            if not art_id:
                art_id = expected_id

            # Duplicate identity detection
            dedup_key = (comm, c_url)
            if dedup_key in seen_keys or art_id in seen_ids:
                duplicate_annotations.append({
                    "row_number": row_idx,
                    "commodity": comm,
                    "article_url": c_url,
                    "article_id": art_id,
                })
                # In non-strict mode, skip duplicate row to ensure clean output
                continue

            seen_keys.add(dedup_key)
            seen_ids.add(art_id)

            # Extract human labels (CRITICAL: NEVER read model predictions!)
            true_rel = cleaned_row.get("true_relevance_class", "").lower()
            true_sent = cleaned_row.get("true_sentiment_label", "").lower()
            true_ev = cleaned_row.get("true_event_type", "").lower()
            true_dir = cleaned_row.get("true_event_impact_direction", "").lower()
            annotator_id = cleaned_row.get("annotator_id", "")
            annotator_notes = cleaned_row.get("annotator_notes", "")

            # Taxonomy validation
            rel_valid = False
            sent_valid = False
            ev_valid = False
            dir_valid = False

            if true_rel:
                labeled_relevance += 1
                if true_rel not in VALID_RELEVANCE_CLASSES:
                    taxonomy_violations.append({
                        "row_number": row_idx,
                        "field": "true_relevance_class",
                        "value": true_rel,
                        "allowed": list(VALID_RELEVANCE_CLASSES),
                    })
                else:
                    rel_valid = True

            if true_sent:
                labeled_sentiment += 1
                if true_sent not in VALID_SENTIMENT_LABELS:
                    taxonomy_violations.append({
                        "row_number": row_idx,
                        "field": "true_sentiment_label",
                        "value": true_sent,
                        "allowed": list(VALID_SENTIMENT_LABELS),
                    })
                else:
                    sent_valid = True

            if true_ev:
                labeled_events += 1
                if true_ev not in VALID_EVENT_TYPES:
                    taxonomy_violations.append({
                        "row_number": row_idx,
                        "field": "true_event_type",
                        "value": true_ev,
                        "allowed": list(VALID_EVENT_TYPES),
                    })
                else:
                    ev_valid = True

            if true_dir:
                labeled_impact += 1
                if true_dir not in VALID_IMPACT_DIRECTIONS:
                    taxonomy_violations.append({
                        "row_number": row_idx,
                        "field": "true_event_impact_direction",
                        "value": true_dir,
                        "allowed": list(VALID_IMPACT_DIRECTIONS),
                    })
                else:
                    dir_valid = True

            # Completion analysis
            any_labeled = bool(true_rel or true_sent or true_ev or true_dir)
            all_core_labeled = bool(rel_valid and sent_valid and ev_valid)

            if all_core_labeled:
                fully_labeled += 1
            elif any_labeled:
                partially_labeled += 1
            else:
                unlabeled += 1

            # Build safe standardized ground-truth record
            record = {
                "article_id": art_id,
                "commodity": comm,
                "article_url": c_url,
                "article_title": title,
                "article_publication_timestamp": cleaned_row.get("article_publication_timestamp") or None,
                "domain": cleaned_row.get("domain", ""),
                "language": cleaned_row.get("language", "unknown"),
                "source_country": cleaned_row.get("source_country") or None,
                "true_relevance_class": true_rel,
                "true_sentiment_label": true_sent,
                "true_event_type": true_ev,
                "true_event_impact_direction": true_dir,
                "annotator_id": annotator_id,
                "annotator_notes": annotator_notes,
            }
            converted_records.append(record)

        # Strict validation enforcement
        if strict:
            err_msgs = []
            if taxonomy_violations:
                err_msgs.append(f"{len(taxonomy_violations)} taxonomy violation(s)")
            if duplicate_annotations:
                err_msgs.append(f"{len(duplicate_annotations)} duplicate article identity(ies)")
            if malformed_rows:
                err_msgs.append(f"{len(malformed_rows)} malformed row(s)")
            if err_msgs:
                raise ValueError(f"Strict validation failed: {', '.join(err_msgs)}.")

        # Write output JSONL
        if output_jsonl_path is not None:
            out_p = Path(output_jsonl_path)
            out_p.parent.mkdir(parents=True, exist_ok=True)
            with open(out_p, "w", encoding="utf-8") as f:
                for rec in converted_records:
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            logger.info("Exported %d ground-truth records to %s", len(converted_records), out_p)

        return {
            "status": "success" if not (taxonomy_violations or duplicate_annotations or malformed_rows) else "completed_with_warnings",
            "csv_path": str(csv_path),
            "output_jsonl_path": str(output_jsonl_path) if output_jsonl_path else None,
            "total_rows_read": total_rows,
            "valid_records_count": len(converted_records),
            "completion_summary": {
                "fully_labeled_count": fully_labeled,
                "fully_labeled_rate": round(fully_labeled / total_rows, 4) if total_rows > 0 else 0.0,
                "partially_labeled_count": partially_labeled,
                "partially_labeled_rate": round(partially_labeled / total_rows, 4) if total_rows > 0 else 0.0,
                "unlabeled_count": unlabeled,
                "unlabeled_rate": round(unlabeled / total_rows, 4) if total_rows > 0 else 0.0,
                "relevance_labeled_count": labeled_relevance,
                "sentiment_labeled_count": labeled_sentiment,
                "events_labeled_count": labeled_events,
                "impact_direction_labeled_count": labeled_impact,
            },
            "duplicate_identities": {
                "duplicate_count": len(duplicate_annotations),
                "duplicate_samples": duplicate_annotations[:10],
            },
            "taxonomy_violations": {
                "violation_count": len(taxonomy_violations),
                "violation_samples": taxonomy_violations[:10],
            },
            "malformed_rows": {
                "malformed_count": len(malformed_rows),
                "malformed_samples": malformed_rows[:10],
            },
        }


def convert_annotated_csv_to_jsonl(
    csv_path: Path,
    output_jsonl_path: Optional[Path] = None,
    overwrite: bool = False,
    strict: bool = False,
) -> Dict[str, Any]:
    """
    Convenience function to import, validate, and convert annotated CSV to JSONL ground truth.
    """
    importer = AnnotationCSVImporter()
    return importer.import_csv(
        csv_path=csv_path,
        output_jsonl_path=output_jsonl_path,
        overwrite=overwrite,
        strict=strict,
    )
