# AI-powered-Commodity-Price-Prediction-and-Supply-Chain-Intelligence-

An end-to-end AI platform for forecasting commodity price fluctuations and analyzing supply-chain risks.

---

## Architecture & Modules (MLOps 3 — NLP & News Intelligence)

This repository contains three decoupled intelligence components for commodity market analysis:
1. **News Ingestion Pipeline**: Automated discovery, ingestion, normalization, and deduplication of global commodity news (Copper, Gold, Crude Oil) from the GDELT DOC 2.0 API.
2. **FinBERT Sentiment Analysis Pipeline**: Pretrained financial domain sentiment inference using `ProsusAI/finbert` via Hugging Face Transformers.
3. **Supply-Chain Event Extraction Pipeline**: Domain-specific extraction of operational disruptions, demand shocks, geopolitical risks, and inventory movements with explainable market implications.

---

### Project Structure

```text
├── .gitignore                      # Ignore venv, caches, raw/processed datasets, and secrets
├── requirements.txt                # Python dependencies (requests, pytest, torch, transformers)
├── run_pipeline.py                 # News ingestion CLI runner
├── run_sentiment.py                # Sentiment analysis CLI runner
├── run_event_extraction.py         # Event extraction CLI runner
├── data/
│   ├── raw/
│   │   ├── .gitkeep                # Tracks directory in git
│   │   └── commodity_news.jsonl    # Raw ingested news records (JSON Lines, git-ignored)
│   └── processed/
│       ├── .gitkeep                # Tracks directory in git
│       ├── commodity_news_sentiment.jsonl # Enriched sentiment records (JSON Lines, git-ignored)
│       └── commodity_events.jsonl         # Extracted supply-chain events (JSON Lines, git-ignored)
├── src/
│   ├── __init__.py
│   ├── news_intelligence/          # News ingestion module
│   │   ├── __init__.py             # Public module interface
│   │   ├── config.py               # Ingestion configuration and commodity queries
│   │   ├── gdelt_client.py         # GDELT DOC 2.0 client (rate-limiting & retries)
│   │   ├── normalizer.py           # Field normalization & timestamp parsing
│   │   ├── storage.py              # JSONL persistence & (commodity, url) deduplication
│   │   └── pipeline.py             # Pipeline orchestrator & CLI
│   ├── sentiment/                  # Sentiment analysis module
│   │   ├── __init__.py             # Public module interface
│   │   ├── config.py               # Sentiment configuration & compute device settings
│   │   ├── analyzer.py             # FinBERT analyzer (lazy loading, batching, safe fallbacks)
│   │   └── pipeline.py             # Sentiment pipeline orchestrator & CLI
│   └── event_extraction/           # Supply-chain event extraction module
│       ├── __init__.py             # Public module interface
│       ├── base.py                 # BaseEventExtractor interface & EventRecord schema
│       ├── config.py               # Pipeline file paths and configuration
│       ├── patterns.py             # Domain pattern taxonomy, negations & economic rules
│       ├── rule_based.py           # RuleBasedEventExtractor implementation
│       └── pipeline.py             # Event extraction orchestrator & CLI
└── tests/
    ├── __init__.py
    ├── test_client.py              # Unit tests for GDELTClient (mocked HTTP/network)
    ├── test_normalizer.py          # Unit tests for schema normalization
    ├── test_storage.py             # Unit tests for deduplication & JSONL storage
    ├── test_pipeline.py            # End-to-end news pipeline tests (mocked offline)
    ├── test_sentiment.py           # Unit tests for FinBERT sentiment (mocked offline)
    └── test_event_extraction.py    # Unit tests for supply-chain event extraction (mocked offline)
```

---

### Installation & Environment Setup

1. **Prerequisites**: Python 3.10+ (tested on Python 3.14).
2. **Clone the repository and checkout the feature branch**:
   ```bash
   git checkout feature/news-intelligence
   ```
3. **Create and activate a virtual environment** (recommended):
   - On Windows (PowerShell):
     ```powershell
     python -m venv .venv
     .\.venv\Scripts\Activate.ps1
     ```
   - On Linux / macOS:
     ```bash
     python -m venv .venv
     source .venv/bin/activate
     ```
4. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

---

### Execution Commands

#### 1. Ingest Raw Commodity News
Queries GDELT DOC 2.0 for copper, gold, and crude oil:
```bash
python run_pipeline.py
```
*Output: `data/raw/commodity_news.jsonl`*

#### 2. Run FinBERT Sentiment Analysis
Enriches raw news records with financial sentiment (`positive`, `negative`, `neutral`):
```bash
python run_sentiment.py
```
*Output: `data/processed/commodity_news_sentiment.jsonl`*

#### 3. Run Supply-Chain Event Extraction
Extracts concrete supply-chain events and evaluates potential market impacts:
```bash
python run_event_extraction.py
```
*Output: `data/processed/commodity_events.jsonl`*

#### Event Extraction CLI Arguments:
| Flag | Default | Description |
|---|---|---|
| `--input` | `data/processed/commodity_news_sentiment.jsonl` | Source file path (falls back to `data/raw/commodity_news.jsonl` if sentiment has not run) |
| `--output` | `data/processed/commodity_events.jsonl` | Output file path for extracted events |
| `--verbose` | `False` | Enable detailed debug logs |

---

### Data Schemas

#### A. Raw Ingestion Schema (`data/raw/commodity_news.jsonl`)
```json
{
  "commodity": "copper",
  "title": "Bimetals : Copper Next Frontier",
  "url": "http://www.prnewswire.com/news-releases/bimetals-coppers-next-frontier-302910564.html",
  "publication_timestamp": "2026-10-15T20:15:00Z",
  "domain": "prnewswire.com",
  "language": "English",
  "source_country": "United States",
  "fetch_timestamp": "2026-10-20T13:11:20Z",
  "provider": "gdelt"
}
```

#### B. Processed Sentiment Schema (`data/processed/commodity_news_sentiment.jsonl`)
```json
{
  "commodity": "copper",
  "title": "Bimetals : Copper Next Frontier",
  "url": "http://www.prnewswire.com/news-releases/bimetals-coppers-next-frontier-302910564.html",
  "publication_timestamp": "2026-10-15T20:15:00Z",
  "domain": "prnewswire.com",
  "language": "English",
  "source_country": "United States",
  "fetch_timestamp": "2026-10-20T13:11:20Z",
  "provider": "gdelt",
  "sentiment_label": "neutral",
  "sentiment_score": 0.8914,
  "sentiment_model": "ProsusAI/finbert",
  "sentiment_status": "success"
}
```

#### C. Extracted Supply-Chain Event Schema (`data/processed/commodity_events.jsonl`)
```json
{
  "commodity": "copper",
  "event_type": "supply_disruption",
  "event_summary": "Grasberg copper mine shutdown announced following workers strike",
  "impact_direction": "potentially_bullish",
  "confidence": 0.90,
  "evidence_url": "https://reuters.com/markets/grasberg-strike",
  "article_publication_timestamp": "2026-10-20T10:00:00Z",
  "extraction_status": "success",
  "evidence_fact": "mine shutdown; workers strike",
  "implication_rationale": "Reported infrastructure stoppage, strike, or export halt may constrain physical market availability."
}
```

#### Event Fields Description:
| Field Name | Type | Description |
|---|---|---|
| `commodity` | string | Target commodity (`copper`, `gold`, `crude oil`, or `unknown`) |
| `event_type` | string | One of the 7 supported event classifications |
| `event_summary` | string | Factual summary derived directly from the reported article headline |
| `impact_direction` | string | Hypothetical market direction: `'potentially_bullish'`, `'potentially_bearish'`, `'mixed'`, or `'unknown'` |
| `confidence` | float | Classification confidence score between `0.0` and `1.0` |
| `evidence_url` | string | Verifiable URL link to the original article source (never fabricated) |
| `article_publication_timestamp` | string \| null | Publication timestamp preserved from source article |
| `extraction_status` | string | Outcome status: `'success'`, `'unclassified'`, `'insufficient_data'`, or `'error'` |
| `evidence_fact` | string \| null | Matched factual keywords/phrases extracted from text |
| `implication_rationale` | string \| null | Economic reasoning explaining the directional hypothesis |

---

### Key Architectural Principles

1. **Non-Destructive Processing**: Raw datasets (`data/raw/`) and intermediate sentiment datasets are treated as read-only. Processed events are persisted into a dedicated destination file (`data/processed/commodity_events.jsonl`).
2. **Facts vs. Implications**: A reported factual event (e.g., a mine strike) is recorded as a factual occurrence; the market impact (`potentially_bullish`) is explicitly labeled as a potential economic hypothesis, acknowledging that market implications are not proven causal guarantees.
3. **Zero Hallucination**: If article text is empty, missing, or ambiguous, the pipeline assigns `event_type: "other"`, `impact_direction: "unknown"`, and `extraction_status: "insufficient_data"`. URLs, publication dates, and event facts are never invented.
4. **Sentiment Independence**: The event classifier evaluates operational and economic supply-chain dynamics rather than relying on sentiment scores. (For example, an accident shutting down a refinery may have negative news sentiment, but represents a bullish supply shock for the refined product.)
5. **Negation & False-Positive Handling**: Headlines negating disruptions (e.g. *"mine strike avoided"*, *"cuts ruled out"*) or containing irrelevant contexts (e.g. sports *"gold medal"*, *"crude joke"*) are filtered and classified as `other` with `unknown` impact.
6. **Commodity-Specific Nuance**: Geopolitical conflict, interest rate changes, and production quotas are evaluated in relation to the specific target commodity rather than applying a blanket bullish/bearish label.
7. **Extensibility for LLMs**: Built around `BaseEventExtractor`. A local open-source LLM extractor can be dropped in simply by subclassing `BaseEventExtractor` without changing pipelines or downstream consumers.

---

### Running Tests

All 61 unit and integration tests run **100% offline with zero external network or model download requirements**:

```bash
python -m pytest -v
```

Test coverage:
- **`test_client.py` (10 tests)**: GDELT API rate-limiting, backoff, HTTP 429/5xx, network timeouts.
- **`test_normalizer.py` (10 tests)**: Schema normalization, date parsing, URL validation.
- **`test_storage.py` (5 tests)**: JSONL persistence and `(commodity, url)` deduplication.
- **`test_pipeline.py` (4 tests)**: End-to-end ingestion and partial failure recovery.
- **`test_sentiment.py` (10 tests)**: FinBERT classification, lazy loading, batching, and error handling.
- **`test_event_extraction.py` (22 tests)**: Major event types, negation statements, sports/slang filtering, commodity-specific geopolitical/policy nuance, missing/null URLs and timestamps, exact schema preservation, and non-destructive storage.

---

### Limitations

1. **Rule-based Linguistic Scope**: The current rule-based extractor uses high-precision regex patterns tailored for English news headlines. Highly idiomatic, metaphorical, or non-English expressions are conservatively categorized as `other` / `unknown`.
2. **Headlines vs. Full Articles**: Event extraction currently operates on article headlines. Nuanced multi-event stories or secondary clauses located within article bodies will benefit from full-text crawling in future milestones.
3. **Market Complexity**: Potential market directions are first-order economic implications and do not account for broader macro regimes, inventory buffers, or offsetting shocks.