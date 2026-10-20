# AI-powered-Commodity-Price-Prediction-and-Supply-Chain-Intelligence-

An end-to-end AI platform for forecasting commodity price fluctuations and analyzing supply-chain risks.

---

## Architecture & Modules (MLOps 3 — NLP & News Intelligence)

This repository contains decoupled, production-grade intelligence components for commodity market analysis:
1. **News Ingestion & Relevance Pipeline**: Automated discovery, targeted search query construction, schema normalization, deduplication, and explainable relevance classification of global commodity news (Copper, Gold, Crude Oil) from the GDELT DOC 2.0 API.
2. **FinBERT Sentiment Analysis Pipeline**: Pretrained financial domain sentiment inference using `ProsusAI/finbert` via Hugging Face Transformers.
3. **Supply-Chain Event Extraction Pipeline**: Domain-specific rule-based extraction of operational disruptions, demand shocks, geopolitical risks, policy actions, and inventory movements with explainable market implications.
4. **Relevance Assessment & Pipeline Evaluation**: Transparent relevance classification into `directly_relevant`, `potentially_relevant` (macro drivers), `irrelevant` (sports, noise, slang), and `uncertain` (boilerplate, uninformative).

---

### Project Structure

```text
├── .gitignore                      # Ignore venv, caches, raw/processed datasets, and secrets
├── requirements.txt                # Python dependencies (requests, pytest, torch, transformers)
├── run_pipeline.py                 # News ingestion CLI runner
├── run_sentiment.py                # Sentiment analysis CLI runner
├── run_event_extraction.py         # Event extraction CLI runner
├── run_relevance_eval.py           # Relevance evaluation CLI runner
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
│   ├── news_intelligence/          # News ingestion & relevance module
│   │   ├── __init__.py             # Public module interface
│   │   ├── config.py               # Ingestion configuration & targeted commodity queries
│   │   ├── gdelt_client.py         # GDELT DOC 2.0 client (rate-limiting & retries)
│   │   ├── normalizer.py           # Field normalization & timestamp parsing
│   │   ├── relevance.py            # Explainable relevance classifier & taxonomy
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
    ├── test_relevance.py           # Unit tests for news relevance & query syntax (mocked offline)
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
Queries GDELT DOC 2.0 with targeted queries for copper, gold, and crude oil:
```bash
python run_pipeline.py
```
*Optional Flag:* `--assess-relevance` appends explainable relevance evaluation to ingested records.
*Output: `data/raw/commodity_news.jsonl`*

#### 2. Evaluate News Relevance
Evaluates ingested records against operational and macroeconomic signals:
```bash
python run_relevance_eval.py --input data/raw/commodity_news.jsonl
```

#### 3. Run FinBERT Sentiment Analysis
Enriches news records with financial sentiment (`positive`, `negative`, `neutral`):
```bash
python run_sentiment.py
```
*Output: `data/processed/commodity_news_sentiment.jsonl`*

#### 4. Run Supply-Chain Event Extraction
Extracts concrete supply-chain events and evaluates potential market impacts:
```bash
python run_event_extraction.py
```
*Output: `data/processed/commodity_events.jsonl`*

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

#### B. Relevance-Enriched Schema (Optional via `--assess-relevance`)
```json
{
  "commodity": "copper",
  "title": "Chilean copper mine strike threatens global supply",
  "url": "https://reuters.com/copper-strike",
  "publication_timestamp": "2026-10-20T10:00:00Z",
  "domain": "reuters.com",
  "language": "English",
  "source_country": "Chile",
  "fetch_timestamp": "2026-10-20T11:00:00Z",
  "provider": "gdelt",
  "relevance_class": "directly_relevant",
  "relevance_score": 0.88,
  "relevance_explanation": "Headline explicitly references target commodity and operational/market signals (mining_production, supply_disruption). Assessment based on headline analysis; full article text was not analyzed.",
  "matched_indicators": ["commodity_copper", "mining_production", "supply_disruption"]
}
```

#### C. Processed Sentiment Schema (`data/processed/commodity_news_sentiment.jsonl`)
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

#### D. Extracted Supply-Chain Event Schema (`data/processed/commodity_events.jsonl`)
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

---

### News Relevance & Query Engineering (Task 4)

#### 1. Diagnosis of Initial News Records
An audit of the initial 15 live records ingested via GDELT revealed why **100%** were classified as `other` in the event-extraction stage:
- **Foreign-Language Noise & Homonyms (73.3%)**: Because initial queries were single-word keywords (`copper`, `gold`, `"crude oil"`) without language constraints, GDELT returned articles in 9 different languages. For copper, GDELT matched Vietnamese articles mentioning `đồng` (Vietnamese currency and homonym), Russian regional athletes, French DIY insulation, and Romanian jokes.
- **Unrelated Keyword Matches (26.7%)**: For gold and crude oil, broad matching pulled event venue rentals (*"New Venues in Boston for Fall 2026 Meetings and Events"*) and generic corporate analyst stock ratings without operational context.
- **Extractor Integrity**: The downstream event extractor correctly classified these records as `other` / `unknown` rather than hallucinating fictitious supply disruptions from non-English text or venue announcements.

#### 2. Improved GDELT DOC 2.0 Query Construction
To eliminate noise, search queries in `src/news_intelligence/config.py` were upgraded to combine domain terms, negative filters, exact phrases, and English language tags:
- **Copper**:
  `copper (mining OR mine OR smelter OR cathode OR LME OR inventories OR strike OR "copper price") sourcelang:english`
- **Gold**:
  `gold ("gold price" OR bullion OR "central bank" OR reserves OR "gold mining" OR "gold futures") -olympic -"gold medal" -sports sourcelang:english`
- **Crude Oil**:
  `"crude oil" (OPEC OR production OR barrels OR refinery OR inventories OR EIA OR sanctions OR "oil price") sourcelang:english`

#### 3. Relevance Taxonomy
Articles are categorized into four transparent classes:
1. `directly_relevant` (Score: 0.80–1.00): Explicitly mentions the target commodity alongside operational, supply-chain, or price indicators (smelter cuts, mine strikes, OPEC quotas, bullion buying).
2. `potentially_relevant` (Score: 0.60–0.80): Captures macroeconomic and cross-market price drivers (Federal Reserve interest rate decisions, inflation CPI/PPI, global manufacturing PMI, US Dollar Index/DXY, trade tariffs) that materially move commodity markets even when the commodity name is omitted.
3. `irrelevant` (Score: 0.00–0.20): Filters false-positive homonyms (sports *"gold medal"*, idiomatic *"crude joke"*, domestic *"copper pipes"*, foreign currency *"đồng"*, unrelated venue announcements).
4. `uncertain` (Score: 0.40–0.50): Assigned to empty headlines, uninformative boilerplates (*"Press Release"*), or non-English text lacking domain context.

---

### Running Tests

All **80 unit and integration tests** run **100% offline with zero external network or model download requirements**:

```bash
python -m pytest -v
```

Test coverage breakdown:
- **`test_client.py` (10 tests)**: GDELT API rate-limiting, exponential backoff, HTTP 429/5xx, timeouts.
- **`test_normalizer.py` (10 tests)**: Schema normalization, date parsing, URL validation.
- **`test_storage.py` (5 tests)**: JSONL persistence and `(commodity, url)` deduplication.
- **`test_pipeline.py` (4 tests)**: End-to-end news pipeline tests and partial failure recovery.
- **`test_relevance.py` (19 tests)**: Direct commodity operational headlines, macroeconomic drivers (Fed rate cuts, CPI, PMI, DXY), sports/slang noise filtering, edge cases (empty/boilerplate), schema preservation, and pipeline relevance integration.
- **`test_sentiment.py` (10 tests)**: FinBERT classification, lazy loading, batching, and error handling.
- **`test_event_extraction.py` (22 tests)**: Major event types, negation handling, commodity-specific geopolitical/policy nuance, missing URLs/timestamps, and non-destructive storage.

---

### Limitations & Production Considerations

1. **GDELT Rate Limits**: The GDELT DOC 2.0 API enforces a strict rate limit of at most 1 request every 5 seconds per client IP. The pipeline defaults to `rate_limit_delay_seconds = 6.0` and exponential backoff (`retry_backoff_seconds = 8.0`) to avoid tripping HTTP 429 penalties.
2. **Headline-Only Analysis**: GDELT DOC 2.0 returns article titles and URLs, not full body text. All relevance assessments and event extractions explicitly acknowledge that analysis is performed on headlines and metadata.
3. **Macroeconomic Attribution**: Broad macroeconomic events (e.g. Fed rate cuts or manufacturing PMI) are classified as `potentially_relevant` because their price transmission mechanisms across commodities are mediated by market expectations.