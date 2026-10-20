# AI-powered-Commodity-Price-Prediction-and-Supply-Chain-Intelligence-

An end-to-end AI platform for forecasting commodity price fluctuations and analyzing supply-chain risks.

---

## Architecture & Modules (MLOps 3 — NLP & News Intelligence)

This repository contains two core intelligence components for commodity market analysis:
1. **News Ingestion Pipeline**: Automated discovery, ingestion, normalization, and deduplication of global commodity news (Copper, Gold, Crude Oil) from the GDELT DOC 2.0 API.
2. **FinBERT Sentiment Analysis Pipeline**: Pretrained financial domain sentiment inference using `ProsusAI/finbert` via Hugging Face Transformers.

---

### Project Structure

```text
├── .gitignore                      # Ignore venv, caches, raw/processed datasets, and secrets
├── requirements.txt                # Python dependencies (requests, pytest, torch, transformers)
├── run_pipeline.py                 # News ingestion CLI runner
├── run_sentiment.py                # Sentiment analysis CLI runner
├── data/
│   ├── raw/
│   │   ├── .gitkeep                # Tracks directory in git
│   │   └── commodity_news.jsonl    # Raw ingested news records (JSON Lines, git-ignored)
│   └── processed/
│       ├── .gitkeep                # Tracks directory in git
│       └── commodity_news_sentiment.jsonl # Enriched sentiment records (JSON Lines, git-ignored)
├── src/
│   ├── __init__.py
│   ├── news_intelligence/          # News ingestion module
│   │   ├── __init__.py             # Public module interface
│   │   ├── config.py               # Ingestion configuration and commodity queries
│   │   ├── gdelt_client.py         # GDELT DOC 2.0 client (rate-limiting & retries)
│   │   ├── normalizer.py           # Field normalization & timestamp parsing
│   │   ├── storage.py              # JSONL persistence & (commodity, url) deduplication
│   │   └── pipeline.py             # Pipeline orchestrator & CLI
│   └── sentiment/                  # Sentiment analysis module
│       ├── __init__.py             # Public module interface
│       ├── config.py               # Sentiment configuration & compute device settings
│       ├── analyzer.py             # FinBERT analyzer (lazy loading, batching, safe fallbacks)
│       └── pipeline.py             # Sentiment pipeline orchestrator & CLI
└── tests/
    ├── __init__.py
    ├── test_client.py              # Unit tests for GDELTClient (mocked HTTP/network)
    ├── test_normalizer.py          # Unit tests for schema normalization
    ├── test_storage.py             # Unit tests for deduplication & JSONL storage
    ├── test_pipeline.py            # End-to-end news pipeline tests (mocked offline)
    └── test_sentiment.py           # Unit tests for FinBERT sentiment (mocked offline)
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

### 1. Running the News Ingestion Pipeline

Ingests articles for **Copper**, **Gold**, and **Crude Oil** from GDELT DOC 2.0 and saves them to `data/raw/commodity_news.jsonl`.

#### Basic Ingestion
```bash
python run_pipeline.py
```
Or as a module:
```bash
python -m src.news_intelligence.pipeline
```

#### Ingestion CLI Arguments
| Flag | Default | Description |
|---|---|---|
| `--commodities` | `copper,gold,crude oil` | Comma-separated list of commodities to fetch |
| `--max-records` | `25` | Maximum number of articles requested per commodity |
| `--timespan` | `7d` | Lookback window (`24h`, `7d`, `1m`) |
| `--output` | `data/raw/commodity_news.jsonl` | Destination JSONL output file path |
| `--rate-limit-delay` | `5.5` | Minimum seconds between API calls (GDELT requires ≥5.0s) |
| `--verbose` | `False` | Enable debug logging to stdout |

---

### 2. Running FinBERT Sentiment Analysis

Enriches existing raw news records with financial sentiment classifications using `ProsusAI/finbert`.
Reads from `data/raw/commodity_news.jsonl` and writes enriched records to `data/processed/commodity_news_sentiment.jsonl`. **Never modifies or overwrites raw data.**

#### Basic Sentiment Run
```bash
python run_sentiment.py
```
Or as a module:
```bash
python -m src.sentiment.pipeline
```

#### Sentiment CLI Arguments
| Flag | Default | Description |
|---|---|---|
| `--input` | `data/raw/commodity_news.jsonl` | Source raw news JSONL file path |
| `--output` | `data/processed/commodity_news_sentiment.jsonl` | Destination processed JSONL file path |
| `--model-name` | `ProsusAI/finbert` | Hugging Face model repository identifier |
| `--batch-size` | `16` | Inference batch size |
| `--device` | Auto-detect | Compute device (`cpu` or `cuda`) |
| `--verbose` | `False` | Enable debug logging to stdout |

---

### Hardware, CPU/GPU & First-time Model Download

- **First-Time Download**: On its first live execution, `transformers` will download the `ProsusAI/finbert` model weights (~440 MB) from the Hugging Face Hub. Files are cached locally in `~/.cache/huggingface/hub/` and reused across subsequent executions.
- **Hardware Acceleration**:
  - The pipeline automatically detects whether an NVIDIA GPU with CUDA is available (`torch.cuda.is_available()`).
  - If a GPU is present, inference runs on device `cuda:0` for maximum throughput.
  - If no GPU is available, inference transparently defaults to `cpu`.
- **Memory Safety & Batching**: Text inputs are processed in chunks of 16 (configurable via `--batch-size`) with sequence truncation capped at 512 tokens to prevent out-of-memory (OOM) errors.
- **Offline / Test Independence**: Unit tests use mock pipelines and **never** trigger model downloads or internet requests.

---

### Data Schemas

#### A. Raw News Record (`data/raw/commodity_news.jsonl`)
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

#### B. Processed Sentiment Record (`data/processed/commodity_news_sentiment.jsonl`)
Preserves all original fields and appends 4 normalized sentiment fields:

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

#### Sentiment Fields Description:
| Field Name | Type | Allowed Values / Description |
|---|---|---|
| `sentiment_label` | string | `'positive'`, `'negative'`, `'neutral'`, or `'unavailable'` |
| `sentiment_score` | float \| null | Model prediction confidence score between 0.0 and 1.0 (e.g. `0.9543`), or `null` if unavailable |
| `sentiment_model` | string | Identifier of the model used (`"ProsusAI/finbert"`) |
| `sentiment_status` | string | Execution outcome: `'success'`, `'empty_text'`, or error description |

---

### Modular Sentiment API Usage

The sentiment module is decoupled and can be imported directly into future services or feature engineering pipelines:

```python
from src.sentiment import FinBERTSentimentAnalyzer

analyzer = FinBERTSentimentAnalyzer()

# Analyze a single headline:
result = analyzer.analyze_text("Gold prices rally sharply as dollar softens")
print(result)
# Output: {'sentiment_label': 'positive', 'sentiment_score': 0.9543, 'sentiment_model': 'ProsusAI/finbert', 'sentiment_status': 'success'}

# Analyze an existing dictionary:
record = {"commodity": "gold", "title": "Fed holds rates steady"}
enriched = analyzer.enrich_record(record)
```

---

### Running Tests

All unit and integration tests run **100% offline with zero external network or model download requirements**:

```bash
python -m pytest -v
```

Test coverage includes:
- **`test_client.py`**: GDELT client retries, HTTP 429 backoff, 5xx handling, timeouts, malformed responses.
- **`test_normalizer.py`**: Date conversions, missing URLs, schema standardizations.
- **`test_storage.py`**: JSONL appending, directory creation, cross-run deduplication.
- **`test_pipeline.py`**: End-to-end news ingestion orchestration and partial failure handling.
- **`test_sentiment.py`**: FinBERT classification for positive/negative/neutral labels, empty/whitespace handling, batch inference, model loading error recovery, and end-to-end enrichment without modifying raw files.

---

### Limitations

1. **Headline vs. Full Article Text**: The sentiment model currently analyzes article headlines. While headlines are strong indicators for high-frequency price movement, subtle nuances in long-form articles require full-body text extraction.
2. **Language Coverage**: `ProsusAI/finbert` is pretrained on English financial texts (Financial PhraseBank). For non-English news headlines, predictions typically classify as neutral or have lower reliability unless pre-translated.
3. **GDELT Rate Limits**: The upstream news ingestion relies on GDELT DOC 2.0, which enforces a strict 1 request every 5 seconds per IP limit.