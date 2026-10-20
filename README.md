# AI-powered-Commodity-Price-Prediction-and-Supply-Chain-Intelligence-

An end-to-end AI platform for forecasting commodity price fluctuations and analyzing supply-chain risks.

---

## News Intelligence Pipeline (MLOps 3 — NLP & News Intelligence)

This component is responsible for the automated discovery, ingestion, normalization, deduplication, and raw storage of real-time global news coverage for key commodities:
- **Copper**
- **Gold**
- **Crude Oil**

It queries the **GDELT DOC 2.0 API** (`https://api.gdeltproject.org/api/v2/doc/doc`) to monitor international news coverage across thousands of media outlets worldwide.

---

### Project Structure

```text
├── .gitignore                      # Ignore venv, caches, raw datasets, and secrets
├── requirements.txt                # Python dependencies
├── run_pipeline.py                 # Convenience CLI runner
├── data/
│   └── raw/
│       ├── .gitkeep                # Keeps raw directory tracked in git
│       └── commodity_news.jsonl    # Ingested news records (JSON Lines, git-ignored)
├── src/
│   ├── __init__.py
│   └── news_intelligence/
│       ├── __init__.py             # Public module interface
│       ├── config.py               # Pipeline configuration and commodity queries
│       ├── gdelt_client.py         # GDELT DOC 2.0 client (rate-limiting & retries)
│       ├── normalizer.py           # Field normalization & timestamp parsing
│       ├── storage.py              # JSONL persistence & (commodity, url) deduplication
│       └── pipeline.py             # Pipeline orchestrator & CLI
└── tests/
    ├── __init__.py
    ├── test_client.py              # Unit tests for GDELTClient (mocked HTTP/network)
    ├── test_normalizer.py          # Unit tests for schema normalization
    ├── test_storage.py             # Unit tests for deduplication & JSONL storage
    └── test_pipeline.py            # End-to-end pipeline tests (mocked offline)
```

---

### Installation

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

### Running the Pipeline

You can run the pipeline directly via the root runner or as a Python module:

#### 1. Basic Ingestion (Default Commodities: Copper, Gold, Crude Oil)
```bash
python run_pipeline.py
```
Or:
```bash
python -m src.news_intelligence.pipeline
```

#### 2. Advanced Options & Customization
Use command-line flags to adjust commodities, number of records, timespan window, or rate limit:
```bash
python run_pipeline.py --commodities "copper,gold,crude oil" --max-records 10 --timespan 7d --verbose
```

Available CLI arguments:
| Flag | Default | Description |
|---|---|---|
| `--commodities` | `copper,gold,crude oil` | Comma-separated list of commodities to fetch |
| `--max-records` | `25` | Maximum number of articles requested per commodity |
| `--timespan` | `7d` | Lookback window (`24h`, `7d`, `1m`) |
| `--output` | `data/raw/commodity_news.jsonl` | Destination JSONL output file path |
| `--rate-limit-delay` | `5.5` | Minimum seconds between API calls (GDELT requires ≥5.0s) |
| `--verbose` | `False` | Enable debug logging to stdout |

---

### Output Data Schema

Records are saved to `data/raw/commodity_news.jsonl` formatted as **JSON Lines** (one valid JSON object per line, UTF-8 encoded).

Each record follows this normalized schema:

| Field Name | Type | Description | Example |
|---|---|---|---|
| `commodity` | string | Target commodity name (lowercase) | `"copper"` |
| `title` | string | Article headline (stripped string) | `"Copper rallies as supply tightens"` |
| `url` | string | Article web address (unique per commodity) | `"https://reuters.com/markets/copper123"` |
| `publication_timestamp` | string \| null | Standardized ISO 8601 UTC timestamp (`YYYY-MM-DDTHH:MM:SSZ`) or `null` if unavailable | `"2026-10-15T12:00:00Z"` |
| `domain` | string | Clean domain of the publisher | `"reuters.com"` |
| `language` | string | Publisher language | `"English"` |
| `source_country` | string \| null | Country of the news source or `null` if unlisted | `"United States"` |
| `fetch_timestamp` | string | Standardized ISO 8601 UTC timestamp when ingested | `"2026-10-20T13:11:20Z"` |
| `provider` | string | Data source provider identifier | `"gdelt"` |

#### Sample Record:
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

---

### Deduplication Strategy

- Deduplication uses a composite key: `(commodity, url)`.
- **Cross-run persistence**: On every run, existing keys in `data/raw/commodity_news.jsonl` are pre-loaded into an in-memory set in `O(N)` time.
- **In-batch deduplication**: Articles within the same fetch batch sharing the same `(commodity, url)` are automatically skipped.
- **Cross-commodity flexibility**: If an article discusses both Gold and Copper, it is legitimately retained under both commodities because the key includes `commodity`.

---

### Resilience & Error Handling

1. **GDELT Rate Limits (HTTP 429)**: GDELT enforces a policy requiring at least 5 seconds between consecutive requests. The pipeline features built-in delay spacing (`5.5s`) and exponential backoff (`6s`, `12s`, `18s`) if an HTTP 429 is encountered.
2. **Transient Failures (HTTP 5xx, Timeouts)**: Automatically retried up to 3 times before declaring a query failed.
3. **Malformed Responses**: Safely handles non-JSON HTML error responses and GDELT text messages ("no matching articles").
4. **Missing Fields**: Safely parses missing or malformed dates, country codes, or titles without crashing. Records lacking a URL are rejected.
5. **Partial Failures**: If one commodity query fails (e.g. rate limit exhausted or network timeout), the pipeline logs the error for that commodity and safely proceeds to ingest the remaining commodities.

---

### Running Tests

All unit and integration tests use mocked network responses. **Tests execute completely offline without requiring an internet connection or calling the live GDELT API.**

Run the test suite:
```bash
python -m pytest -v
```

Test coverage includes:
- `test_client.py`: HTTP 200 parsing, HTTP 429 backoff/retries, HTTP 5xx retry, fatal 404, timeouts, connection errors, corrupted JSON, and empty responses.
- `test_normalizer.py`: Date format parsing (`YYYYMMDDTHHMMSSZ`, digits only, ISO), missing URL rejection, whitespace cleaning, domain fallback, and null field handling.
- `test_storage.py`: Directory creation, JSONL formatting, in-batch deduplication, cross-run deduplication, and corrupt line recovery.
- `test_pipeline.py`: End-to-end pipeline execution across all 3 commodities, partial failure isolation, and duplicate skip tracking.

---

### Limitations of GDELT DOC 2.0 API

1. **Strict Rate Limits**: GDELT DOC 2.0 allows only 1 request every 5 seconds per IP address. Rapid sequential requests from multiple threads or scripts will trigger HTTP 429 errors.
2. **Metadata Discovery Only**: GDELT returns discovery metadata (headline, URL, language, country, date), not full article body text. Downstream NLP tasks (full-text sentiment, event extraction) will require a secondary page fetcher or scraping step.
3. **Timestamp Granularity**: GDELT's `seendate` reflects the timestamp when GDELT indexed the article, which can occasionally lag the real publication time by minutes or hours.
4. **Multi-language Ingestion**: GDELT crawls articles in over 100 languages. Downstream NLP components should incorporate multilingual models or language-based filtering.