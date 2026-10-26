"""
GDELT DOC 2.0 API client with built-in rate-limiting, retries, and error handling.
"""

import json
import logging
import time
from typing import Any, Dict, List, Optional
import requests

from .config import PipelineConfig

logger = logging.getLogger(__name__)


class GDELTClientError(Exception):
    """Base exception for GDELT API client errors."""


class GDELTNetworkError(GDELTClientError):
    """Raised when network connectivity fails or times out."""


class GDELTRateLimitError(GDELTClientError):
    """Raised when GDELT rate limits the client (HTTP 429)."""


class GDELTResponseError(GDELTClientError):
    """Raised when GDELT returns an unexpected or invalid response."""


class GDELTClient:
    """
    HTTP client for querying the GDELT DOC 2.0 API.
    
    Includes rate-limiting compliance (GDELT requires >= 5s between requests)
    and robust error handling for network errors, rate limits, and malformed JSON.
    """

    def __init__(
        self,
        config: Optional[PipelineConfig] = None,
        session: Optional[requests.Session] = None,
    ):
        self.config = config or PipelineConfig()
        self.session = session or requests.Session()
        self._last_request_time: float = 0.0

    def _enforce_rate_limit(self) -> None:
        """Sleep if necessary to ensure at least rate_limit_delay_seconds between requests."""
        if self.config.rate_limit_delay_seconds <= 0:
            return

        elapsed = time.time() - self._last_request_time
        if elapsed < self.config.rate_limit_delay_seconds:
            sleep_duration = self.config.rate_limit_delay_seconds - elapsed
            logger.debug("Rate limit pause: sleeping for %.2f seconds", sleep_duration)
            time.sleep(sleep_duration)

    def fetch_articles(
        self,
        query: str,
        max_records: Optional[int] = None,
        timespan: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Fetch news articles matching a search query from GDELT DOC 2.0 API.

        Args:
            query: Search query string (e.g., 'copper', '"crude oil"').
            max_records: Maximum number of records to return (defaults to config.max_records).
            timespan: Timespan string, e.g., '24h', '7d' (defaults to config.timespan).

        Returns:
            A list of raw article dictionaries from GDELT.

        Raises:
            GDELTNetworkError: If a connection error or timeout persists.
            GDELTRateLimitError: If HTTP 429 persists after retries.
            GDELTResponseError: If server returns invalid status or malformed JSON.
        """
        params = {
            "query": query,
            "mode": self.config.mode,
            "format": self.config.output_format,
            "maxrecords": max_records or self.config.max_records,
            "sort": self.config.sort,
            "timespan": timespan or self.config.timespan,
        }

        headers = {
            "User-Agent": self.config.user_agent,
            "Accept": "application/json",
        }

        max_attempts = max(1, self.config.max_retries)
        last_exception: Optional[Exception] = None

        for attempt in range(1, max_attempts + 1):
            self._enforce_rate_limit()
            self._last_request_time = time.time()

            try:
                logger.info(
                    "Fetching GDELT articles for query '%s' (attempt %d/%d)",
                    query,
                    attempt,
                    max_attempts,
                )
                response = self.session.get(
                    self.config.api_url,
                    params=params,
                    headers=headers,
                    timeout=self.config.timeout_seconds,
                )

                # Check for HTTP 429 Too Many Requests
                if response.status_code == 429:
                    wait_time = self.config.retry_backoff_seconds * attempt
                    logger.warning(
                        "GDELT rate limit reached (HTTP 429). Backing off for %.1f seconds.",
                        wait_time,
                    )
                    time.sleep(wait_time)
                    last_exception = GDELTRateLimitError(
                        f"GDELT rate limit (HTTP 429): {response.text.strip()}"
                    )
                    continue

                # Check for transient server errors (500, 502, 503, 504)
                if response.status_code in (500, 502, 503, 504):
                    wait_time = self.config.retry_backoff_seconds * attempt
                    logger.warning(
                        "GDELT server error %d. Retrying in %.1f seconds.",
                        response.status_code,
                        wait_time,
                    )
                    time.sleep(wait_time)
                    last_exception = GDELTResponseError(
                        f"Server error {response.status_code}: {response.text[:200]}"
                    )
                    continue

                # Raise for other non-200 HTTP statuses (e.g. 400, 403, 404)
                if response.status_code != 200:
                    raise GDELTResponseError(
                        f"HTTP {response.status_code} error from GDELT: {response.text[:200]}"
                    )

                # Status is 200 OK: parse JSON payload
                text_content = response.text.strip()
                if not text_content:
                    logger.info("Empty response received for query '%s'.", query)
                    return []

                try:
                    data = response.json()
                except (json.JSONDecodeError, requests.exceptions.JSONDecodeError):
                    # Sometimes GDELT returns plain text messages like "Your query yielded no results."
                    lowered = text_content.lower()
                    if "no matching" in lowered or "no results" in lowered or "yielded no" in lowered:
                        logger.info("No articles found by GDELT for query '%s'.", query)
                        return []
                    raise GDELTResponseError(
                        f"GDELT returned non-JSON content: {text_content[:200]}"
                    )

                if not isinstance(data, dict):
                    logger.warning("Expected JSON object from GDELT, got %s", type(data))
                    return []

                articles = data.get("articles")
                if articles is None:
                    logger.info("No 'articles' field in response for query '%s'.", query)
                    return []

                if not isinstance(articles, list):
                    logger.warning(
                        "'articles' field is not a list (got %s) for query '%s'.",
                        type(articles),
                        query,
                    )
                    return []

                logger.info(
                    "Successfully fetched %d raw articles for query '%s'.",
                    len(articles),
                    query,
                )
                return articles

            except requests.exceptions.Timeout as e:
                logger.warning("Timeout connecting to GDELT: %s", e)
                last_exception = GDELTNetworkError(f"Request timed out: {e}")
                if attempt < max_attempts:
                    time.sleep(self.config.retry_backoff_seconds)
            except requests.exceptions.ConnectionError as e:
                logger.warning("Connection error connecting to GDELT: %s", e)
                last_exception = GDELTNetworkError(f"Connection failed: {e}")
                if attempt < max_attempts:
                    time.sleep(self.config.retry_backoff_seconds)
            except requests.exceptions.RequestException as e:
                logger.warning("Request exception connecting to GDELT: %s", e)
                last_exception = GDELTNetworkError(f"Request error: {e}")
                if attempt < max_attempts:
                    time.sleep(self.config.retry_backoff_seconds)

        # If loop finished without returning, raise the final exception
        if last_exception:
            raise last_exception
        raise GDELTClientError(f"Failed to fetch articles for query '{query}' after {max_attempts} attempts.")
