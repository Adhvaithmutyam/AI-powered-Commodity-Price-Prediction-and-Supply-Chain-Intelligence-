"""
Unit tests for GDELTClient using mocked responses.
"""

from unittest.mock import MagicMock, patch
import pytest
import requests

from src.news_intelligence.config import PipelineConfig
from src.news_intelligence.gdelt_client import (
    GDELTClient,
    GDELTNetworkError,
    GDELTRateLimitError,
    GDELTResponseError,
)


@pytest.fixture
def mock_config():
    """Config with rate limiting and retry delays disabled for fast testing."""
    return PipelineConfig(
        rate_limit_delay_seconds=0.0,
        retry_backoff_seconds=0.0,
        max_retries=2,
        timeout_seconds=5,
    )


def test_fetch_articles_success(mock_config):
    mock_session = MagicMock()
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.text = '{"articles": [{"title": "Copper rises", "url": "https://example.com/copper1"}]}'
    mock_response.json.return_value = {
        "articles": [{"title": "Copper rises", "url": "https://example.com/copper1"}]
    }
    mock_session.get.return_value = mock_response

    client = GDELTClient(config=mock_config, session=mock_session)
    articles = client.fetch_articles(query="copper", max_records=10)

    assert len(articles) == 1
    assert articles[0]["title"] == "Copper rises"
    assert articles[0]["url"] == "https://example.com/copper1"
    mock_session.get.assert_called_once()


def test_fetch_articles_rate_limit_retry_then_success(mock_config):
    mock_session = MagicMock()
    
    # First response: 429 Too Many Requests
    resp_429 = MagicMock()
    resp_429.status_code = 429
    resp_429.text = "Please limit requests to one every 5 seconds"

    # Second response: 200 OK
    resp_200 = MagicMock()
    resp_200.status_code = 200
    resp_200.text = '{"articles": [{"title": "Gold hits record", "url": "https://example.com/gold1"}]}'
    resp_200.json.return_value = {
        "articles": [{"title": "Gold hits record", "url": "https://example.com/gold1"}]
    }

    mock_session.get.side_effect = [resp_429, resp_200]

    with patch("time.sleep"):  # Prevent actual sleep during tests
        client = GDELTClient(config=mock_config, session=mock_session)
        articles = client.fetch_articles(query="gold")

    assert len(articles) == 1
    assert articles[0]["title"] == "Gold hits record"
    assert mock_session.get.call_count == 2


def test_fetch_articles_rate_limit_exhausted(mock_config):
    mock_session = MagicMock()
    resp_429 = MagicMock()
    resp_429.status_code = 429
    resp_429.text = "Please limit requests"
    mock_session.get.return_value = resp_429

    with patch("time.sleep"):
        client = GDELTClient(config=mock_config, session=mock_session)
        with pytest.raises(GDELTRateLimitError):
            client.fetch_articles(query="gold")

    assert mock_session.get.call_count == mock_config.max_retries


def test_fetch_articles_server_error_retry(mock_config):
    mock_session = MagicMock()
    resp_503 = MagicMock()
    resp_503.status_code = 503
    resp_503.text = "Service Unavailable"

    resp_200 = MagicMock()
    resp_200.status_code = 200
    resp_200.text = '{"articles": []}'
    resp_200.json.return_value = {"articles": []}

    mock_session.get.side_effect = [resp_503, resp_200]

    with patch("time.sleep"):
        client = GDELTClient(config=mock_config, session=mock_session)
        articles = client.fetch_articles(query="crude oil")

    assert articles == []
    assert mock_session.get.call_count == 2


def test_fetch_articles_fatal_http_404(mock_config):
    mock_session = MagicMock()
    resp_404 = MagicMock()
    resp_404.status_code = 404
    resp_404.text = "Not Found"
    mock_session.get.return_value = resp_404

    client = GDELTClient(config=mock_config, session=mock_session)
    with pytest.raises(GDELTResponseError) as exc_info:
        client.fetch_articles(query="copper")

    assert "HTTP 404 error" in str(exc_info.value)


def test_fetch_articles_network_timeout(mock_config):
    mock_session = MagicMock()
    mock_session.get.side_effect = requests.exceptions.Timeout("Connection timed out")

    with patch("time.sleep"):
        client = GDELTClient(config=mock_config, session=mock_session)
        with pytest.raises(GDELTNetworkError):
            client.fetch_articles(query="copper")


def test_fetch_articles_connection_error(mock_config):
    mock_session = MagicMock()
    mock_session.get.side_effect = requests.exceptions.ConnectionError("Failed to resolve host")

    with patch("time.sleep"):
        client = GDELTClient(config=mock_config, session=mock_session)
        with pytest.raises(GDELTNetworkError):
            client.fetch_articles(query="copper")


def test_fetch_articles_no_matching_text_message(mock_config):
    mock_session = MagicMock()
    resp = MagicMock()
    resp.status_code = 200
    resp.text = "Your query yielded no results."
    resp.json.side_effect = requests.exceptions.JSONDecodeError("Expecting value", "", 0)
    mock_session.get.return_value = resp

    client = GDELTClient(config=mock_config, session=mock_session)
    articles = client.fetch_articles(query="obscure_term")

    assert articles == []


def test_fetch_articles_corrupted_json(mock_config):
    mock_session = MagicMock()
    resp = MagicMock()
    resp.status_code = 200
    resp.text = "<html>Error occurred</html>"
    resp.json.side_effect = requests.exceptions.JSONDecodeError("Expecting value", "", 0)
    mock_session.get.return_value = resp

    client = GDELTClient(config=mock_config, session=mock_session)
    with pytest.raises(GDELTResponseError) as exc_info:
        client.fetch_articles(query="copper")

    assert "non-JSON content" in str(exc_info.value)


def test_fetch_articles_missing_articles_key(mock_config):
    mock_session = MagicMock()
    resp = MagicMock()
    resp.status_code = 200
    resp.text = "{}"
    resp.json.return_value = {}
    mock_session.get.return_value = resp

    client = GDELTClient(config=mock_config, session=mock_session)
    articles = client.fetch_articles(query="copper")

    assert articles == []
