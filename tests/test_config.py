"""Tests for configuration loading."""

import pytest

from mba_rr import config


def test_load_settings_reads_env(monkeypatch):
    """load_settings should return the bearer token when set."""
    monkeypatch.setenv("TWITTER_BEARER_TOKEN", "abc123")
    settings = config.load_settings()
    assert settings.twitter_bearer_token == "abc123"


def test_load_settings_errors_when_missing_token(monkeypatch):
    """load_settings should raise if the bearer token is unavailable."""
    monkeypatch.delenv("TWITTER_BEARER_TOKEN", raising=False)
    monkeypatch.setattr(config, "load_dotenv", lambda: None)
    with pytest.raises(RuntimeError):
        config.load_settings()
