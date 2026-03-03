"""Configuration helpers for the mba-rr CLI."""

from __future__ import annotations

from dataclasses import dataclass
import os

from dotenv import load_dotenv


@dataclass(slots=True)
class Settings:
    """Runtime settings sourced from environment variables."""

    twitter_bearer_token: str


def load_settings() -> Settings:
    """Load settings from environment variables, supporting `.env` files."""
    load_dotenv()
    token = os.getenv("TWITTER_BEARER_TOKEN")
    if not token:
        raise RuntimeError(
            "TWITTER_BEARER_TOKEN is not set. Copy .env.template to .env and provide your X API bearer token."
        )
    return Settings(twitter_bearer_token=token)
