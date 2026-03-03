# mba-rr

Command-line utility that queries the official X (Twitter) Recent Search API for OpenAI and Anthropic chatter and summarizes sentiment with VADER.

The project is managed via a standard `pyproject.toml` file and assumes the [uv](https://github.com/astral-sh/uv) workflow for dependency management. Ruff handles linting, Black manages formatting, and Pytest covers the code under `tests/`.

## Configuration

1. Copy the provided template and fill in your bearer token from https://developer.twitter.com/:
   ```bash
   cp .env.template .env
   # edit .env and paste TWITTER_BEARER_TOKEN=...
   ```
2. The CLI currently only needs `TWITTER_BEARER_TOKEN`. The additional keys in the template are placeholders in case you later extend the project to OAuth 1.0a flows.

> ⚠️ Every Recent Search call consumes paid X API credits (currently about $0.005/post). Run the tool sparingly as intended.

## Quick Start

1. **Create / activate a virtual environment** (optional when using `uv`):
   ```bash
   uv venv
   source .venv/bin/activate
   ```
2. **Install dependencies**:
   ```bash
   uv pip install -e .[dev]
   ```
3. **Run the CLI** (defaults to OpenAI & Anthropic, 40 tweets each, 7-day window):
   ```bash
   uv run python -m mba_rr --samples 2
   ```
4. **Emit JSON instead of text**:
   ```bash
   uv run python -m mba_rr --json --limit 25 --since-days 3
   ```
5. **Write CSV while keeping console output**:
   ```bash
   uv run python -m mba_rr --bucket-days 7 --csv-path sentiment.csv
   ```
6. **Run quality checks**:
   ```bash
   ruff check src tests
   black --check src tests
   pytest
   ```

## Project Layout

- `src/mba_rr/cli.py` – CLI wiring and argument parsing.
- `src/mba_rr/twitter_sentiment.py` – X API client, sentiment scoring, and reporting helpers.
- `tests/` – pytest-based test suite.
- `pyproject.toml` – metadata, tooling config, and uv dev dependencies.

## Sentiment Analysis CLI

Key options:

- `--companies`: One or more company keywords (defaults to `OpenAI` and `Anthropic`).
- `--limit`: Maximum tweets requested per company (default `40`, capped by API max of `100`).
- `--since-days`: Lookback window in days (minimum `1`).
- `--samples`: Number of representative tweets to show in the summary (default `3`).
- `--bucket-days`: Aggregate sentiment into N-day windows (default `0`, meaning disabled). Must be ≤ `--since-days`.
- `--json`: Emit structured JSON rather than a formatted text report.
- `--csv-path`: Write the aggregated summary to a CSV file (console output is still shown).

Example output:

```
Company: OpenAI
  Tweets analyzed: 32
  Average score: +0.184
  Positive: 14 | Neutral: 12 | Negative: 6
  Sample tweets:
    - (positive +0.72) @researcher: OpenAI's new model is a huge leap for alignment work...
      https://twitter.com/... 
```

## Next Steps

- Extend the CLI with additional commands or subcommands as needed.
- Wire up tasks (lint, test) in `.vscode/tasks.json` or your preferred automation tool.
- Update this README with any new workflows as the project evolves.
