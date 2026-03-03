"""Command-line entry point for the mba-rr sentiment analyzer."""

from __future__ import annotations

import argparse
from typing import Sequence

from .config import load_settings
from .twitter_sentiment import (
    collect_company_sentiments,
    format_report,
    reports_to_json,
    write_reports_csv,
)

DEFAULT_COMPANIES = ("OpenAI", "Anthropic")

def build_parser() -> argparse.ArgumentParser:
    """Create the argument parser used by the CLI."""
    parser = argparse.ArgumentParser(
        description="Analyze recent X (Twitter) sentiment for target companies."
    )
    parser.add_argument(
        "--companies",
        nargs="+",
        default=list(DEFAULT_COMPANIES),
        metavar="NAME",
        help="Company names to scrape (defaults to OpenAI and Anthropic).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=40,
        help="Maximum tweets to analyze per company (default: 40).",
    )
    parser.add_argument(
        "--since-days",
        type=int,
        default=7,
        dest="since_days",
        help="Lookback window in days for recent tweets (default: 7).",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=3,
        help="Number of representative tweets to display per company.",
    )
    parser.add_argument(
        "--bucket-days",
        type=int,
        default=0,
        dest="bucket_days",
        help="Aggregate sentiment into N-day windows (0 disables bucketing).",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit the sentiment report as JSON instead of plain text.",
    )
    parser.add_argument(
        "--csv-path",
        dest="csv_path",
        help="Optional path to write the aggregated sentiment as CSV.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Entrypoint invoked by `python -m mba_rr` or the `mba-rr` script."""
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        settings = load_settings()
    except RuntimeError as exc:
        parser.error(str(exc))

    if args.bucket_days < 0:
        parser.error("--bucket-days must be non-negative.")
    if args.bucket_days and args.bucket_days > args.since_days:
        parser.error("--bucket-days must be less than or equal to --since-days.")

    try:
        reports = collect_company_sentiments(
            companies=args.companies,
            limit=max(1, args.limit),
            since_days=max(1, args.since_days),
            bearer_token=settings.twitter_bearer_token,
            bucket_days=args.bucket_days,
        )
    except RuntimeError as exc:
        parser.error(str(exc))

    if args.json:
        print(reports_to_json(reports, sample_limit=max(1, args.samples)))
    else:
        sample_limit = max(1, args.samples)
        for idx, report in enumerate(reports):
            if idx:
                print()
            print(format_report(report, sample_limit=sample_limit))

    if args.csv_path:
        try:
            write_reports_csv(reports, args.csv_path)
            print(f"CSV report written to {args.csv_path}")
        except RuntimeError as exc:
            parser.error(str(exc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
