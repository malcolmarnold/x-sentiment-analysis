"""Fetch tweets from X via the official API and run sentiment analysis."""

from __future__ import annotations

import csv
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import itertools
import json
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

import httpx
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

API_URL = "https://api.twitter.com/2/tweets/search/recent"
USER_AGENT = "mba-rr-cli/0.1"
HTTP_TIMEOUT = 10.0
EPOCH = datetime(1970, 1, 1, tzinfo=timezone.utc)


@dataclass(slots=True)
class TweetSample:
    """Normalized view of a tweet returned by the X API."""

    company: str
    content: str
    date: datetime
    url: str
    username: str
    score: float | None = None
    label: str | None = None

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serialisable representation of the tweet sample."""
        return {
            "company": self.company,
            "content": self.content,
            "date": self.date.isoformat(),
            "url": self.url,
            "username": self.username,
            "score": self.score,
            "label": self.label,
        }


@dataclass(slots=True)
class CompanySentiment:
    """Aggregate sentiment metrics for a single company."""

    company: str
    analyzed: int
    positive: int
    neutral: int
    negative: int
    average_score: float
    samples: list[TweetSample]
    bucket_start: datetime | None = None
    bucket_end: datetime | None = None

    def to_dict(self, sample_limit: int = 3) -> dict[str, object]:
        """Convert the aggregate result to a serialisable payload."""
        payload = {
            "company": self.company,
            "analyzed": self.analyzed,
            "positive": self.positive,
            "neutral": self.neutral,
            "negative": self.negative,
            "average_score": self.average_score,
            "samples": [
                sample.to_dict() for sample in itertools.islice(self.samples, sample_limit)
            ],
        }
        if self.bucket_start is not None:
            payload["bucket_start"] = self.bucket_start.isoformat()
        if self.bucket_end is not None:
            payload["bucket_end"] = self.bucket_end.isoformat()
        return payload


_ANALYZER = SentimentIntensityAnalyzer()


def label_from_score(score: float) -> str:
    """Map a compound VADER score to a coarse sentiment label."""
    if score >= 0.05:
        return "positive"
    if score <= -0.05:
        return "negative"
    return "neutral"


def summarize_company(
    company: str,
    tweets: Sequence[TweetSample],
    analyzer: SentimentIntensityAnalyzer | None = None,
    *,
    bucket_start: datetime | None = None,
    bucket_end: datetime | None = None,
) -> CompanySentiment:
    """Attach sentiment labels to tweets and aggregate company metrics."""
    if analyzer is None:
        analyzer = _ANALYZER

    positive = negative = neutral = 0
    scores: list[float] = []

    for sample in tweets:
        result = analyzer.polarity_scores(sample.content)
        score = result["compound"]
        label = label_from_score(score)
        sample.score = score
        sample.label = label
        scores.append(score)
        if label == "positive":
            positive += 1
        elif label == "negative":
            negative += 1
        else:
            neutral += 1

    analyzed = len(tweets)
    average = sum(scores) / analyzed if analyzed else 0.0
    return CompanySentiment(
        company=company,
        analyzed=analyzed,
        positive=positive,
        neutral=neutral,
        negative=negative,
        average_score=round(average, 4),
        samples=list(tweets),
        bucket_start=bucket_start,
        bucket_end=bucket_end,
    )


def _build_query(company: str) -> str:
    """Construct the X search query for the target company name."""
    return f'("{company}" OR {company}) lang:en -is:retweet -is:reply'


def _start_time_iso(since_days: int) -> str:
    since = datetime.now(timezone.utc) - timedelta(days=max(since_days, 1))
    rounded = since.replace(second=0, microsecond=0)
    if rounded < since:
        rounded += timedelta(minutes=1)
    return rounded.isoformat().replace("+00:00", "Z")


def _parse_timestamp(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def _bucket_bounds(timestamp: datetime, bucket_days: int) -> tuple[datetime, datetime]:
    ts = timestamp.astimezone(timezone.utc)
    bucket_days = max(1, bucket_days)
    delta_days = (ts - EPOCH).days
    index = delta_days // bucket_days
    start = EPOCH + timedelta(days=index * bucket_days)
    end = start + timedelta(days=bucket_days)
    return start, end


def _group_by_bucket(
    samples: Sequence[TweetSample], bucket_days: int
) -> list[tuple[tuple[datetime, datetime], list[TweetSample]]]:
    buckets: dict[tuple[datetime, datetime], list[TweetSample]] = defaultdict(list)
    for sample in samples:
        bounds = _bucket_bounds(sample.date, bucket_days)
        buckets[bounds].append(sample)
    return sorted(buckets.items(), key=lambda item: item[0][0])


def fetch_company_tweets(
    company: str,
    limit: int,
    since_days: int,
    bearer_token: str,
    http_get: Callable[[dict[str, str]], dict[str, Any]] | None = None,
) -> list[TweetSample]:
    """Fetch recent tweets matching the company name using the official API."""

    if not bearer_token:
        raise RuntimeError("A Twitter bearer token is required to fetch tweets.")

    limit = max(1, limit)
    max_results = min(max(limit, 10), 100)
    base_params = {
        "query": _build_query(company),
        "start_time": _start_time_iso(since_days),
        "max_results": str(max_results),
        "tweet.fields": "created_at,lang,author_id",
        "expansions": "author_id",
        "user.fields": "username,name",
    }
    headers = {
        "Authorization": f"Bearer {bearer_token}",
        "User-Agent": USER_AGENT,
    }

    if http_get is None:
        def _http_get(params: dict[str, str]) -> dict[str, Any]:  # pragma: no cover - exercised via integration
            try:
                response = httpx.get(API_URL, params=params, headers=headers, timeout=HTTP_TIMEOUT)
                response.raise_for_status()
            except httpx.HTTPStatusError as exc:  # pragma: no cover - network errors
                detail = exc.response.text.strip()
                raise RuntimeError(
                    f"Twitter API request failed for {company}: {exc}. Response: {detail[:500]}"
                ) from exc
            except httpx.HTTPError as exc:  # pragma: no cover - network errors
                raise RuntimeError(f"Twitter API request failed for {company}: {exc}") from exc
            return response.json()

        http_get = _http_get

    tweets: list[TweetSample] = []
    next_token: str | None = None

    while len(tweets) < limit:
        params = dict(base_params)
        if next_token:
            params["next_token"] = next_token

        payload = http_get(params)
        if errors := payload.get("errors"):
            raise RuntimeError(f"Twitter API returned errors for {company}: {errors}")

        data = payload.get("data", [])
        users = {user["id"]: user.get("username", "unknown") for user in payload.get("includes", {}).get("users", [])}

        for tweet in data:
            username = users.get(tweet.get("author_id", ""), "unknown")
            text = tweet.get("text", "").strip()
            created_at = tweet.get("created_at")
            timestamp = _parse_timestamp(created_at) if created_at else datetime.now(timezone.utc)
            tweets.append(
                TweetSample(
                    company=company,
                    content=text,
                    date=timestamp,
                    url=f"https://x.com/{username}/status/{tweet.get('id')}",
                    username=username,
                )
            )
            if len(tweets) >= limit:
                break

        meta = payload.get("meta", {})
        next_token = meta.get("next_token")
        if not next_token or not data:
            break

    return tweets


def collect_company_sentiments(
    companies: Sequence[str],
    limit: int,
    since_days: int,
    bearer_token: str,
    bucket_days: int = 0,
    fetcher: Callable[[str, int, int, str], Sequence[TweetSample]] | None = None,
    analyzer: SentimentIntensityAnalyzer | None = None,
) -> list[CompanySentiment]:
    """Collect tweets for each company and aggregate sentiment."""

    if fetcher is None:
        fetcher = fetch_company_tweets

    reports: list[CompanySentiment] = []
    reports: list[CompanySentiment] = []
    for company in companies:
        tweets = list(fetcher(company, limit, since_days, bearer_token))
        if bucket_days > 0:
            groups = _group_by_bucket(tweets, bucket_days)
            if not groups:
                reports.append(summarize_company(company, tweets, analyzer=analyzer))
            else:
                for (start, end), samples in groups:
                    reports.append(
                        summarize_company(
                            company,
                            samples,
                            analyzer=analyzer,
                            bucket_start=start,
                            bucket_end=end,
                        )
                    )
        else:
            reports.append(summarize_company(company, tweets, analyzer=analyzer))
    return reports


def format_report(report: CompanySentiment, sample_limit: int = 3) -> str:
    """Create a human-readable summary for console output."""
    lines = [
        f"Company: {report.company}",
        (
            f"  Window: {report.bucket_start.date().isoformat()} → {report.bucket_end.date().isoformat()}"
            if report.bucket_start and report.bucket_end
            else None
        ),
        f"  Tweets analyzed: {report.analyzed}",
        f"  Average score: {report.average_score:+.3f}",
        f"  Positive: {report.positive} | Neutral: {report.neutral} | Negative: {report.negative}",
    ]
    lines = [line for line in lines if line is not None]

    samples = list(itertools.islice(report.samples, sample_limit))
    if not samples:
        lines.append("  No tweets collected for this window.")
    else:
        lines.append("  Sample tweets:")
        for sample in samples:
            snippet = sample.content.replace("\n", " ")
            if len(snippet) > 110:
                snippet = snippet[:107] + "..."
            lines.append(
                f"    - ({sample.label} {sample.score:+.2f}) @{sample.username}: {snippet}"
            )
            lines.append(f"      {sample.url}")
    return "\n".join(lines)


def reports_to_json(reports: Iterable[CompanySentiment], sample_limit: int = 3) -> str:
    """Render the sentiment reports as pretty-printed JSON."""
    payload = [report.to_dict(sample_limit=sample_limit) for report in reports]
    return json.dumps(payload, indent=2)


CSV_HEADERS = (
    "company",
    "bucket_start",
    "bucket_end",
    "analyzed",
    "positive",
    "neutral",
    "negative",
    "average_score",
)


def write_reports_csv(
    reports: Sequence[CompanySentiment], path: str | Path
) -> None:
    """Persist aggregated sentiment as a CSV file."""

    dest = Path(path)
    if dest.exists() and dest.is_dir():
        raise RuntimeError(f"CSV path {dest} is a directory.")

    try:
        dest.parent.mkdir(parents=True, exist_ok=True)
        with dest.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=CSV_HEADERS)
            writer.writeheader()
            for report in reports:
                writer.writerow(
                    {
                        "company": report.company,
                        "bucket_start": report.bucket_start.isoformat()
                        if report.bucket_start
                        else "",
                        "bucket_end": report.bucket_end.isoformat()
                        if report.bucket_end
                        else "",
                        "analyzed": report.analyzed,
                        "positive": report.positive,
                        "neutral": report.neutral,
                        "negative": report.negative,
                        "average_score": report.average_score,
                    }
                )
    except OSError as exc:
        raise RuntimeError(f"Failed to write CSV report to {dest}: {exc}") from exc
