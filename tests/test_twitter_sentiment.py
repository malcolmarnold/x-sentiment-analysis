"""Unit tests for the twitter_sentiment helpers."""

from datetime import datetime, timedelta, timezone

import pytest

from mba_rr import twitter_sentiment as ts


class _StubAnalyzer:
    """Simple analyzer that returns deterministic compound scores."""

    def __init__(self, mapping):
        self._mapping = mapping

    def polarity_scores(self, text):
        return {"compound": self._mapping[text]}


def _sample(text: str) -> ts.TweetSample:
    return ts.TweetSample(
        company="Anthropic",
        content=text,
        date=datetime(2026, 2, 24, tzinfo=timezone.utc),
        url="https://example.com/tweet",
        username="tester",
    )


def test_label_from_score_boundaries():
    """Ensure the sentiment labels use VADER-style thresholds."""
    assert ts.label_from_score(0.2) == "positive"
    assert ts.label_from_score(-0.2) == "negative"
    assert ts.label_from_score(0.01) == "neutral"


def test_summarize_company_counts_and_average():
    """Summaries should track counts and running average."""
    samples = [_sample("great news"), _sample("meh"), _sample("bad news")]
    analyzer = _StubAnalyzer({"great news": 0.9, "meh": 0.0, "bad news": -0.6})

    report = ts.summarize_company("Anthropic", samples, analyzer=analyzer)

    assert report.company == "Anthropic"
    assert report.analyzed == 3
    assert report.positive == 1
    assert report.neutral == 1
    assert report.negative == 1
    assert report.average_score == 0.1


def test_format_report_includes_samples():
    """Formatted reports should include sample tweets when available."""
    sample = _sample("exciting launch inbound")
    sample.score = 0.5
    sample.label = "positive"
    report = ts.CompanySentiment(
        company="OpenAI",
        analyzed=1,
        positive=1,
        neutral=0,
        negative=0,
        average_score=0.5,
        samples=[sample],
    )

    text = ts.format_report(report, sample_limit=1)
    assert "Company: OpenAI" in text
    assert "exciting launch" in text
    assert "positive" in text


def test_fetch_company_tweets_handles_pagination():
    """Fetcher should follow next_token links until the limit is reached."""

    responses = [
        {
            "data": [
                {
                    "id": "1",
                    "text": "OpenAI expands its team",
                    "author_id": "10",
                    "created_at": "2026-02-23T10:00:00Z",
                },
                {
                    "id": "2",
                    "text": "Anthropic shares research",
                    "author_id": "11",
                    "created_at": "2026-02-23T11:00:00Z",
                },
            ],
            "includes": {"users": [{"id": "10", "username": "alice"}, {"id": "11", "username": "bob"}]},
            "meta": {"next_token": "NEXT"},
        },
        {
            "data": [
                {
                    "id": "3",
                    "text": "Anthropic ships Constitutional AI",
                    "author_id": "12",
                    "created_at": "2026-02-23T12:00:00Z",
                }
            ],
            "includes": {"users": [{"id": "12", "username": "carol"}]},
            "meta": {},
        },
    ]

    def fake_http_get(params):
        return responses.pop(0)

    tweets = ts.fetch_company_tweets(
        company="OpenAI",
        limit=3,
        since_days=2,
        bearer_token="token",
        http_get=fake_http_get,
    )

    assert len(tweets) == 3
    assert tweets[0].username == "alice"
    assert tweets[-1].content.startswith("Anthropic ships")


def test_fetch_company_tweets_raises_on_api_errors():
    """Fetcher should surface Twitter API error payloads."""

    def fake_http_get(params):  # noqa: ARG001
        return {"errors": [{"message": "bad auth"}]}

    with pytest.raises(RuntimeError) as excinfo:
        ts.fetch_company_tweets(
            company="OpenAI",
            limit=1,
            since_days=1,
            bearer_token="token",
            http_get=fake_http_get,
        )

    assert "bad auth" in str(excinfo.value)


def test_collect_company_sentiments_passes_bearer_token():
    """Collector should forward the bearer token to the fetcher."""

    calls = []

    def fake_fetch(company, limit, since_days, token):
        calls.append((company, token))
        sample = _sample("neutral tweet")
        return [sample]

    class _NeutralAnalyzer:
        def polarity_scores(self, text):  # noqa: ARG002
            return {"compound": 0.0}

    reports = ts.collect_company_sentiments(
        companies=["OpenAI", "Anthropic"],
        limit=1,
        since_days=1,
        bearer_token="secret",
        fetcher=fake_fetch,
        analyzer=_NeutralAnalyzer(),
    )

    assert len(reports) == 2
    assert calls == [("OpenAI", "secret"), ("Anthropic", "secret")]


def test_collect_company_sentiments_groups_into_buckets():
    """Collect should emit one report per time bucket when configured."""

    base = datetime(2026, 1, 1, tzinfo=timezone.utc)

    def fake_fetch(company, limit, since_days, token):  # noqa: ARG001
        return [
            ts.TweetSample(
                company=company,
                content="early",
                date=base,
                url="https://x.com/a",
                username="user",
            ),
            ts.TweetSample(
                company=company,
                content="late",
                date=base + timedelta(days=31),
                url="https://x.com/b",
                username="user",
            ),
        ]

    class _Analyzer:
        def polarity_scores(self, text):  # noqa: ARG002
            return {"compound": 0.1}

    reports = ts.collect_company_sentiments(
        companies=["OpenAI"],
        limit=10,
        since_days=60,
        bearer_token="token",
        bucket_days=30,
        fetcher=fake_fetch,
        analyzer=_Analyzer(),
    )

    assert len(reports) == 2
    assert reports[0].bucket_start.date().isoformat() <= "2026-01-01"
    assert reports[1].bucket_start > reports[0].bucket_start
    assert reports[0].analyzed == 1


def test_write_reports_csv(tmp_path):
    """CSV writer should emit headers and bucket metadata when available."""

    report = ts.CompanySentiment(
        company="OpenAI",
        analyzed=2,
        positive=1,
        neutral=1,
        negative=0,
        average_score=0.25,
        samples=[],
        bucket_start=datetime(2026, 1, 1, tzinfo=timezone.utc),
        bucket_end=datetime(2026, 1, 31, tzinfo=timezone.utc),
    )

    dest = tmp_path / "sentiment.csv"
    ts.write_reports_csv([report], dest)

    content = dest.read_text(encoding="utf-8").splitlines()
    assert content[0] == "company,bucket_start,bucket_end,analyzed,positive,neutral,negative,average_score"
    assert "OpenAI" in content[1]


def test_start_time_iso_rounds_to_minute(monkeypatch):
    """start_time should be truncated to minute precision for X API."""

    fixed = datetime(2026, 2, 24, 10, 15, 42, tzinfo=timezone.utc)
    monkeypatch.setattr(ts, "datetime", type("_dt", (), {"now": staticmethod(lambda tz: fixed)}))

    value = ts._start_time_iso(1)
    assert value.endswith("T10:16:00Z")
