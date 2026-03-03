"""Tests for the mba_rr.cli module."""

from dataclasses import dataclass
from datetime import datetime, timezone

import pytest

from mba_rr import cli
from mba_rr.twitter_sentiment import CompanySentiment, TweetSample


def _fake_reports() -> list[CompanySentiment]:
    sample = TweetSample(
        company="OpenAI",
        content="OpenAI releases a new model",
        date=datetime(2026, 2, 24, tzinfo=timezone.utc),
        url="https://example.com/tweet",
        username="researcher",
        score=0.92,
        label="positive",
    )
    report = CompanySentiment(
        company="OpenAI",
        analyzed=1,
        positive=1,
        neutral=0,
        negative=0,
        average_score=0.92,
        samples=[sample],
    )
    return [report]


@dataclass
class _Settings:
    twitter_bearer_token: str = "token"


def test_cli_prints_plain_text_report(monkeypatch, capsys):
    """CLI should emit a human-readable report by default."""

    def stub_collect(companies, limit, since_days, bearer_token, bucket_days):  # noqa: ARG001
        assert bearer_token == "token"
        assert bucket_days == 0
        return _fake_reports()

    monkeypatch.setattr(cli, "load_settings", lambda: _Settings())
    monkeypatch.setattr(cli, "collect_company_sentiments", stub_collect)

    exit_code = cli.main([])
    captured = capsys.readouterr()
    assert exit_code == 0
    assert "Company: OpenAI" in captured.out
    assert "Tweets analyzed" in captured.out


def test_cli_can_emit_json(monkeypatch, capsys):
    """CLI should serialize the report as JSON when requested."""

    def stub_collect(companies, limit, since_days, bearer_token, bucket_days):  # noqa: ARG001
        assert bearer_token == "token"
        assert bucket_days == 10
        return _fake_reports()

    monkeypatch.setattr(cli, "load_settings", lambda: _Settings())
    monkeypatch.setattr(cli, "collect_company_sentiments", stub_collect)

    exit_code = cli.main(["--json", "--samples", "1", "--bucket-days", "10", "--since-days", "10"])
    captured = capsys.readouterr()
    assert exit_code == 0
    assert '"company": "OpenAI"' in captured.out
    assert '"samples"' in captured.out


def test_cli_rejects_invalid_bucket_days(monkeypatch):
    """Args parser should enforce bucket <= since-days."""

    monkeypatch.setattr(cli, "load_settings", lambda: _Settings())

    with pytest.raises(SystemExit):
        cli.main(["--bucket-days", "8", "--since-days", "7"])


def test_cli_writes_csv_when_path_provided(monkeypatch, tmp_path):
    """CLI should forward reports to the CSV writer when requested."""

    called = {}

    def stub_collect(companies, limit, since_days, bearer_token, bucket_days):  # noqa: ARG001
        return _fake_reports()

    def stub_write(reports, path):
        called["path"] = path
        called["count"] = len(reports)

    monkeypatch.setattr(cli, "load_settings", lambda: _Settings())
    monkeypatch.setattr(cli, "collect_company_sentiments", stub_collect)
    monkeypatch.setattr(cli, "write_reports_csv", stub_write)

    output = tmp_path / "report.csv"
    exit_code = cli.main(["--csv-path", str(output)])
    assert exit_code == 0
    assert called["path"] == str(output)
    assert called["count"] == 1


def test_cli_surfaces_csv_errors(monkeypatch):
    """CSV write failures should terminate with a SystemExit."""

    def stub_collect(companies, limit, since_days, bearer_token, bucket_days):  # noqa: ARG001
        return _fake_reports()

    def stub_write(reports, path):  # noqa: ARG001
        raise RuntimeError("disk full")

    monkeypatch.setattr(cli, "load_settings", lambda: _Settings())
    monkeypatch.setattr(cli, "collect_company_sentiments", stub_collect)
    monkeypatch.setattr(cli, "write_reports_csv", stub_write)

    with pytest.raises(SystemExit):
        cli.main(["--csv-path", "report.csv"])
