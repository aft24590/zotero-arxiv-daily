"""Regression tests for empty RSS, bounded API fallback and durable checkpoints."""
import json
from datetime import datetime, timezone
from types import SimpleNamespace

import feedparser
import pytest

from zotero_arxiv_daily.arxiv_state import load_state, paper_id, save_state
from zotero_arxiv_daily.retriever.arxiv_retriever import ArxivRetriever
import zotero_arxiv_daily.retriever.arxiv_retriever as module


def test_state_persists_and_deduplicates_versions(monkeypatch, tmp_path):
    path = tmp_path / "checkpoint.json"
    monkeypatch.setenv("ARXIV_STATE_PATH", str(path))
    assert load_state() == (set(), None)
    save_state([SimpleNamespace(source="arxiv", url="https://arxiv.org/abs/2611.12345v1")])
    seen, checked = load_state()
    assert seen == {"2611.12345"}
    assert checked is not None and checked.tzinfo is not None
    assert paper_id("oai:arXiv.org:2611.12345v2") in seen
    save_state([SimpleNamespace(source="biorxiv", url="https://example.com/ignored")])
    assert load_state()[0] == seen


def test_bad_checkpoint_fails_closed(monkeypatch, tmp_path):
    path = tmp_path / "checkpoint.json"
    path.write_text('{"version": 99}', encoding="utf-8")
    monkeypatch.setenv("ARXIV_STATE_PATH", str(path))
    with pytest.raises(RuntimeError, match="Invalid arXiv checkpoint"):
        load_state()


def test_empty_weekday_rss_uses_bounded_date_query(config, monkeypatch, tmp_path):
    # Explicit backfill also tests this behavior on weekends.
    monkeypatch.setenv("ARXIV_BACKFILL_DAYS", "1")
    monkeypatch.delenv("ARXIV_STATE_PATH", raising=False)
    monkeypatch.setattr(module.feedparser, "parse", lambda url: feedparser.FeedParserDict(
        feed={"title": "arXiv"}, entries=[], status=200, bozo=False
    ))
    queries = []
    result = SimpleNamespace(
        title="Recovered", authors=[], summary="Abstract",
        entry_id="https://arxiv.org/abs/2611.12345v1",
        pdf_url="https://arxiv.org/pdf/2611.12345",
    )
    class Client:
        def __init__(self, **kwargs):
            pass
        def results(self, search):
            queries.append(search)
            return iter([result, result])
    monkeypatch.setattr(module.arxiv, "Client", Client)
    recovered = ArxivRetriever(config)._retrieve_raw_papers()
    assert len(recovered) == 1
    assert "submittedDate:[" in queries[0].query
    assert "cat:cs.AI" in queries[0].query


def test_existing_rss_with_only_seen_items_skips_api(config, mock_feedparser, monkeypatch, tmp_path):
    monkeypatch.delenv("ARXIV_BACKFILL_DAYS", raising=False)
    checkpoint = tmp_path / "state.json"
    checkpoint.write_text(json.dumps({
        "version": 1,
        "seen_ids": [paper_id(e.id) for e in mock_feedparser.entries],
        "last_completed_utc": datetime.now(timezone.utc).isoformat(),
    }), encoding="utf-8")
    monkeypatch.setenv("ARXIV_STATE_PATH", str(checkpoint))
    class Client:
        def __init__(self, **kwargs):
            pass
        def results(self, search):
            raise AssertionError("unexpected arXiv API call")
    monkeypatch.setattr(module.arxiv, "Client", Client)
    assert ArxivRetriever(config)._retrieve_raw_papers() == []


def test_empty_rss_api_failure_is_not_silent(config, monkeypatch):
    monkeypatch.setenv("ARXIV_BACKFILL_DAYS", "1")
    monkeypatch.delenv("ARXIV_STATE_PATH", raising=False)
    monkeypatch.setattr(module.feedparser, "parse", lambda url: feedparser.FeedParserDict(
        feed={"title": "arXiv"}, entries=[], status=200, bozo=False
    ))
    class Client:
        def __init__(self, **kwargs):
            pass
        def results(self, search):
            return iter([])
    monkeypatch.setattr(module.arxiv, "Client", Client)
    with pytest.raises(RuntimeError, match="returned zero results"):
        ArxivRetriever(config)._retrieve_raw_papers()
