"""Tests for ArxivRetriever."""

import time
from types import SimpleNamespace

import feedparser

from zotero_arxiv_daily.retriever.arxiv_retriever import ArxivRetriever, _run_with_hard_timeout
import zotero_arxiv_daily.retriever.arxiv_retriever as arxiv_retriever


def _sleep_and_return(value: str, delay_seconds: float) -> str:
    time.sleep(delay_seconds)
    return value


def _raise_runtime_error() -> None:
    raise RuntimeError("boom")


def _make_fake_result(pid: str) -> SimpleNamespace:
    return SimpleNamespace(
        title=f"title {pid}",
        authors=[SimpleNamespace(name="Test Author")],
        summary="Test abstract",
        pdf_url=f"https://arxiv.org/pdf/{pid}",
        entry_id=f"https://arxiv.org/abs/{pid}",
        source_url=lambda pid=pid: f"https://arxiv.org/e-print/{pid}",
    )


def _make_feed_entry(pid: str) -> feedparser.FeedParserDict:
    return feedparser.FeedParserDict(
        id=f"oai:arXiv.org:{pid}",
        title=f"title {pid}",
        arxiv_announce_type="new",
    )


def test_arxiv_retriever(config, mock_feedparser, monkeypatch):
    monkeypatch.setattr("zotero_arxiv_daily.retriever.base.sleep", lambda _: None)

    # The RSS fixture gives us paper IDs.  After feedparser, the code calls
    # arxiv.Client().results(search) which makes real HTTP requests.  We mock
    # the arxiv Client so the test stays offline.
    new_entries = [
        e for e in mock_feedparser.entries
        if e.get("arxiv_announce_type", "new") == "new"
    ]
    paper_ids = [e.id.removeprefix("oai:arXiv.org:") for e in new_entries]

    # Build fake ArxivResult-like objects matching each RSS entry
    fake_results = []
    for entry in new_entries:
        pid = entry.id.removeprefix("oai:arXiv.org:")
        fake_results.append(SimpleNamespace(
            title=entry.title,
            authors=[SimpleNamespace(name="Test Author")],
            summary="Test abstract",
            pdf_url=f"https://arxiv.org/pdf/{pid}",
            entry_id=f"https://arxiv.org/abs/{pid}",
            source_url=lambda pid=pid: f"https://arxiv.org/e-print/{pid}",
        ))

    class FakeClient:
        def __init__(self, **kw):
            pass
        def results(self, search):
            return iter(fake_results)

    monkeypatch.setattr(arxiv_retriever.arxiv, "Client", FakeClient)

    # Skip file downloads in convert_to_paper
    monkeypatch.setattr(arxiv_retriever, "extract_text_from_html", lambda paper: None)
    monkeypatch.setattr(arxiv_retriever, "extract_text_from_pdf", lambda paper: None)
    monkeypatch.setattr(arxiv_retriever, "extract_text_from_tar", lambda paper: None)

    retriever = ArxivRetriever(config)
    papers = retriever.retrieve_papers()

    assert len(papers) == len(new_entries)
    assert set(p.title for p in papers) == set(e.title for e in new_entries)


def test_retrieve_raw_papers_fallbacks_to_per_paper_on_batch_406(config, mock_feedparser, monkeypatch):
    monkeypatch.setattr(arxiv_retriever, "sleep", lambda _: None)
    paper_ids = [f"2609.{i:05d}v1" for i in range(20)]
    mock_feedparser.entries = [_make_feed_entry(pid) for pid in paper_ids]
    result_by_id = {pid: _make_fake_result(pid) for pid in paper_ids}
    call_id_lists: list[list[str]] = []

    class FakeClient:
        def __init__(self, **kw):
            pass

        def results(self, search):
            ids = list(search.id_list)
            call_id_lists.append(ids)
            if len(ids) > 1:
                raise arxiv_retriever.arxiv.HTTPError("https://export.arxiv.org/api/query", 0, 406)
            return iter([result_by_id[ids[0]]])

    monkeypatch.setattr(arxiv_retriever.arxiv, "Client", FakeClient)
    warnings: list[str] = []
    monkeypatch.setattr(arxiv_retriever, "logger", SimpleNamespace(warning=warnings.append))

    retriever = ArxivRetriever(config)
    raw_papers = retriever._retrieve_raw_papers()

    assert len(raw_papers) == len(paper_ids)
    assert set(p.entry_id for p in raw_papers) == {f"https://arxiv.org/abs/{pid}" for pid in paper_ids}
    assert call_id_lists[0] == paper_ids
    assert all(len(ids) == 1 for ids in call_id_lists[1:])
    assert any("falling back to per-paper requests" in msg for msg in warnings)


def test_retrieve_raw_papers_fallback_skips_failed_single_paper(config, mock_feedparser, monkeypatch):
    monkeypatch.setattr(arxiv_retriever, "sleep", lambda _: None)
    paper_ids = [f"2609.{i:05d}v1" for i in range(3)]
    mock_feedparser.entries = [_make_feed_entry(pid) for pid in paper_ids]
    failed_id = paper_ids[1]
    result_by_id = {pid: _make_fake_result(pid) for pid in paper_ids if pid != failed_id}

    class FakeClient:
        def __init__(self, **kw):
            pass

        def results(self, search):
            ids = list(search.id_list)
            if len(ids) > 1:
                raise arxiv_retriever.arxiv.HTTPError("https://export.arxiv.org/api/query", 0, 406)
            if ids[0] == failed_id:
                raise arxiv_retriever.arxiv.HTTPError("https://export.arxiv.org/api/query", 0, 406)
            return iter([result_by_id[ids[0]]])

    monkeypatch.setattr(arxiv_retriever.arxiv, "Client", FakeClient)
    warnings: list[str] = []
    monkeypatch.setattr(arxiv_retriever, "logger", SimpleNamespace(warning=warnings.append))

    retriever = ArxivRetriever(config)
    raw_papers = retriever._retrieve_raw_papers()

    assert {p.entry_id for p in raw_papers} == {
        f"https://arxiv.org/abs/{paper_ids[0]}",
        f"https://arxiv.org/abs/{paper_ids[2]}",
    }
    assert any(f"Skipping arXiv paper {failed_id}" in msg for msg in warnings)


def test_retrieve_raw_papers_keeps_batch_path_when_successful(config, mock_feedparser, monkeypatch):
    monkeypatch.setattr(arxiv_retriever, "sleep", lambda _: None)
    paper_ids = [f"2609.{i:05d}v1" for i in range(20)]
    mock_feedparser.entries = [_make_feed_entry(pid) for pid in paper_ids]
    result_by_id = {pid: _make_fake_result(pid) for pid in paper_ids}
    call_id_lists: list[list[str]] = []

    class FakeClient:
        def __init__(self, **kw):
            pass

        def results(self, search):
            ids = list(search.id_list)
            call_id_lists.append(ids)
            return iter([result_by_id[pid] for pid in ids])

    monkeypatch.setattr(arxiv_retriever.arxiv, "Client", FakeClient)

    retriever = ArxivRetriever(config)
    raw_papers = retriever._retrieve_raw_papers()

    assert len(raw_papers) == len(paper_ids)
    assert call_id_lists == [paper_ids]


def test_retrieve_batch_retries_429_without_fallback(config, monkeypatch):
    slept: list[int] = []
    monkeypatch.setattr(arxiv_retriever, "sleep", lambda s: slept.append(s))

    paper_ids = ["2609.00001v1", "2609.00002v1"]
    result_by_id = {pid: _make_fake_result(pid) for pid in paper_ids}
    attempts = 0

    class FakeClient:
        def __init__(self, **kw):
            pass

        def results(self, search):
            nonlocal attempts
            attempts += 1
            if attempts < 3:
                raise arxiv_retriever.arxiv.HTTPError("https://export.arxiv.org/api/query", attempts, 429)
            return iter([result_by_id[pid] for pid in search.id_list])

    monkeypatch.setattr(arxiv_retriever.arxiv, "Client", FakeClient)

    retriever = ArxivRetriever(config)
    batch = retriever._retrieve_batch_with_retries(
        client=FakeClient(),
        batch_ids=paper_ids,
        batch_index=0,
        max_retries=5,
        retry_delay=30,
    )

    assert batch is not None
    assert [paper.entry_id for paper in batch] == [f"https://arxiv.org/abs/{pid}" for pid in paper_ids]
    assert slept == [30, 60]


def test_run_with_hard_timeout_returns_value():
    result = _run_with_hard_timeout(
        _sleep_and_return, ("done", 0.01), timeout=1, operation="test op", paper_title="paper"
    )
    assert result == "done"


def test_run_with_hard_timeout_returns_none_on_timeout(monkeypatch):
    warnings: list[str] = []
    monkeypatch.setattr(arxiv_retriever, "logger", SimpleNamespace(warning=warnings.append))
    result = _run_with_hard_timeout(
        _sleep_and_return, ("done", 1.0), timeout=0.01, operation="test op", paper_title="paper"
    )
    assert result is None
    assert "timed out" in warnings[0]


def test_run_with_hard_timeout_returns_none_on_failure(monkeypatch):
    warnings: list[str] = []
    monkeypatch.setattr(arxiv_retriever, "logger", SimpleNamespace(warning=warnings.append))
    result = _run_with_hard_timeout(
        _raise_runtime_error, (), timeout=1, operation="test op", paper_title="paper"
    )
    assert result is None
    assert "boom" in warnings[0]
