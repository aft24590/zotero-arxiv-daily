from .base import BaseRetriever, register_retriever
import arxiv
from arxiv import Result as ArxivResult
from ..protocol import Paper
from ..utils import extract_markdown_from_pdf, extract_tex_code_from_tar
from tempfile import TemporaryDirectory
import feedparser
from tqdm import tqdm
import multiprocessing
import os
from queue import Empty
from dataclasses import dataclass
from time import monotonic, sleep
from typing import Any, Callable, TypeVar
from loguru import logger
import requests

T = TypeVar("T")

DOWNLOAD_TIMEOUT = (10, 60)
PDF_EXTRACT_TIMEOUT = 180
TAR_EXTRACT_TIMEOUT = 180

ARXIV_BATCH_SIZE = 20
ARXIV_CLIENT_DELAY_SECONDS = 3
ARXIV_MAX_RETRIES = 3
ARXIV_RETRY_DELAY_SECONDS = 30
RETRYABLE_ARXIV_STATUSES = {429, 500, 502, 503, 504}
MIN_HEALTH_CHECK_CANDIDATES = 20
MIN_RETRIEVAL_SUCCESS_RATE = 0.80


@dataclass
class ArxivRetrievalStats:
    candidates: int = 0
    retrieved: int = 0
    batch_requests: int = 0
    batch_failures: int = 0
    fallback_requests: int = 0
    fallback_failures: int = 0
    circuit_open: bool = False
    circuit_status: int | None = None
    circuit_reason: str | None = None
    elapsed_seconds: float = 0.0

    @property
    def success_rate(self) -> float:
        if self.candidates == 0:
            return 1.0
        return self.retrieved / self.candidates


@dataclass
class IndividualRetrievalResult:
    papers: list[ArxivResult]
    requests: int
    failures: int
    circuit_status: int | None = None


def _download_file(url: str, path: str) -> None:
    with requests.get(url, stream=True, timeout=DOWNLOAD_TIMEOUT) as response:
        response.raise_for_status()
        with open(path, "wb") as file:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    file.write(chunk)


def _run_in_subprocess(
    result_queue: Any,
    func: Callable[..., T | None],
    args: tuple[Any, ...],
) -> None:
    try:
        result_queue.put(("ok", func(*args)))
    except Exception as exc:
        result_queue.put(("error", f"{type(exc).__name__}: {exc}"))


def _run_with_hard_timeout(
    func: Callable[..., T | None],
    args: tuple[Any, ...],
    *,
    timeout: float,
    operation: str,
    paper_title: str,
) -> T | None:
    start_methods = multiprocessing.get_all_start_methods()
    context = multiprocessing.get_context("fork" if "fork" in start_methods else start_methods[0])
    result_queue = context.Queue()
    process = context.Process(target=_run_in_subprocess, args=(result_queue, func, args))
    process.start()

    try:
        status, payload = result_queue.get(timeout=timeout)
    except Empty:
        if process.is_alive():
            process.kill()
        process.join(5)
        result_queue.close()
        result_queue.join_thread()
        logger.warning(f"{operation} timed out for {paper_title} after {timeout} seconds")
        return None

    process.join(5)
    result_queue.close()
    result_queue.join_thread()

    if status == "ok":
        return payload

    logger.warning(f"{operation} failed for {paper_title}: {payload}")
    return None


def _extract_text_from_pdf_worker(pdf_url: str) -> str:
    with TemporaryDirectory() as temp_dir:
        path = os.path.join(temp_dir, "paper.pdf")
        _download_file(pdf_url, path)
        return extract_markdown_from_pdf(path)


def _extract_text_from_html_worker(html_url: str) -> str | None:
    import trafilatura

    downloaded = trafilatura.fetch_url(html_url)
    if downloaded is None:
        raise ValueError(f"Failed to download HTML from {html_url}")
    text = trafilatura.extract(downloaded, include_comments=False, include_tables=False)
    if not text:
        raise ValueError(f"No text extracted from {html_url}")
    return text


def _extract_text_from_tar_worker(source_url: str, paper_id: str, paper_title: str | None = None) -> str | None:
    with TemporaryDirectory() as temp_dir:
        path = os.path.join(temp_dir, "paper.tar.gz")
        _download_file(source_url, path)
        file_contents = extract_tex_code_from_tar(path, paper_id, paper_title=paper_title)
        if not file_contents or "all" not in file_contents:
            raise ValueError("Main tex file not found.")
        return file_contents["all"]


@register_retriever("arxiv")
class ArxivRetriever(BaseRetriever):
    def __init__(self, config):
        super().__init__(config)
        if self.config.source.arxiv.category is None:
            raise ValueError("category must be specified for arxiv.")

    def _retrieve_raw_papers(self) -> list[ArxivResult]:
        started = monotonic()
        client = arxiv.Client(
            num_retries=0,
            delay_seconds=ARXIV_CLIENT_DELAY_SECONDS,
        )
        query = '+'.join(self.config.source.arxiv.category)
        include_cross_list = self.config.source.arxiv.get("include_cross_list", False)

        feed = feedparser.parse(f"https://rss.arxiv.org/atom/{query}")
        if 'Feed error for query' in feed.feed.title:
            raise Exception(f"Invalid ARXIV_QUERY: {query}.")

        allowed_announce_types = {"new", "cross"} if include_cross_list else {"new"}
        all_paper_ids = [
            i.id.removeprefix("oai:arXiv.org:")
            for i in feed.entries
            if i.get("arxiv_announce_type", "new") in allowed_announce_types
        ]
        if self.config.executor.debug:
            all_paper_ids = all_paper_ids[:10]

        stats = ArxivRetrievalStats(candidates=len(all_paper_ids))
        raw_papers: list[ArxivResult] = []
        bar = tqdm(total=len(all_paper_ids))

        for i in range(0, len(all_paper_ids), ARXIV_BATCH_SIZE):
            batch_ids = all_paper_ids[i:i + ARXIV_BATCH_SIZE]
            batch_index = i // ARXIV_BATCH_SIZE
            stats.batch_requests += 1

            batch, status = self._retrieve_batch_with_retries(
                client=client,
                batch_ids=batch_ids,
                batch_index=batch_index,
                max_retries=ARXIV_MAX_RETRIES,
                retry_delay=ARXIV_RETRY_DELAY_SECONDS,
            )

            if batch is not None:
                raw_papers.extend(batch)
                stats.retrieved += len(batch)
                bar.update(len(batch_ids))
            else:
                stats.batch_failures += 1

                if status in RETRYABLE_ARXIV_STATUSES:
                    stats.circuit_open = True
                    stats.circuit_status = status
                    stats.circuit_reason = (
                        f"batch {batch_index} exhausted retries with HTTP {status}"
                    )
                    logger.warning(
                        f"Opening arXiv circuit after batch {batch_index} exhausted retries "
                        f"with HTTP {status}; remaining candidates will not be requested"
                    )
                    break

                probe_id = batch_ids[0]
                stats.fallback_requests += 1
                probe, probe_status = self._retrieve_single_paper_once(
                    client=client,
                    paper_id=probe_id,
                )

                if probe is not None:
                    raw_papers.append(probe)
                    stats.retrieved += 1

                    fallback = self._retrieve_individual_papers(
                        client=client,
                        paper_ids=batch_ids[1:],
                        max_retries=ARXIV_MAX_RETRIES,
                        retry_delay=ARXIV_RETRY_DELAY_SECONDS,
                    )
                    raw_papers.extend(fallback.papers)
                    stats.retrieved += len(fallback.papers)
                    stats.fallback_requests += fallback.requests
                    stats.fallback_failures += fallback.failures

                    if fallback.circuit_status is not None:
                        stats.circuit_open = True
                        stats.circuit_status = fallback.circuit_status
                        stats.circuit_reason = (
                            f"individual fallback exhausted retries with HTTP "
                            f"{fallback.circuit_status}"
                        )
                        logger.warning(
                            f"Opening arXiv circuit after individual fallback exhausted retries "
                            f"with HTTP {fallback.circuit_status}; remaining candidates will not "
                            "be requested"
                        )

                    bar.update(len(batch_ids))
                    if stats.circuit_open:
                        break
                else:
                    stats.fallback_failures += 1
                    if probe_status == status or probe_status in RETRYABLE_ARXIV_STATUSES:
                        stats.circuit_open = True
                        stats.circuit_status = probe_status
                        stats.circuit_reason = (
                            f"batch {batch_index} failed with HTTP {status} and single-paper "
                            f"probe failed with HTTP {probe_status}"
                        )
                        logger.warning(
                            f"Opening arXiv circuit: batch {batch_index} failed with HTTP {status} "
                            f"and probe {probe_id} failed with HTTP {probe_status}; remaining "
                            "candidates will not be requested"
                        )
                        break

                    fallback = self._retrieve_individual_papers(
                        client=client,
                        paper_ids=batch_ids[1:],
                        max_retries=ARXIV_MAX_RETRIES,
                        retry_delay=ARXIV_RETRY_DELAY_SECONDS,
                    )
                    raw_papers.extend(fallback.papers)
                    stats.retrieved += len(fallback.papers)
                    stats.fallback_requests += fallback.requests
                    stats.fallback_failures += fallback.failures
                    bar.update(len(batch_ids))

                    if fallback.circuit_status is not None:
                        stats.circuit_open = True
                        stats.circuit_status = fallback.circuit_status
                        stats.circuit_reason = (
                            f"individual fallback exhausted retries with HTTP "
                            f"{fallback.circuit_status}"
                        )
                        logger.warning(
                            f"Opening arXiv circuit after individual fallback exhausted retries "
                            f"with HTTP {fallback.circuit_status}; remaining candidates will not "
                            "be requested"
                        )
                        break

            if i + ARXIV_BATCH_SIZE < len(all_paper_ids):
                sleep(ARXIV_CLIENT_DELAY_SECONDS)

        bar.close()
        stats.elapsed_seconds = monotonic() - started
        logger.info(
            "ArXiv retrieval summary: "
            f"candidates={stats.candidates}, retrieved={stats.retrieved}, "
            f"success_rate={stats.success_rate:.1%}, batch_requests={stats.batch_requests}, "
            f"batch_failures={stats.batch_failures}, fallback_requests={stats.fallback_requests}, "
            f"fallback_failures={stats.fallback_failures}, circuit_open={stats.circuit_open}, "
            f"circuit_status={stats.circuit_status}, elapsed_seconds={stats.elapsed_seconds:.1f}"
        )

        if (
            stats.candidates >= MIN_HEALTH_CHECK_CANDIDATES
            and stats.success_rate < MIN_RETRIEVAL_SUCCESS_RATE
        ):
            logger.warning(
                "Discarding degraded arXiv retrieval result: "
                f"retrieved {stats.retrieved}/{stats.candidates} papers "
                f"({stats.success_rate:.1%}), below the "
                f"{MIN_RETRIEVAL_SUCCESS_RATE:.0%} health threshold"
            )
            return []

        return raw_papers

    def _retrieve_batch_with_retries(
        self,
        *,
        client: arxiv.Client,
        batch_ids: list[str],
        batch_index: int,
        max_retries: int,
        retry_delay: int,
    ) -> tuple[list[ArxivResult] | None, int | None]:
        search = arxiv.Search(id_list=batch_ids)
        for attempt in range(max_retries):
            try:
                return list(client.results(search)), None
            except arxiv.HTTPError as exc:
                status = getattr(exc, "status", None)
                if status in RETRYABLE_ARXIV_STATUSES and attempt < max_retries - 1:
                    wait = retry_delay * (attempt + 1)
                    logger.warning(
                        f"arXiv API {status} on batch {batch_index}, "
                        f"retry {attempt + 1}/{max_retries} in {wait}s"
                    )
                    sleep(wait)
                    continue

                if status in RETRYABLE_ARXIV_STATUSES:
                    logger.warning(
                        f"arXiv batch {batch_index} exhausted {max_retries} attempts "
                        f"with HTTP {status}"
                    )
                else:
                    logger.warning(
                        f"arXiv batch request failed for batch {batch_index} with HTTP {status}; "
                        "probing a single paper before fallback"
                    )
                return None, status

        return None, None

    def _retrieve_single_paper_once(
        self,
        *,
        client: arxiv.Client,
        paper_id: str,
    ) -> tuple[ArxivResult | None, int | None]:
        try:
            results = list(client.results(arxiv.Search(id_list=[paper_id])))
        except arxiv.HTTPError as exc:
            return None, getattr(exc, "status", None)

        if not results:
            logger.warning(f"No arXiv paper found for {paper_id}; skipping")
            return None, None
        return results[0], None

    def _retrieve_individual_papers(
        self,
        *,
        client: arxiv.Client,
        paper_ids: list[str],
        max_retries: int,
        retry_delay: int,
    ) -> IndividualRetrievalResult:
        papers: list[ArxivResult] = []
        requests = 0
        failures = 0

        for index, paper_id in enumerate(paper_ids):
            requests += 1
            search = arxiv.Search(id_list=[paper_id])

            for attempt in range(max_retries):
                try:
                    results = list(client.results(search))
                    if results:
                        papers.append(results[0])
                    else:
                        failures += 1
                        logger.warning(f"No arXiv paper found for {paper_id}; skipping")
                    break
                except arxiv.HTTPError as exc:
                    status = getattr(exc, "status", None)
                    if status in RETRYABLE_ARXIV_STATUSES and attempt < max_retries - 1:
                        wait = retry_delay * (attempt + 1)
                        logger.warning(
                            f"arXiv API {status} on paper {paper_id}, "
                            f"retry {attempt + 1}/{max_retries} in {wait}s"
                        )
                        sleep(wait)
                        continue

                    failures += 1
                    if status in RETRYABLE_ARXIV_STATUSES:
                        logger.warning(
                            f"arXiv paper {paper_id} exhausted {max_retries} attempts "
                            f"with HTTP {status}"
                        )
                        return IndividualRetrievalResult(
                            papers=papers,
                            requests=requests,
                            failures=failures,
                            circuit_status=status,
                        )

                    logger.warning(f"Skipping arXiv paper {paper_id} after API error: {exc}")
                    break

            if index < len(paper_ids) - 1:
                sleep(1)

        return IndividualRetrievalResult(
            papers=papers,
            requests=requests,
            failures=failures,
        )

    def convert_to_paper(self, raw_paper: ArxivResult) -> Paper:
        title = raw_paper.title
        authors = [a.name for a in raw_paper.authors]
        abstract = raw_paper.summary
        pdf_url = raw_paper.pdf_url
        full_text = extract_text_from_tar(raw_paper)
        if full_text is None:
            full_text = extract_text_from_html(raw_paper)
        if full_text is None:
            full_text = extract_text_from_pdf(raw_paper)
        return Paper(
            source=self.name,
            title=title,
            authors=authors,
            abstract=abstract,
            url=raw_paper.entry_id,
            pdf_url=pdf_url,
            full_text=full_text,
        )


def extract_text_from_html(paper: ArxivResult) -> str | None:
    html_url = paper.entry_id.replace("/abs/", "/html/")
    try:
        return _extract_text_from_html_worker(html_url)
    except Exception as exc:
        logger.warning(f"HTML extraction failed for {paper.title}: {exc}")
        return None


def extract_text_from_pdf(paper: ArxivResult) -> str | None:
    if paper.pdf_url is None:
        logger.warning(f"No PDF URL available for {paper.title}")
        return None
    return _run_with_hard_timeout(
        _extract_text_from_pdf_worker,
        (paper.pdf_url,),
        timeout=PDF_EXTRACT_TIMEOUT,
        operation="PDF extraction",
        paper_title=paper.title,
    )


def extract_text_from_tar(paper: ArxivResult) -> str | None:
    source_url = paper.source_url()
    if source_url is None:
        logger.warning(f"No source URL available for {paper.title}")
        return None
    return _run_with_hard_timeout(
        _extract_text_from_tar_worker,
        (source_url, paper.entry_id, paper.title),
        timeout=TAR_EXTRACT_TIMEOUT,
        operation="Tar extraction",
        paper_title=paper.title,
    )
