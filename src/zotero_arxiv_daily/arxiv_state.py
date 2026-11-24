"""A small, durable checkpoint for arXiv recommendations in GitHub Actions.

The workflow restores this JSON from its latest successful artifact. The file is
written only after the email completes (or a confirmed empty run). A dry run
never advances the checkpoint.
"""
import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path


def paper_id(value: str) -> str:
    """Normalize an Atom ID/URL; versions of the same paper share one key."""
    value = str(value).removeprefix("oai:arXiv.org:").rsplit("/", 1)[-1]
    return re.sub(r"v[0-9]+$", "", value)


def load_state() -> tuple[set[str], datetime | None]:
    path_value = os.environ.get("ARXIV_STATE_PATH")
    if not path_value or not Path(path_value).exists():
        return set(), None
    try:
        data = json.loads(Path(path_value).read_text(encoding="utf-8"))
        if data.get("version") != 1 or not isinstance(data.get("seen_ids"), list):
            raise ValueError("unsupported checkpoint schema")
        seen = set(data["seen_ids"])
        checked = data.get("last_completed_utc")
        last = datetime.fromisoformat(checked.replace("Z", "+00:00")) if checked else None
        if last is not None and last.tzinfo is None:
            raise ValueError("checkpoint timestamp is not timezone-aware")
        return seen, last
    except (ValueError, TypeError, OSError) as exc:
        raise RuntimeError(f"Invalid arXiv checkpoint: {path_value}: {exc}") from exc


def save_state(papers: list) -> None:
    path_value = os.environ.get("ARXIV_STATE_PATH")
    if not path_value:
        return
    seen, _ = load_state()
    seen.update(
        paper_id(p.url)
        for p in papers
        if getattr(p, "source", None) == "arxiv" and getattr(p, "url", None)
    )
    path = Path(path_value)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "version": 1,
        "seen_ids": sorted(seen),
        "last_completed_utc": datetime.now(timezone.utc).isoformat(),
    }
    temp_path = path.with_suffix(path.suffix + ".tmp")
    temp_path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    os.replace(temp_path, path)
