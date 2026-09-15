"""The evidence bank behind ``save_evidence`` and the ``bank`` compaction arm.

One directory (``<workspace>/evidence`` by default) holds one Markdown file per
saved piece of evidence — ``E1.md``, ``E2.md``, … — plus ``index.json`` listing
them. The ids are what the agent cites in its outline and what the bank writer
resolves back to the files when it drafts each section, so the report is built
from the saved quotes rather than from what survived in the context window.

Where the bank lives is decided once per run by the react node (``EVIDENCE_DIR``
contextvar); ``FRONTIER_AGENT_EVIDENCE_DIR`` is the fallback for tools that run
without a node (tests, ad-hoc scripts), then ``/workspace/evidence`` resolved the
way every other filesystem tool resolves the workspace.
"""

from __future__ import annotations

import contextvars
import json
import logging
import os
import re
import threading
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

EVIDENCE_DIR: contextvars.ContextVar[str] = contextvars.ContextVar(
    "frontier_agent_evidence_dir", default="",
)
INDEX_NAME = "index.json"
OUTLINE_NAME = "outline.md"
MAX_QUOTE_CHARS = 2_000
MAX_NOTE_CHARS = 400
_ID_RE = re.compile(r"\bE(\d{1,4})\b")
_lock = threading.Lock()


def evidence_dir() -> Path:
    raw = EVIDENCE_DIR.get() or os.environ.get("FRONTIER_AGENT_EVIDENCE_DIR", "").strip()
    if not raw:
        from plugins.tools._sandbox import resolve_runtime_path

        raw = os.path.join(str(resolve_runtime_path("/workspace")), "evidence")
    return Path(raw)


def load_index(root: Path | None = None) -> list[dict[str, Any]]:
    path = (root or evidence_dir()) / INDEX_NAME
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return []
    return [e for e in data if isinstance(e, dict) and e.get("id")] if isinstance(data, list) else []


def _write_index(root: Path, entries: list[dict[str, Any]]) -> None:
    root.mkdir(parents=True, exist_ok=True)
    tmp = root / f".{INDEX_NAME}.tmp"
    tmp.write_text(json.dumps(entries, ensure_ascii=False, indent=1), encoding="utf-8")
    tmp.replace(root / INDEX_NAME)


def save_entry(
    *, url: str, quote: str, note: str, date: str = "", title: str = "", root: Path | None = None,
) -> tuple[dict[str, Any], bool]:
    """Append one entry; returns ``(entry, created)``. The same url+quote is not stored twice."""
    root = root or evidence_dir()
    url, quote, note = url.strip(), quote.strip()[:MAX_QUOTE_CHARS], note.strip()[:MAX_NOTE_CHARS]
    date, title = date.strip()[:40], title.strip()[:200]
    with _lock:
        entries = load_index(root)
        for e in entries:
            if e.get("url") == url and str(e.get("quote", "")).strip() == quote:
                return e, False
        n = max((int(e["id"][1:]) for e in entries if _ID_RE.fullmatch(str(e["id"]))), default=0) + 1
        entry = {"id": f"E{n}", "url": url, "date": date, "title": title, "note": note, "quote": quote}
        root.mkdir(parents=True, exist_ok=True)
        (root / f"E{n}.md").write_text(render_entry(entry), encoding="utf-8")
        entries.append(entry)
        _write_index(root, entries)
    return entry, True


def render_entry(e: dict[str, Any]) -> str:
    head = f"# {e['id']}" + (f" — {e['title']}" if e.get("title") else "")
    quote = "\n".join(f"> {line}" for line in str(e.get("quote", "")).splitlines() or [""])
    return (
        f"{head}\n- url: {e.get('url', '')}\n- date: {e.get('date') or 'unknown'}\n"
        f"- note: {e.get('note', '')}\n\n{quote}\n"
    )


def read_entry(eid: str, root: Path | None = None) -> str:
    root = root or evidence_dir()
    try:
        return (root / f"{eid}.md").read_text(encoding="utf-8")
    except OSError:
        for e in load_index(root):
            if e.get("id") == eid:
                return render_entry(e)
    return ""


def index_text(entries: list[dict[str, Any]], *, max_chars: int = 8_000) -> str:
    """One line per entry — what the workspace message shows the agent between rounds."""
    lines = []
    for e in entries:
        label = e.get("title") or e.get("note") or ""
        note = f" — {e['note']}" if e.get("title") and e.get("note") else ""
        lines.append(f"{e['id']} | {e.get('date') or '?'} | {label}{note} | {e.get('url', '')}")
    out = "\n".join(lines)
    return out if len(out) <= max_chars else out[:max_chars] + "\n…(index truncated)"


def cited_ids(text: str) -> list[str]:
    """Evidence ids mentioned in ``text``, in first-seen order."""
    seen: list[str] = []
    for m in _ID_RE.finditer(text or ""):
        eid = f"E{int(m.group(1))}"
        if eid not in seen:
            seen.append(eid)
    return seen


__all__ = [
    "EVIDENCE_DIR", "INDEX_NAME", "OUTLINE_NAME", "cited_ids", "evidence_dir", "index_text",
    "load_index", "read_entry", "render_entry", "save_entry",
]
