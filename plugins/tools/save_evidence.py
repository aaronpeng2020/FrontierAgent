"""save_evidence — file one quote into the evidence bank (the ``bank`` harness arm).

Runs in-process like ``recover_result``: the bank directory is chosen by the
react node, the model never names a path, and nothing here touches the sandbox.
"""

from __future__ import annotations

import logging

from frontier_agent.core.tool import tool
from plugins.tools._evidence import MAX_NOTE_CHARS, MAX_QUOTE_CHARS, load_index, save_entry

logger = logging.getLogger(__name__)


@tool
async def save_evidence(url: str, quote: str, note: str, date: str = "", title: str = "") -> str:
    """Save one piece of evidence to the evidence bank and get back its id (E1, E2, ...).

    Call it right after reading a page that contains something the answer will
    rest on — a number, a date, a rule, a statement — before you move on. One
    call per distinct fact; the quote must be copied verbatim from the page, not
    paraphrased. Later rounds only see the bank's index and your outline, so an
    unsaved fact is a lost fact. Cite saved ids as [E3] in your notes and answer.

    Args:
        url: The page the quote comes from (the exact URL you fetched).
        quote: The verbatim passage, at most 2000 characters. Include the numbers
            and dates as written on the page.
        note: One line, in your own words, saying what this evidence establishes
            and why it matters for the task. At most 400 characters.
        date: When the information is from (publication date or the date the
            figure refers to), ISO ``YYYY-MM-DD`` when known. Empty when unknown.
        title: Optional page or document title.

    Returns:
        The new id and the size of the bank, or an explanation of what was missing.
    """
    if not str(quote or "").strip():
        return "save_evidence: quote is required — copy the passage verbatim from the page."
    if not str(url or "").strip():
        return "save_evidence: url is required — the page the quote comes from."
    if not str(note or "").strip():
        return "save_evidence: note is required — one line on what this evidence establishes."
    q = str(quote)
    clipped = " (quote clipped to 2000 chars)" if len(q.strip()) > MAX_QUOTE_CHARS else ""
    entry, created = save_entry(
        url=str(url), quote=q, note=str(note)[:MAX_NOTE_CHARS], date=str(date or ""), title=str(title or ""),
    )
    total = len(load_index())
    if not created:
        return f"{entry['id']} already holds this quote from that url — cite [{entry['id']}]. Bank: {total} entries."
    return f"{entry['id']} saved{clipped} — cite it as [{entry['id']}]. Bank: {total} entries."


__all__ = ["save_evidence"]
