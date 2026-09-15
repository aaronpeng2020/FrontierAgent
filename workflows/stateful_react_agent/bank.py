"""Evidence-bank harness (``agent.context_compaction: bank``).

The ``rounds`` arm keeps one evolving report as the agent's memory. This arm
splits that memory in two, the way a human researcher keeps a card file and an
outline:

* **the bank** — every useful quote goes to disk through the ``save_evidence``
  tool the moment it is read (``plugins/tools/_evidence.py``), with its URL,
  date and a one-line note; it never has to survive a compaction because it
  never lived in the context;
* **the outline** — what the round compaction rewrites: a dynamic outline of
  the final answer whose bullets cite bank ids (``[E7]``), plus open questions
  and dead ends. The workspace message rebuilt every round is
  ``task + bank index + outline``, so the agent always sees what it has and what
  is still missing, never the pages themselves.

At loop end :class:`BankReportObserver` writes the report section by section:
each section's LLM call gets only the evidence it cites, loaded back from the
files, then one polish pass makes the sections read as one document. The
agent's own final ```json block (the lab's forecast contract) is carried over
verbatim. Everything fails open to the agent's own final answer.
"""

from __future__ import annotations

import asyncio
import logging
import re
from pathlib import Path
from typing import Any

from frontier_agent.core.loop_types import AgentLoopResult, BaseObserver
from frontier_agent.core.messages import Message, assistant_msg_with_reasoning, user_msg
from frontier_agent.infra.llm.summary_prompt import format_conversation_for_summary
from plugins.tools._evidence import (
    OUTLINE_NAME,
    cited_ids,
    index_text,
    load_index,
    read_entry,
)
from workflows._shared.sdk_shim import ReporterDeltaEmitter
from workflows.stateful_react_agent.rounds import WORKSPACE_HEADER, RoundsCompactor

__all__ = ["BANK_PROMPT_ADDENDUM", "BankCompactor", "BankReportObserver"]

logger = logging.getLogger(__name__)

BANK_PROMPT_ADDENDUM = """

## Evidence bank
Your context is rebuilt every few turns; only the evidence bank and your outline survive. So:
- After every page that contains something the answer will rest on, call `save_evidence` with the \
verbatim quote, the URL, the date and a one-line note — before you search again. One call per fact. \
A fact you did not save is a fact you will have to find again.
- Cite bank ids as [E3] in your notes and in the final answer. Never cite a page you did not save.
- `read_file /workspace/evidence/E3.md` reopens a saved quote when you need its exact wording.
- The final report is assembled from the bank by an editor: it can only contain what the bank holds. \
Your last message must still state your conclusion and the required output format in full.
"""

_OUTLINE_PROMPT = """\
You maintain the outline of the final answer to a research task. Rewrite the WHOLE outline from \
the previous version, the evidence bank index and the newest round of work. The outline plus the \
bank are the only memory that survives: anything you leave out is lost to the researcher.

Task:
{task}

Evidence bank (id | date | what it shows | url) — the quotes themselves are on disk:
{index}

Previous outline (empty on the first round):
{outline}

Newest round of work (assistant reasoning, tool calls, tool results):
{conversation}

Rules:
- Markdown. One "## " heading per section of the eventual answer, in answer order, each followed \
by bullets stating what that section will say; every bullet that states a fact or number cites \
the bank id(s) that back it, like [E4]. Keep numbers exact with units and dates.
- A fact that appears in the round's work but is NOT in the bank cannot be cited: list it under \
"## To save" (url + what to save) so the researcher banks it next round.
- Keep contradictions between sources visible, with both ids.
- "## Dead ends": queries / URLs that returned nothing useful, so they are not retried.
- End with "## Open questions / next steps": at most 5 concrete items, most valuable first. If \
the task can already be answered, say so there and state the answer.
- At most {max_words} words, written in the language of the task. Output only the outline.
"""

_WORKSPACE_TEMPLATE = """\
{header} {n}]
Earlier turns were replaced by the bank index and the outline below; they are your only record \
of the work so far. Tool results from before this point are gone. The quotes are on disk: \
read_file /workspace/evidence/E<n>.md reopens one.

Task:
{task}

Evidence bank ({count} entries — id | date | what it shows | url):
{index}

Outline:
{outline}

Continue from "Open questions / next steps" and bank anything under "To save". When the outline \
answers the task, stop researching and deliver the final answer, citing bank ids.
"""

_SECTION_PROMPT = """\
You are writing ONE section of a research report. Write only this section, from the evidence \
given below and nothing else; every factual sentence cites its evidence id in square brackets, \
like [E4]. Quote numbers and dates exactly as the evidence states them. If the outline claims \
something the evidence below does not support, say what the evidence does show instead — do not \
invent support. No heading, no preamble, at most {max_words} words, in {language}.

Task:
{task}

Section: {title}
Outline for this section:
{body}

Evidence available for this section:
{evidence}
"""

_POLISH_PROMPT = """\
Below is a research report assembled section by section, followed by the researcher's own \
closing message. Produce the final report:
- Keep every section and every [E<n>] citation exactly where it is; you may reorder sentences, \
remove repetition between sections, fix transitions, and add a two-to-four sentence summary at the top.
- Do not add facts that are not in the sections. Do not drop numbers or dates.
- If the closing message contains a required structured answer (for example a ```json block), \
reproduce that block verbatim at the end of the report.
- Write in {language}. Output only the report.

Task:
{task}

Assembled sections:
{draft}

Researcher's closing message:
{closing}
"""

_JSON_BLOCK_RE = re.compile(r"```json\s*\{[\s\S]*?\}\s*```")
_SKIP_SECTIONS = ("open questions", "dead ends", "to save", "next steps")


class BankCompactor(RoundsCompactor):
    """Round compaction whose memory is ``bank index + outline`` instead of a report."""

    def __init__(self, *, evidence_root: str | Path, **kwargs: Any) -> None:
        root = Path(evidence_root)
        kwargs.setdefault("report_path", root / OUTLINE_NAME)
        super().__init__(**kwargs)
        self._root = root

    @property
    def outline(self) -> str:
        return self.report

    def _rewrite_prompt_text(self, work: list[Message]) -> str:
        return _OUTLINE_PROMPT.format(
            task=self._task,
            index=index_text(load_index(self._root)) or "(empty — nothing saved yet)",
            outline=self.report or "(empty)",
            conversation=format_conversation_for_summary(work),
            max_words=self._max_words,
        )

    def _workspace_text(self) -> str:
        entries = load_index(self._root)
        return _WORKSPACE_TEMPLATE.format(
            header=WORKSPACE_HEADER, n=self.round, task=self._task, count=len(entries),
            index=index_text(entries) or "(empty — nothing saved yet)",
            outline=self.report or "(nothing yet)",
        )


def _split_sections(outline: str) -> list[tuple[str, str]]:
    """``## Title`` sections of the outline that belong in the answer (bookkeeping ones dropped)."""
    out: list[tuple[str, str]] = []
    title, body = "", []
    for line in (outline or "").splitlines():
        if line.startswith("## "):
            if title:
                out.append((title, "\n".join(body).strip()))
            title, body = line[3:].strip(), []
        elif title:
            body.append(line)
    if title:
        out.append((title, "\n".join(body).strip()))
    return [(t, b) for t, b in out if not any(k in t.lower() for k in _SKIP_SECTIONS)]


def _last_json_block(text: str) -> str:
    hits = _JSON_BLOCK_RE.findall(text or "")
    return hits[-1] if hits else ""


def _content_text(resp: Any) -> str:
    text: Any = getattr(resp, "content", None) or ""
    if isinstance(text, list):
        text = "".join(c.get("text", "") if isinstance(c, dict) else str(c) for c in text)
    return str(text).strip()


class BankReportObserver(BaseObserver):
    """Write the final report from the bank at loop end: one LLM call per outline section, one polish."""

    critical = True

    def __init__(
        self,
        *,
        llm: Any,
        task: str,
        evidence_root: str | Path,
        language: str = "English",
        timeout: float | None = None,
        emitter: Any | None = None,
        usage_aggregator: Any | None = None,
        thinking_format: str = "tag",
        section_max_words: int = 350,
        max_sections: int = 12,
        max_evidence_chars: int = 24_000,
    ) -> None:
        self._llm = llm
        self._task = task
        self._root = Path(evidence_root)
        self._language = language
        self._timeout = timeout
        self._emitter = emitter
        self._usage_aggregator = usage_aggregator
        self._thinking_format = thinking_format
        self._section_max_words = max(120, int(section_max_words))
        self._max_sections = max(1, int(max_sections))
        self._max_evidence_chars = max(2_000, int(max_evidence_chars))
        self.sections_written = 0

    async def on_loop_end(self, result: AgentLoopResult) -> None:
        if getattr(result, "stopped_by", "") == "paused":
            return
        baseline = str(result.metadata.get("final_answer") or result.final_content or "")
        entries = load_index(self._root)
        if not entries:
            logger.info("bank: empty evidence bank — keeping the agent's own answer")
            return
        outline = self._read_outline() or self._outline_from_history(result.messages)
        sections = _split_sections(outline)[: self._max_sections]
        if not sections:
            logger.info("bank: no outline sections — keeping the agent's own answer")
            return
        by_id = {e["id"]: e for e in entries}
        try:
            drafted: list[str] = []
            for title, body in sections:
                evidence = self._evidence_for(body, by_id)
                if not evidence:
                    logger.info("bank: section %r cites no banked evidence — skipped", title)
                    continue
                text = await self._chat(_SECTION_PROMPT.format(
                    task=self._task, title=title, body=body or "(no bullets)", evidence=evidence,
                    max_words=self._section_max_words, language=self._language,
                ))
                if text:
                    drafted.append(f"## {title}\n{text}")
                    self.sections_written += 1
            if not drafted:
                logger.info("bank: nothing drafted — keeping the agent's own answer")
                return
            report = await self._chat(_POLISH_PROMPT.format(
                task=self._task, draft="\n\n".join(drafted), closing=baseline[:8_000] or "(none)",
                language=self._language,
            ))
            if not report:
                raise RuntimeError("empty polish")
        except Exception as exc:
            logger.warning("bank: report assembly failed (%s) — keeping the agent's own answer", exc)
            return
        block = _last_json_block(baseline)
        if block and block not in report:
            report = f"{report.rstrip()}\n\n{block}"
        report = f"{report.rstrip()}\n\n{self._references(report, by_id)}"

        delta = ReporterDeltaEmitter(self._emitter)
        delta.start()
        delta.stream_output(report)
        history = list(result.messages)
        if history and history[-1].get("role") == "user":
            history.pop()
        history.append(user_msg("Write the final report from the evidence bank."))
        history.append(assistant_msg_with_reasoning(report, "", thinking_format=self._thinking_format))
        result.messages = history
        result.metadata["final_answer"] = report
        result.metadata["final_answer_source"] = "bank_writer"
        result.metadata["final_answer_rescued"] = False
        result.metadata["final_answer_rescue_mode"] = ""
        result.metadata["bank_sections"] = self.sections_written
        result.metadata["bank_entries"] = len(entries)
        result.final_content = report
        logger.info("bank: report written from %d sections over %d entries", self.sections_written, len(entries))

    # ── helpers ──────────────────────────────────────────────────────────────

    def _read_outline(self) -> str:
        try:
            return (self._root / OUTLINE_NAME).read_text(encoding="utf-8").strip()
        except OSError:
            return ""

    @staticmethod
    def _outline_from_history(messages: list[Message]) -> str:
        """The outline of the last workspace message, when no round ever wrote the file."""
        for m in reversed(messages):
            if m.get("role") != "user":
                continue
            text = str(m.get("content") or "")
            if text.startswith(WORKSPACE_HEADER) and "\nOutline:\n" in text:
                return text.split("\nOutline:\n", 1)[1].split("\n\nContinue from", 1)[0].strip()
        return ""

    def _evidence_for(self, body: str, by_id: dict[str, dict[str, Any]]) -> str:
        parts: list[str] = []
        used = 0
        for eid in cited_ids(body):
            if eid not in by_id:
                continue
            text = read_entry(eid, self._root).strip()
            if used + len(text) > self._max_evidence_chars:
                break
            parts.append(text)
            used += len(text)
        return "\n\n".join(parts)

    @staticmethod
    def _references(report: str, by_id: dict[str, dict[str, Any]]) -> str:
        lines = ["## References"]
        for eid in cited_ids(report):
            e = by_id.get(eid)
            if e is None:
                continue
            label = e.get("title") or e.get("note") or ""
            date = f" ({e['date']})" if e.get("date") else ""
            lines.append(f"- [{eid}] {label}{date} — {e.get('url', '')}")
        return "\n".join(lines) if len(lines) > 1 else ""

    async def _chat(self, prompt: str) -> str:
        call = self._llm.chat([user_msg(prompt)])
        resp = await (asyncio.wait_for(call, self._timeout) if self._timeout else call)
        return _content_text(resp)
