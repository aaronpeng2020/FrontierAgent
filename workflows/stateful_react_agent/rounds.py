"""IterResearch-style *round* compaction for the stateful ReAct agent.

Tiered compaction (``tiered_compact``) shrinks the history only when the context
window is nearly full, so the model spends most of a long run reasoning over a
window packed with stale page dumps. ``rounds`` rewrites the history on a fixed
cadence instead. Every ``k`` turns the compactor asks the model to rewrite ONE
evolving research report from (previous report + the turns of the round just
finished) and then rebuilds the history as::

    [system, user("[Research workspace — round n]" + task + report), <recent tail>]

so every round starts from the same compact workspace: the task, everything
learned so far with its sources, and the open questions. The report is the only
memory that survives a round; the raw tool results of earlier rounds are gone,
which is the point — the agent never reasons over a window full of pages it
already read, and the per-turn context stays flat instead of growing with the
turn count (WebResearcher / IterResearch, arXiv 2509.13309).

The report is also written to ``report_path`` after every rewrite, so the
container's /workspace shows the latest state, and the reporter reads the final
one from the history at loop end.

Selected by the profile key ``agent.context_compaction: rounds`` (the react
node wires :class:`RoundsPolicy` + :class:`RoundsCompactor` together).
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Any

from frontier_agent.core.loop_types import CompactionEvent
from frontier_agent.core.messages import Message, text_of, user_msg
from frontier_agent.core.runtime.loop.compact import estimate_tokens
from frontier_agent.core.runtime.loop.compact_llm import LLMSummaryCompactor
from frontier_agent.core.runtime.loop.tiered_compact import InputTokenGauge
from frontier_agent.infra.llm.summary_prompt import format_conversation_for_summary

__all__ = ["WORKSPACE_HEADER", "RoundsCompactor", "RoundsPolicy"]

logger = logging.getLogger(__name__)

#: First line of the rebuilt user message. It is how the compactor recognises its
#: own workspace message on the next round (so the previous report is not fed to
#: the rewrite twice as "conversation") — the same substring convention the
#: spill manifest uses, since ``Message`` carries no free-form metadata.
WORKSPACE_HEADER = "[Research workspace — round"

_REWRITE_PROMPT = """\
You maintain the single evolving research report for a task. Rewrite the WHOLE \
report from the previous version and the newest round of work. This report is the \
only memory that survives: anything you leave out is lost to the researcher.

Task:
{task}

Previous report (empty on the first round):
{report}

Newest round of work (assistant reasoning, tool calls, tool results):
{conversation}

Rules:
- Keep every verified fact with its source URL and the date of the information; \
merge duplicates; resolve contradictions or flag them explicitly.
- Keep numbers exact: quote them with units and dates, never round them away.
- Record what did NOT work under "Dead ends" (queries / URLs that returned nothing \
useful) so it is not retried.
- End with "Open questions / next steps": at most 5 concrete items, most valuable first. \
If the task can already be answered, say so there and state the answer.
- Markdown, at most {max_words} words, written in the language of the task. \
Output only the report, no preamble.
"""

_WORKSPACE_TEMPLATE = """\
{header} {n}]
Earlier turns were replaced by the report below; it is your only record of the work \
so far. Tool results from before this point are gone, and re-running a call listed \
under "Dead ends" costs a round for nothing.

Task:
{task}

Current report:
{report}

Continue from "Open questions / next steps". When the report answers the task, stop \
researching and deliver the final answer.
"""


class RoundsPolicy:
    """Fire every ``k`` turns, or earlier when the real input tokens cross ``token_trigger``.

    The token trigger is the safety net for a round whose tool results are large
    (a few 100 KB page bodies can fill the window in three turns); ``k`` is the
    cadence that gives the arm its behaviour. Both ``0``/``None`` disable that
    leg. ``should_compact`` records the turn it was asked about, and
    :meth:`mark_fired` — called by the compactor, which is the only party that
    knows a compaction actually ran (forced passes bypass the policy) — restarts
    the cadence from there.
    """

    def __init__(
        self,
        k: int,
        *,
        gauge: InputTokenGauge | None = None,
        token_trigger: int = 0,
    ) -> None:
        if k <= 0:
            raise ValueError(f"RoundsPolicy: k must be positive, got {k}")
        self._k = k
        self._gauge = gauge
        self._token_trigger = int(token_trigger or 0)
        self._last_fire_turn = 0
        self._turn = 0

    @property
    def k(self) -> int:
        return self._k

    def should_compact(
        self, turn: int, messages: list[Message], estimated_tokens: int,
    ) -> bool:
        self._turn = turn
        if turn - self._last_fire_turn >= self._k:
            logger.info("rounds: turn=%d — round of %d turns complete", turn, self._k)
            return True
        if self._token_trigger > 0:
            real = self._gauge.tokens if self._gauge is not None else 0
            scale = (
                min(max(self._gauge.real_to_estimate_scale(), 1.0), 3.0)
                if self._gauge is not None else 1.0
            )
            projected = int(estimated_tokens * scale)
            if max(real, projected) > self._token_trigger:
                logger.info(
                    "rounds: turn=%d real=%d projected=%d > trigger=%d — early round",
                    turn, real, projected, self._token_trigger,
                )
                return True
        return False

    def mark_fired(self) -> None:
        self._last_fire_turn = self._turn


class RoundsCompactor:
    """Rewrite the history into ``[system, workspace(task + report), tail]``.

    ``keep_recent`` from the loop is ignored on purpose: the loop's value is
    sized for tiered compaction (five turns of raw results), while a round wants
    a tail of one or two turns so the workspace message dominates the window.
    ``keep_recent_msgs`` is the tail actually kept (orphan tool results are
    avoided the same way ``LLMSummaryCompactor`` does it).
    """

    def __init__(
        self,
        *,
        summary_llm: Any,
        task: str,
        policy: RoundsPolicy | None = None,
        report_path: str | Path | None = None,
        keep_recent_msgs: int = 6,
        max_words: int = 1500,
        timeout_s: float | None = None,
    ) -> None:
        self._llm = summary_llm
        self._task = task
        self._policy = policy
        self._report_path = Path(report_path) if report_path else None
        self._keep_recent = max(0, int(keep_recent_msgs))
        self._max_words = max(200, int(max_words))
        self._timeout_s = timeout_s
        #: The evolving report — the arm's whole memory.
        self.report: str = ""
        self.round: int = 0
        #: What the most recent ``compact`` did (read by the loop for ``on_compaction``).
        self.last_event: CompactionEvent | None = None

    async def compact(self, messages: list[Message], keep_recent: int) -> list[Message]:
        del keep_recent  # sized for tiered compaction; see the class docstring
        sys_msgs, middle, recent = LLMSummaryCompactor._partition(messages, self._keep_recent)
        tokens_before = estimate_tokens(messages)
        # The previous workspace message is the previous report, already an input
        # of the rewrite; feeding it again as "conversation" only invites the
        # summariser to copy it verbatim.
        work = [m for m in middle if not _is_workspace_msg(m)]
        self.round += 1
        rollback_reason = ""
        if work:
            try:
                self.report = await self._rewrite(work)
            except Exception as exc:
                rollback_reason = "llm_error"
                logger.warning("rounds: report rewrite failed on round %d: %s", self.round, exc)
                self.report = self._fallback(work)
        else:
            logger.info("rounds: round %d has no new work; workspace re-issued", self.round)
        workspace = user_msg(self._workspace_text())
        out: list[Message] = [*sys_msgs, workspace, *recent]
        self._write_report()
        if self._policy is not None:
            self._policy.mark_fired()
        tokens_after = estimate_tokens(out)
        self.last_event = CompactionEvent(
            turn=0, seq=0, selected="rounds", tokens_before=tokens_before,
            tokens_after=tokens_after, relief_met=tokens_after < tokens_before,
            spill_refs=0, attempts=1 if work else 0, summary=self.report,
            rollback_reason=rollback_reason,
        )
        logger.info(
            "rounds: round %d rebuilt history %d→%d msgs, %d→%d tokens (report %d chars%s)",
            self.round, len(messages), len(out), tokens_before, tokens_after,
            len(self.report), f", {rollback_reason}" if rollback_reason else "",
        )
        return out

    # The two texts a subclass changes to give the round a different memory
    # (``bank.py`` keeps an outline over an evidence index instead of a report).
    def _rewrite_prompt_text(self, work: list[Message]) -> str:
        return _REWRITE_PROMPT.format(
            task=self._task, report=self.report or "(empty)",
            conversation=format_conversation_for_summary(work), max_words=self._max_words,
        )

    def _workspace_text(self) -> str:
        return _WORKSPACE_TEMPLATE.format(
            header=WORKSPACE_HEADER, n=self.round, task=self._task,
            report=self.report or "(nothing yet)",
        )

    async def _rewrite(self, work: list[Message]) -> str:
        prompt = self._rewrite_prompt_text(work)
        call = self._llm.chat([user_msg(prompt)])
        resp = await (asyncio.wait_for(call, self._timeout_s) if self._timeout_s else call)
        text: Any = getattr(resp, "content", None) or ""
        if isinstance(text, list):
            text = "".join(c.get("text", "") if isinstance(c, dict) else str(c) for c in text)
        text = str(text).strip()
        if not text:
            raise RuntimeError("empty report")
        return text

    def _fallback(self, work: list[Message]) -> str:
        """Previous report plus a bounded raw transcript of the round: worse than a rewrite, better than losing the round."""
        raw = format_conversation_for_summary(work)
        if len(raw) > 8000:
            raw = raw[:8000] + "\n...[truncated]..."
        head = self.report or ""
        return f"{head}\n\n## Round {self.round} (rewrite failed — raw notes)\n{raw}".strip()

    def _write_report(self) -> None:
        if self._report_path is None:
            return
        try:
            self._report_path.parent.mkdir(parents=True, exist_ok=True)
            self._report_path.write_text(self.report, encoding="utf-8")
        except OSError as exc:
            logger.warning("rounds: could not write %s: %s", self._report_path, exc)


def _is_workspace_msg(m: Message) -> bool:
    return m.get("role") == "user" and text_of(m.get("content")).startswith(WORKSPACE_HEADER)
