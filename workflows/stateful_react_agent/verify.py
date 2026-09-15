"""Local verifier (``agent.local_verifier: true``): a side audit every ``k`` turns.

Every ``k`` turns a separate LLM call reads the turns of the round just finished
and the task, and answers one question: is anything going wrong that the agent
cannot see from inside the loop — looping on the same query, taking a snippet
for a page, building on a number no second source confirms, treating a pending
event as settled? If so it writes at most ``max_words`` of direction, which the
loop injects as a user message before the next turn (``Intervention.inject_messages``);
if not, it answers ``OK`` and nothing is injected.

The verifier uses the same model as the agent — it is a harness arm, not a
stronger judge — and it only looks at the window, so it is cheap: one extra call
per round, no tools. The global check (are the answer's claims supported by the
pages it cites) is done after the run by the review pipeline, not here.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from frontier_agent.core.loop_types import BaseObserver, Intervention, TurnContext
from frontier_agent.core.messages import Message, user_msg
from frontier_agent.infra.llm.summary_prompt import format_conversation_for_summary

__all__ = ["LocalVerifierObserver"]

logger = logging.getLogger(__name__)

_AUDIT_PROMPT = """\
You audit a research agent every few steps. You see the task and the agent's last {k} turns \
(its reasoning, tool calls and tool results). Today is turn {turn}.

Task:
{task}

Last turns:
{conversation}

Look only for problems the agent cannot see from inside its loop:
1. Loops: the same or near-same query/URL repeated, or a dead site retried.
2. Snippets taken for pages: a key fact taken from a search snippet without opening the page.
3. Single-sourced numbers or dates that the answer will rest on.
4. Date sanity: a scheduled or pending event treated as already happened; a source older than the \
question needs; a "current" figure that is actually stale.
5. Drift: work that does not serve the task or its resolution rules as written.

If the round was productive and none of these apply, answer exactly: OK
Otherwise write at most {max_words} words of concrete direction to the agent — what to stop, what \
to check, which page to open — most important first, no preamble, no praise.
"""


class LocalVerifierObserver(BaseObserver):
    """Every ``k`` turns, audit the round and inject a short steer when needed."""

    critical = True

    def __init__(
        self,
        *,
        llm: Any,
        task: str,
        k: int = 8,
        max_words: int = 200,
        timeout_s: float | None = None,
        max_notes: int = 50,
    ) -> None:
        if k <= 0:
            raise ValueError(f"LocalVerifierObserver: k must be positive, got {k}")
        self._llm = llm
        self._task = task
        self._k = int(k)
        self._max_words = max(40, int(max_words))
        self._timeout = timeout_s
        self._max_notes = max(0, int(max_notes))
        self._last_turn = 0
        #: Every note injected so far (for tests and trajectory inspection).
        self.notes: list[str] = []

    async def on_turn_end(self, ctx: TurnContext) -> Intervention | None:
        if ctx.turn - self._last_turn < self._k:
            return None
        self._last_turn = ctx.turn
        if len(self.notes) >= self._max_notes:
            return None
        window = _last_turns(list(ctx.messages or []), self._k)
        if not any(m.get("role") == "tool" for m in window):
            return None  # nothing was researched this round; nothing to audit
        prompt = _AUDIT_PROMPT.format(
            k=self._k, turn=ctx.turn, task=self._task,
            conversation=format_conversation_for_summary(window), max_words=self._max_words,
        )
        try:
            call = self._llm.chat([user_msg(prompt)])
            resp = await (asyncio.wait_for(call, self._timeout) if self._timeout else call)
        except Exception as exc:
            logger.warning("verifier: audit at turn %d failed: %s", ctx.turn, exc)
            return None
        text: Any = getattr(resp, "content", None) or ""
        if isinstance(text, list):
            text = "".join(c.get("text", "") if isinstance(c, dict) else str(c) for c in text)
        text = str(text).strip()
        if not text or text.upper().rstrip(".!") == "OK" or text.upper().startswith("OK\n"):
            logger.info("verifier: turn %d OK", ctx.turn)
            return None
        note = f"[Verifier note — turn {ctx.turn}]\n{text}"
        self.notes.append(note)
        logger.info("verifier: turn %d steer (%d chars)", ctx.turn, len(text))
        return Intervention(inject_messages=[note])


def _last_turns(messages: list[Message], k: int) -> list[Message]:
    """Messages from the k-th most recent assistant message onward (never a system message)."""
    seen = 0
    start = 0
    for i in range(len(messages) - 1, -1, -1):
        if messages[i].get("role") == "assistant":
            seen += 1
            if seen >= k:
                start = i
                break
    return [m for m in messages[start:] if m.get("role") != "system"]
