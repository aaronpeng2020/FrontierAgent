"""Round-based compaction (``agent.context_compaction: rounds``).

The arm's contract: every ``k`` turns the history collapses to
``[system, workspace(task + report), tail]``, the report is the rewrite of the
previous report plus the round's work (never the previous workspace message fed
back as conversation), it lands on disk, and a failed rewrite degrades to the
previous report plus raw notes instead of losing the round.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from frontier_agent.core.messages import assistant_msg, system_msg, tool_msg, user_msg
from frontier_agent.core.runtime.loop.tiered_compact import InputTokenGauge
from workflows.stateful_react_agent.rounds import (
    WORKSPACE_HEADER,
    RoundsCompactor,
    RoundsPolicy,
)


class _LLM:
    def __init__(self, reply: str = "REPORT", fail: bool = False) -> None:
        self.reply, self.fail, self.prompts = reply, fail, []

    async def chat(self, messages):
        self.prompts.append(messages[-1]["content"])
        if self.fail:
            raise RuntimeError("endpoint down")
        return SimpleNamespace(content=f"{self.reply} v{len(self.prompts)}")


def _history(turns: int) -> list:
    msgs = [system_msg("sys"), user_msg("Task: how many moons does Mars have?")]
    for i in range(turns):
        msgs.append(assistant_msg("", tool_calls=[{"id": f"c{i}", "name": "web_search", "args": {"q": f"q{i}"}}]))
        msgs.append(tool_msg(f"result {i} " * 50, f"c{i}"))
    return msgs


# ── policy ───────────────────────────────────────────────────────────────────


def test_policy_fires_every_k_turns_and_restarts_after_mark() -> None:
    p = RoundsPolicy(4)
    fired = [t for t in range(1, 13) if p.should_compact(t, [], 0) and (p.mark_fired() or True)]
    assert fired == [4, 8, 12]


def test_policy_without_mark_keeps_firing() -> None:
    """The compactor owns the cadence: a forced pass that never calls mark_fired
    must not silently reset it, and a policy nobody marks stays hot."""
    p = RoundsPolicy(3)
    assert [p.should_compact(t, [], 0) for t in (1, 2, 3, 4)] == [False, False, True, True]


def test_policy_token_trigger_ends_a_round_early() -> None:
    gauge = InputTokenGauge()
    gauge.tokens, gauge.estimate = 150_000, 150_000
    p = RoundsPolicy(50, gauge=gauge, token_trigger=100_000)
    assert p.should_compact(2, [], 150_000)


def test_policy_rejects_nonpositive_k() -> None:
    with pytest.raises(ValueError):
        RoundsPolicy(0)


# ── compactor ────────────────────────────────────────────────────────────────


def test_round_rebuilds_history_and_writes_report(tmp_path) -> None:
    llm = _LLM()
    policy = RoundsPolicy(4)
    policy.should_compact(4, [], 0)
    c = RoundsCompactor(summary_llm=llm, task="how many moons does Mars have?", policy=policy,
                        report_path=tmp_path / "report.md", keep_recent_msgs=2)
    out = asyncio.run(c.compact(_history(4), keep_recent=999))

    assert out[0]["role"] == "system"
    assert out[1]["role"] == "user" and out[1]["content"].startswith(f"{WORKSPACE_HEADER} 1]")
    assert "REPORT v1" in out[1]["content"]
    # tail: the last assistant+tool pair, never an orphan tool result at the head
    assert [m["role"] for m in out[2:]] == ["assistant", "tool"]
    assert (tmp_path / "report.md").read_text() == "REPORT v1"
    assert c.last_event is not None and c.last_event.selected == "rounds"
    assert c.last_event.summary == "REPORT v1" and c.last_event.rollback_reason == ""
    assert c.last_event.tokens_after < c.last_event.tokens_before
    # the cadence restarted from the turn the policy was last asked about
    assert not policy.should_compact(7, [], 0) and policy.should_compact(8, [], 0)


def test_second_round_feeds_previous_report_not_previous_workspace() -> None:
    llm = _LLM()
    c = RoundsCompactor(summary_llm=llm, task="T", keep_recent_msgs=2)
    first = asyncio.run(c.compact(_history(3), keep_recent=0))
    # the agent works on: previous workspace + tail + 3 new turns
    nxt = list(first)
    for i in range(10, 13):
        nxt.append(assistant_msg("", tool_calls=[{"id": f"c{i}", "name": "web_fetch", "args": {"url": f"u{i}"}}]))
        nxt.append(tool_msg(f"page {i}", f"c{i}"))
    second = asyncio.run(c.compact(nxt, keep_recent=0))

    assert c.round == 2 and second[1]["content"].startswith(f"{WORKSPACE_HEADER} 2]")
    prompt = llm.prompts[-1]
    assert "Previous report (empty on the first round):\nREPORT v1" in prompt
    assert WORKSPACE_HEADER not in prompt          # workspace message filtered out of the round's work
    assert "page 10" in prompt and "page 11" in prompt


def test_failed_rewrite_keeps_previous_report_plus_raw_notes(tmp_path) -> None:
    llm = _LLM()
    c = RoundsCompactor(summary_llm=llm, task="T", report_path=tmp_path / "r.md", keep_recent_msgs=0)
    asyncio.run(c.compact(_history(2), keep_recent=0))
    llm.fail = True
    hist = [*_history(0)[:1], user_msg(f"{WORKSPACE_HEADER} 1]\n..."), *_history(3)[2:]]
    out = asyncio.run(c.compact(hist, keep_recent=0))

    assert c.report.startswith("REPORT v1")
    assert "Round 2 (rewrite failed" in c.report and "result 2" in c.report
    assert c.last_event is not None and c.last_event.rollback_reason == "llm_error"
    assert (tmp_path / "r.md").read_text() == c.report
    assert len(out) == 2  # system + workspace, nothing else survives a zero tail


def test_round_without_new_work_reissues_workspace_without_llm_call() -> None:
    llm = _LLM()
    c = RoundsCompactor(summary_llm=llm, task="T", keep_recent_msgs=0)
    asyncio.run(c.compact(_history(1), keep_recent=0))
    out = asyncio.run(c.compact([system_msg("sys"), user_msg(f"{WORKSPACE_HEADER} 1]\nold")], keep_recent=0))
    assert len(llm.prompts) == 1 and c.round == 2
    assert out[1]["content"].startswith(f"{WORKSPACE_HEADER} 2]") and "REPORT v1" in out[1]["content"]
