"""Local verifier (``agent.local_verifier``): a side audit every k turns that may inject a steer."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

from frontier_agent.core.loop_types import TurnContext
from frontier_agent.core.messages import assistant_msg, system_msg, tool_msg, user_msg
from workflows.stateful_react_agent.verify import LocalVerifierObserver, _last_turns


class _LLM:
    def __init__(self, replies: list[str], fail: bool = False) -> None:
        self.replies, self.fail, self.prompts = list(replies), fail, []

    async def chat(self, messages):
        self.prompts.append(messages[-1]["content"])
        if self.fail:
            raise RuntimeError("endpoint down")
        return SimpleNamespace(content=self.replies.pop(0) if self.replies else "OK")


def _history(turns: int, *, tools: bool = True) -> list:
    msgs = [system_msg("sys"), user_msg("Task: will X happen by 2026-12-31?")]
    for i in range(turns):
        if tools:
            msgs.append(assistant_msg("", tool_calls=[{"id": f"c{i}", "name": "web_search", "args": {"q": f"q{i}"}}]))
            msgs.append(tool_msg(f"result {i}", f"c{i}"))
        else:
            msgs.append(assistant_msg(f"thinking {i}"))
    return msgs


def _ctx(turn: int, messages: list) -> TurnContext:
    return TurnContext(turn=turn, max_turns=100, task_id="t", role_id="r", ai_text="", thinking="",
                       tool_calls=[], messages=messages, usage=None, metadata={})


def _run(obs: LocalVerifierObserver, turns: range, msgs) -> dict[int, object]:
    return {t: asyncio.run(obs.on_turn_end(_ctx(t, msgs))) for t in turns}


def test_audits_every_k_turns_and_injects_only_a_steer() -> None:
    llm = _LLM(["OK", "Stop repeating the same query; open the Reuters page you found at turn 6."])
    obs = LocalVerifierObserver(llm=llm, task="T", k=4)
    out = _run(obs, range(1, 9), _history(8))

    assert [t for t, r in out.items() if llm.prompts and t in (4, 8)] == [4, 8]
    assert len(llm.prompts) == 2                       # one audit per round, none in between
    assert out[4] is None                              # "OK" injects nothing
    assert out[8] is not None and out[8].inject_messages
    note = out[8].inject_messages[0]
    assert note.startswith("[Verifier note — turn 8]") and "Reuters" in note
    assert obs.notes == [note]


def test_round_without_tool_results_is_not_audited() -> None:
    llm = _LLM(["should not be asked"])
    obs = LocalVerifierObserver(llm=llm, task="T", k=3)
    out = _run(obs, range(1, 4), _history(3, tools=False))
    assert llm.prompts == [] and all(r is None for r in out.values())


def test_audit_failure_is_silent() -> None:
    llm = _LLM([], fail=True)
    obs = LocalVerifierObserver(llm=llm, task="T", k=2)
    assert asyncio.run(obs.on_turn_end(_ctx(2, _history(2)))) is None
    # the cadence still advanced: the next audit is at turn 4, not 3
    assert asyncio.run(obs.on_turn_end(_ctx(3, _history(3)))) is None and len(llm.prompts) == 1


def test_prompt_carries_task_and_only_the_window() -> None:
    llm = _LLM(["OK"])
    obs = LocalVerifierObserver(llm=llm, task="will X happen?", k=2)
    asyncio.run(obs.on_turn_end(_ctx(6, _history(6))))
    prompt = llm.prompts[0]
    assert "will X happen?" in prompt and "Today is turn 6" in prompt
    assert "result 5" in prompt and "result 4" in prompt and "result 3" not in prompt


def test_last_turns_never_includes_system() -> None:
    msgs = _history(3)
    window = _last_turns(msgs, 2)
    assert all(m["role"] != "system" for m in window)
    assert [m["role"] for m in window] == ["assistant", "tool", "assistant", "tool"]
