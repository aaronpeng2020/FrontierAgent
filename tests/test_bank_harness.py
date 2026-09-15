"""Evidence-bank harness (``agent.context_compaction: bank``).

Contract: ``save_evidence`` files quotes as E<n>.md + index.json; the bank
compactor rebuilds the history as ``[system, workspace(task + index + outline), tail]``
and writes the outline to disk; the report observer drafts one section per
outline heading from ONLY the evidence that section cites, polishes, carries the
agent's ```json block over verbatim, and fails open to the agent's own answer.
"""

from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace

from frontier_agent.core.loop_types import AgentLoopResult
from frontier_agent.core.messages import assistant_msg, system_msg, tool_msg, user_msg
from plugins.tools._evidence import EVIDENCE_DIR, cited_ids, load_index, save_entry
from plugins.tools.save_evidence import save_evidence
from workflows.stateful_react_agent.bank import BankCompactor, BankReportObserver, _split_sections
from workflows.stateful_react_agent.rounds import WORKSPACE_HEADER


class _LLM:
    def __init__(self, replies: list[str] | None = None, fail: bool = False) -> None:
        self.replies, self.fail, self.prompts = list(replies or []), fail, []

    async def chat(self, messages):
        self.prompts.append(messages[-1]["content"])
        if self.fail:
            raise RuntimeError("endpoint down")
        return SimpleNamespace(content=self.replies.pop(0) if self.replies else f"reply {len(self.prompts)}")


def _history(turns: int) -> list:
    msgs = [system_msg("sys"), user_msg("Task: how many moons does Mars have?")]
    for i in range(turns):
        msgs.append(assistant_msg("", tool_calls=[{"id": f"c{i}", "name": "web_fetch", "args": {"url": f"u{i}"}}]))
        msgs.append(tool_msg(f"page {i} " * 20, f"c{i}"))
    return msgs


def _bank(root, n: int) -> None:
    for i in range(1, n + 1):
        save_entry(url=f"https://ex.com/{i}", quote=f"quote {i}", note=f"fact {i}", date=f"2026-09-0{i}",
                   title=f"Page {i}", root=root)


# ── tool ─────────────────────────────────────────────────────────────────────


def test_save_evidence_files_quotes_and_dedupes(tmp_path) -> None:
    token = EVIDENCE_DIR.set(str(tmp_path / "evidence"))
    try:
        r1 = asyncio.run(save_evidence.ainvoke({"url": "https://a", "quote": "Mars has two moons.", "note": "count", "date": "2026-09-01"}))
        r2 = asyncio.run(save_evidence.ainvoke({"url": "https://b", "quote": "Phobos and Deimos.", "note": "names"}))
        r3 = asyncio.run(save_evidence.ainvoke({"url": "https://a", "quote": "Mars has two moons.", "note": "again"}))
        missing = asyncio.run(save_evidence.ainvoke({"url": "https://c", "quote": "", "note": "x"}))
    finally:
        EVIDENCE_DIR.reset(token)

    assert r1.startswith("E1 saved") and r2.startswith("E2 saved")
    assert r3.startswith("E1 already holds") and "quote is required" in missing
    root = tmp_path / "evidence"
    idx = json.loads((root / "index.json").read_text())
    assert [e["id"] for e in idx] == ["E1", "E2"] and idx[0]["date"] == "2026-09-01"
    assert "> Mars has two moons." in (root / "E1.md").read_text()
    assert cited_ids("see [E2] and E1, then [E2] again") == ["E2", "E1"]


# ── compactor ────────────────────────────────────────────────────────────────


def test_bank_round_shows_index_and_outline_and_writes_outline(tmp_path) -> None:
    root = tmp_path / "evidence"
    _bank(root, 2)
    llm = _LLM(["## Count\n- two moons [E1]\n\n## Open questions / next steps\n- none"])
    c = BankCompactor(evidence_root=root, summary_llm=llm, task="moons of Mars?", keep_recent_msgs=2)
    out = asyncio.run(c.compact(_history(3), keep_recent=999))

    ws = out[1]["content"]
    assert ws.startswith(f"{WORKSPACE_HEADER} 1]")
    assert "Evidence bank (2 entries" in ws and "E1 | 2026-09-01 | Page 1 — fact 1 | https://ex.com/1" in ws
    assert "## Count\n- two moons [E1]" in ws
    assert (root / "outline.md").read_text() == c.outline
    prompt = llm.prompts[0]
    assert "E2 | 2026-09-02" in prompt and "page 0" in prompt and "Previous outline (empty on the first round):\n(empty)" in prompt
    assert [m["role"] for m in out[2:]] == ["assistant", "tool"]


# ── report observer ──────────────────────────────────────────────────────────


def _result(final: str) -> AgentLoopResult:
    return AgentLoopResult(messages=[*_history(1), assistant_msg(final)], final_content=final,
                           metadata={"final_answer": final})


def test_report_is_written_per_section_from_cited_evidence_only(tmp_path) -> None:
    root = tmp_path / "evidence"
    _bank(root, 3)
    (root / "outline.md").write_text(
        "## Count\n- Mars has two moons [E1]\n\n## Names\n- Phobos, Deimos [E2] [E3]\n\n"
        "## Dead ends\n- nasa.gov timed out\n\n## Open questions / next steps\n- none\n"
    )
    llm = _LLM(["Two moons orbit Mars [E1].", "They are Phobos and Deimos [E2][E3].",
                "# Report\n\nsummary\n\n## Count\nTwo moons orbit Mars [E1].\n\n## Names\nThey are Phobos and Deimos [E2][E3]."])
    final = 'Answer: 2 moons.\n\n```json\n{"probability": 0.9, "confidence": "high"}\n```'
    res = _result(final)
    obs = BankReportObserver(llm=llm, task="moons of Mars?", evidence_root=root)
    asyncio.run(obs.on_loop_end(res))

    assert obs.sections_written == 2 and len(llm.prompts) == 3
    # section 1 saw E1 only; section 2 saw E2 and E3 only
    assert "quote 1" in llm.prompts[0] and "quote 2" not in llm.prompts[0]
    assert "quote 2" in llm.prompts[1] and "quote 3" in llm.prompts[1] and "quote 1" not in llm.prompts[1]
    assert "Section: Count" in llm.prompts[0]
    # polish saw the drafted sections and the agent's closing message
    assert "## Count\nTwo moons orbit Mars [E1]." in llm.prompts[2] and "Answer: 2 moons." in llm.prompts[2]
    report = res.final_content
    assert report.startswith("# Report") and res.metadata["final_answer"] == report
    assert res.metadata["final_answer_source"] == "bank_writer"
    assert '```json\n{"probability": 0.9, "confidence": "high"}\n```' in report   # carried over verbatim
    assert "## References" in report and "- [E1] Page 1 (2026-09-01) — https://ex.com/1" in report
    assert res.messages[-1]["role"] == "assistant"


def test_report_falls_back_to_agent_answer_when_bank_empty_or_llm_fails(tmp_path) -> None:
    root = tmp_path / "evidence"
    res = _result("agent answer")
    asyncio.run(BankReportObserver(llm=_LLM(), task="T", evidence_root=root).on_loop_end(res))
    assert res.final_content == "agent answer" and "final_answer_source" not in res.metadata

    _bank(root, 1)
    (root / "outline.md").write_text("## A\n- x [E1]\n")
    res = _result("agent answer")
    asyncio.run(BankReportObserver(llm=_LLM(fail=True), task="T", evidence_root=root).on_loop_end(res))
    assert res.final_content == "agent answer"


def test_outline_recovered_from_workspace_message_when_file_missing(tmp_path) -> None:
    root = tmp_path / "evidence"
    _bank(root, 1)
    ws = user_msg(f"{WORKSPACE_HEADER} 2]\nstuff\n\nOutline:\n## A\n- x [E1]\n\nContinue from ...")
    res = AgentLoopResult(messages=[system_msg("s"), ws, assistant_msg("done")], final_content="done",
                          metadata={"final_answer": "done"})
    llm = _LLM(["section a [E1]", "final report [E1]"])
    asyncio.run(BankReportObserver(llm=llm, task="T", evidence_root=root).on_loop_end(res))
    assert res.final_content.startswith("final report [E1]") and len(llm.prompts) == 2


def test_split_sections_drops_bookkeeping_headings() -> None:
    out = _split_sections("intro\n## A\n- a\n## To save\n- u\n## B\n- b\n## Dead ends\n- d\n## Open questions / next steps\n- q")
    assert out == [("A", "- a"), ("B", "- b")]


def test_load_index_ignores_garbage(tmp_path) -> None:
    (tmp_path / "index.json").write_text("not json")
    assert load_index(tmp_path) == []
