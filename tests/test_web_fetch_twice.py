"""``web_fetch`` reads generic pages through twice.sh.

Everything below drives the tool through ``httpx.MockTransport`` so the
request shapes (auth header, body, polling, content download) are what the
service actually sees. The single live test at the bottom runs only when
``TWICE_API_KEY`` is exported.
"""

from __future__ import annotations

import importlib
import json
import os
from collections.abc import Callable
from types import SimpleNamespace

import httpx
import pytest

from plugins.tools._scrape_cache import scrape_result_cache

# ``plugins.tools.web_fetch`` the attribute is the Tool object; the module
# has to be resolved by name.
wf = importlib.import_module("plugins.tools.web_fetch")
_run = wf.web_fetch.func  # the async function behind the Tool wrapper

_BASE = "https://twice.test"
_KEY = "tw_unit_test_key"
_PAGE = "https://example.org/article"
_LONG = "Paragraph with figures 12.5% and 300 units. " * 400  # > _SUMMARY_MIN_CHARS


def _twice_json(
    content: str,
    *,
    status: int = 200,
    challenge: str = "none",
    answer: str | None = None,
    url: str = _PAGE,
) -> dict:
    page = {
        "url": url, "final_url": url, "status": status,
        "title": "Example article", "challenge": challenge,
        "engine": "cf", "chars": len(content),
    }
    if answer is not None:
        page["answer"] = answer
    return {
        "run_id": "run_x", "status": "completed", "engine": "cf",
        "overall": "pass", "summary": "ok", "page": page, "content": content,
        "range": {"start": 0, "end": len(content), "total": len(content)},
    }


@pytest.fixture(autouse=True)
def _isolate(monkeypatch: pytest.MonkeyPatch):
    scrape_result_cache.clear()
    monkeypatch.setattr(
        wf, "get_config",
        lambda: SimpleNamespace(twice_api_key=_KEY, twice_base_url=_BASE),
    )
    # The SSRF vet resolves DNS; it has its own tests (test_network_tools) and
    # a sandbox resolver must not decide these. Everything below is public.
    async def _public(_url: str) -> str:
        return ""
    monkeypatch.setattr(wf, "non_public_url_error", _public)
    # Retry backoff must not slow the suite down.
    async def _no_sleep(_seconds: float) -> None:
        return None
    monkeypatch.setattr(wf.asyncio, "sleep", _no_sleep)
    # The dedicated summary endpoint must only run when the test says so.
    async def _unexpected(content: str, focus: str) -> str:
        raise AssertionError("summary LLM must not be called here")
    monkeypatch.setattr(wf, "_summary_llm_summarize", _unexpected)
    yield
    scrape_result_cache.clear()


def _install_transport(
    monkeypatch: pytest.MonkeyPatch, handler: Callable[[httpx.Request], httpx.Response],
) -> list[httpx.Request]:
    """Route every ``httpx.AsyncClient`` the module opens through ``handler``.

    Also asserts nothing but the twice origin is ever contacted: the page
    itself must never be fetched directly from this process.
    """
    seen: list[httpx.Request] = []

    def guarded(request: httpx.Request) -> httpx.Response:
        assert str(request.url).startswith(_BASE + "/"), request.url
        assert request.headers["authorization"] == f"Bearer {_KEY}"
        seen.append(request)
        return handler(request)

    real_client = httpx.AsyncClient

    def factory(**kwargs):
        kwargs["transport"] = httpx.MockTransport(guarded)
        return real_client(**kwargs)

    monkeypatch.setattr(wf.httpx, "AsyncClient", factory)
    return seen


async def test_generic_page_is_read_through_twice(monkeypatch: pytest.MonkeyPatch) -> None:
    seen = _install_transport(
        monkeypatch, lambda req: httpx.Response(200, json=_twice_json("# Hello\n\nBody text.")),
    )

    result = await _run(_PAGE)

    assert result == "# Hello\n\nBody text."
    assert len(seen) == 1
    req = seen[0]
    assert req.method == "POST" and req.url.path == "/v1/fetch"
    body = json.loads(req.content)
    assert body["url"] == _PAGE
    assert body["format"] == "markdown"
    assert "question" not in body


async def test_short_page_with_focus_is_returned_verbatim(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The "short number-dense pages are returned verbatim" rule survives:
    twice's answer is ignored when the page is under ``_SUMMARY_MIN_CHARS``."""
    content = "Revenue grew 12.5% to $300M in Q2."
    seen = _install_transport(
        monkeypatch,
        lambda req: httpx.Response(200, json=_twice_json(content, answer="12.5%")),
    )

    result = await _run(_PAGE, "revenue growth")

    assert result == content
    assert json.loads(seen[0].content)["question"] == "revenue growth"


async def test_long_page_with_focus_uses_twice_answer(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_transport(
        monkeypatch,
        lambda req: httpx.Response(
            200, json=_twice_json(_LONG, answer="Growth was 12.5% (300 units)."),
        ),
    )

    result = await _run(_PAGE, "growth figure")

    assert result == "Growth was 12.5% (300 units)."


async def test_long_page_without_answer_falls_back_to_summary_llm(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_transport(monkeypatch, lambda req: httpx.Response(200, json=_twice_json(_LONG)))
    calls: list[tuple[str, str]] = []

    async def fake_summary(content: str, focus: str) -> str:
        calls.append((content, focus))
        return "SUMMARY"

    monkeypatch.setattr(wf, "_summary_llm_summarize", fake_summary)

    result = await _run(_PAGE, "growth figure")

    assert result == "SUMMARY"
    assert calls == [(_LONG, "growth figure")]


async def test_202_is_polled_and_content_downloaded(monkeypatch: pytest.MonkeyPatch) -> None:
    polls = {"n": 0}

    def handler(req: httpx.Request) -> httpx.Response:
        if req.url.path == "/v1/fetch":
            return httpx.Response(202, json={"run_id": "run_slow", "status": "running"})
        if req.url.path == "/v1/runs/run_slow/wait":
            polls["n"] += 1
            if polls["n"] == 1:
                return httpx.Response(200, json={"run_id": "run_slow", "status": "running"})
            return httpx.Response(200, json={
                "run_id": "run_slow", "status": "completed", "error": None,
                "verdict": {"page": {
                    "url": _PAGE, "status": 200, "title": "PDF", "challenge": "passed",
                    "content_url": f"{_BASE}/a/run_slow/page.md",
                }},
            })
        if req.url.path == "/a/run_slow/page.md":
            return httpx.Response(200, text="PDF text from the runner.")
        raise AssertionError(req.url)

    seen = _install_transport(monkeypatch, handler)

    result = await _run(_PAGE)

    assert result == "PDF text from the runner."
    assert polls["n"] == 2
    assert seen[-1].url.path == "/a/run_slow/page.md"


async def test_blocked_challenge_reports_bot_protection(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_transport(
        monkeypatch,
        lambda req: httpx.Response(
            200, json=_twice_json("Just a moment...", status=403, challenge="blocked"),
        ),
    )

    result = await _run(_PAGE)

    assert result.startswith(f"Could not extract content from {_PAGE}")
    assert "blocks automated access" in result
    assert "curl" in result


async def test_origin_error_page_is_not_returned_as_content(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_transport(
        monkeypatch,
        lambda req: httpx.Response(200, json=_twice_json('{"error":"not found"}', status=404)),
    )

    result = await _run(_PAGE)

    assert result == f"Could not extract content from {_PAGE}"


async def test_rate_limit_is_retried(monkeypatch: pytest.MonkeyPatch) -> None:
    attempts = {"n": 0}

    def handler(req: httpx.Request) -> httpx.Response:
        attempts["n"] += 1
        if attempts["n"] == 1:
            return httpx.Response(429, json={"error": "slow down"})
        return httpx.Response(200, json=_twice_json("after retry"))

    _install_transport(monkeypatch, handler)

    assert await _run(_PAGE) == "after retry"
    assert attempts["n"] == 2


async def test_bad_api_key_is_not_retried(monkeypatch: pytest.MonkeyPatch) -> None:
    seen = _install_transport(
        monkeypatch, lambda req: httpx.Response(401, json={"error": "missing or malformed"}),
    )

    result = await _run(_PAGE)

    assert result == f"Could not extract content from {_PAGE}"
    assert len(seen) == 1


async def test_missing_key_names_the_variable(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        wf, "get_config", lambda: SimpleNamespace(twice_api_key="", twice_base_url=_BASE),
    )

    def no_transport(**kwargs: object) -> None:
        raise AssertionError("no request may leave without a key")

    monkeypatch.setattr(wf.httpx, "AsyncClient", no_transport)

    result = await _run(_PAGE)

    assert "TWICE_API_KEY" in result
    assert result.startswith("Could not extract content from")


async def test_academic_route_falls_back_to_twice(monkeypatch: pytest.MonkeyPatch) -> None:
    """PMC keeps its BioC-first routing; twice is only the leaf it lands on."""
    pmc_url = "https://pmc.ncbi.nlm.nih.gov/articles/PMC1234567/"

    async def no_fulltext(pmcid: str) -> str:
        assert pmcid == "PMC1234567"
        return ""

    monkeypatch.setattr(wf, "fetch_pmc_fulltext", no_fulltext)
    seen = _install_transport(
        monkeypatch,
        lambda req: httpx.Response(200, json=_twice_json("Rendered PMC page", url=pmc_url)),
    )

    result = await _run(pmc_url)

    assert result == "Rendered PMC page"
    assert json.loads(seen[0].content)["url"] == pmc_url


async def test_content_url_on_another_origin_never_gets_the_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def handler(req: httpx.Request) -> httpx.Response:
        if req.url.path == "/v1/fetch":
            return httpx.Response(202, json={"run_id": "run_x", "status": "running"})
        return httpx.Response(200, json={
            "status": "completed", "error": None,
            "verdict": {"page": {"status": 200, "content_url": "https://evil.test/page.md"}},
        })

    _install_transport(monkeypatch, handler)

    # The guard in ``_install_transport`` would fail on a request to evil.test.
    assert (await _run(_PAGE)).startswith("Could not extract content from")


@pytest.mark.skipif(not os.getenv("TWICE_API_KEY"), reason="TWICE_API_KEY not exported")
async def test_live_twice_reads_its_own_landing_page(monkeypatch: pytest.MonkeyPatch) -> None:
    """Opt-in end-to-end check against the real service."""
    from frontier_agent.infra.config import get_config

    config = get_config(force_reload=True)
    assert config.twice_api_key, "TWICE_API_KEY exported but not picked up by config"
    # Undo the fixture's fake endpoint: this one talks to the real service.
    monkeypatch.setattr(wf, "get_config", lambda: config)

    result = await _run("https://twice.sh/")

    assert "measure twice, cut once" in result.lower()
