"""``web_fetch`` reads generic pages through the searchs.io gateway
(``GET /v1/extract``; twice.sh is the engine underneath).

Everything below drives the tool through ``httpx.MockTransport`` so the
request shapes (auth header, query params, error envelopes) are what the
gateway actually sees. The single live test at the bottom runs only when
``SEARCHS_API_KEY`` is exported.
"""

from __future__ import annotations

import importlib
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

_BASE = "https://searchs.test"
_KEY = "sk_live_unit_test_key"
_PAGE = "https://example.org/article"
_LONG = "Paragraph with figures 12.5% and 300 units. " * 400  # > _SUMMARY_MIN_CHARS


def _twice_json(
    content: str,
    *,
    status: int = 200,
    challenge: str | None = None,
    answer: str | None = None,
    url: str = _PAGE,
    engine: str = "twice",
) -> dict:
    """One ExtractResult as ``GET /v1/extract`` returns it."""
    data = {
        "url": url, "final_url": url, "status": status,
        "title": "Example article", "engine": engine,
        "markdown": content, "content": content, "truncated": False,
    }
    if challenge is not None:
        data["challenge"] = challenge
    if answer is not None:
        data["answer"] = answer
    return data


@pytest.fixture(autouse=True)
def _isolate(monkeypatch: pytest.MonkeyPatch):
    scrape_result_cache.clear()
    monkeypatch.setattr(
        wf, "get_config",
        lambda: SimpleNamespace(searchs_api_key=_KEY, searchs_base_url=_BASE),
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

    Also asserts nothing but the gateway origin is ever contacted: the page
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
    assert req.method == "GET" and req.url.path == "/v1/extract"
    assert req.url.params["url"] == _PAGE
    assert req.url.params["engine"] == "auto"
    assert req.url.params["wait"] == str(wf._TWICE_WAIT_S)
    assert req.url.params["max_chars"] == str(wf._TWICE_MAX_CHARS)
    assert "question" not in req.url.params


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
    assert seen[0].url.params["question"] == "revenue growth"


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


async def test_gateway_fetch_failed_is_not_retried(monkeypatch: pytest.MonkeyPatch) -> None:
    """502 ``fetch_failed`` means every engine already gave up on the page."""
    seen = _install_transport(
        monkeypatch,
        lambda req: httpx.Response(
            502, json={"error": "fetch_failed", "message": "twice: HTTP 503 from origin"},
        ),
    )

    result = await _run(_PAGE)

    assert result == f"Could not extract content from {_PAGE}"
    assert len(seen) == 1


async def test_gateway_5xx_without_envelope_is_retried(monkeypatch: pytest.MonkeyPatch) -> None:
    attempts = {"n": 0}

    def handler(req: httpx.Request) -> httpx.Response:
        attempts["n"] += 1
        if attempts["n"] == 1:
            return httpx.Response(503, text="upstream unavailable")
        return httpx.Response(200, json=_twice_json("after 503"))

    _install_transport(monkeypatch, handler)

    assert await _run(_PAGE) == "after 503"
    assert attempts["n"] == 2


async def test_plain_fetch_result_without_challenge_field(monkeypatch: pytest.MonkeyPatch) -> None:
    """engine=fetch / jina results carry no ``challenge``; they read like any page."""
    _install_transport(
        monkeypatch,
        lambda req: httpx.Response(200, json=_twice_json("Static page.", engine="fetch")),
    )

    assert await _run(_PAGE) == "Static page."


async def test_missing_key_names_the_variable(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        wf, "get_config", lambda: SimpleNamespace(searchs_api_key="", searchs_base_url=_BASE),
    )

    def no_transport(**kwargs: object) -> None:
        raise AssertionError("no request may leave without a key")

    monkeypatch.setattr(wf.httpx, "AsyncClient", no_transport)

    result = await _run(_PAGE)

    assert "SEARCHS_API_KEY" in result
    assert result.startswith("Could not extract content from")


async def test_academic_route_falls_back_to_twice(monkeypatch: pytest.MonkeyPatch) -> None:
    """PMC keeps its BioC-first routing; the gateway is only the leaf it lands on."""
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
    assert seen[0].url.params["url"] == pmc_url


@pytest.mark.skipif(not os.getenv("SEARCHS_API_KEY"), reason="SEARCHS_API_KEY not exported")
async def test_live_gateway_reads_a_public_page(monkeypatch: pytest.MonkeyPatch) -> None:
    """Opt-in end-to-end check against the real gateway."""
    from frontier_agent.infra.config import get_config

    config = get_config(force_reload=True)
    assert config.searchs_api_key, "SEARCHS_API_KEY exported but not picked up by config"
    # Undo the fixture's fake endpoint: this one talks to the real service.
    monkeypatch.setattr(wf, "get_config", lambda: config)

    result = await _run("https://example.com/")

    assert "example domain" in result.lower()
