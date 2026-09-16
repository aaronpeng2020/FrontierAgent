"""``web_fetch`` renders generic pages with moli before asking twice.

moli is driven as a subprocess, so a tiny shell script stands in for the
binary; twice is a ``httpx.MockTransport`` that records whether it was asked
at all. The live test at the bottom needs a real ``moli`` on PATH.
"""

from __future__ import annotations

import importlib
import json
import os
import shutil
import stat
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest

from plugins.tools._scrape_cache import scrape_result_cache

wf = importlib.import_module("plugins.tools.web_fetch")
_run = wf.web_fetch.func

_BASE = "https://twice.test"
_PAGE = "https://example.org/article"
_ARTICLE = "Rendered by moli. " * 60          # > _MOLI_MIN_CHARS
_TWICE_ARTICLE = "Rendered by twice. " * 60


def _fake_moli(tmp_path: Path, script: str) -> str:
    path = tmp_path / "moli"
    path.write_text("#!/bin/sh\n" + script + "\n")
    path.chmod(path.stat().st_mode | stat.S_IEXEC)
    return str(path)


@pytest.fixture(autouse=True)
def _isolate(monkeypatch: pytest.MonkeyPatch):
    scrape_result_cache.clear()
    wf._moli_resolved.clear()
    monkeypatch.setattr(wf, "_moli_slots", None)

    async def _public(_url: str) -> str:
        return ""
    monkeypatch.setattr(wf, "non_public_url_error", _public)

    async def _no_sleep(_seconds: float) -> None:
        return None
    monkeypatch.setattr(wf.asyncio, "sleep", _no_sleep)

    async def _unexpected(content: str, focus: str) -> str:
        raise AssertionError("summary LLM must not be called here")
    monkeypatch.setattr(wf, "_summary_llm_summarize", _unexpected)
    yield
    scrape_result_cache.clear()
    wf._moli_resolved.clear()


def _use(monkeypatch: pytest.MonkeyPatch, moli_bin: str, twice_key: str = "tw_key") -> list[httpx.Request]:
    """Point the tool at ``moli_bin`` and a recording fake twice."""
    monkeypatch.setattr(
        wf, "get_config",
        lambda: SimpleNamespace(twice_api_key=twice_key, twice_base_url=_BASE, moli_bin=moli_bin),
    )
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        page = {
            "url": _PAGE, "final_url": _PAGE, "status": 200, "title": "t",
            "challenge": "none", "engine": "cf", "chars": len(_TWICE_ARTICLE),
        }
        return httpx.Response(200, json={
            "run_id": "run_x", "status": "completed", "engine": "cf",
            "overall": "pass", "summary": "ok", "page": page,
            "content": _TWICE_ARTICLE,
            "range": {"start": 0, "end": len(_TWICE_ARTICLE), "total": len(_TWICE_ARTICLE)},
        })

    real_client = httpx.AsyncClient

    def factory(**kwargs: object) -> httpx.AsyncClient:
        kwargs["transport"] = httpx.MockTransport(handler)
        return real_client(**kwargs)
    monkeypatch.setattr(wf.httpx, "AsyncClient", factory)
    return seen


@pytest.mark.asyncio
async def test_moli_render_skips_twice(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    calls = tmp_path / "calls.json"
    moli = _fake_moli(tmp_path, f'echo "$@" > {calls}; printf "%s" "{_ARTICLE}"')
    seen = _use(monkeypatch, moli)

    result = await _run(_PAGE)

    assert result.strip() == _ARTICLE.strip()
    assert seen == [], "twice must not be asked when moli rendered the page"
    argv = calls.read_text().split()
    assert argv[:3] == ["fetch", "--dump", "markdown"]
    assert argv[-1] == _PAGE


@pytest.mark.asyncio
async def test_short_render_falls_through_to_twice(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    moli = _fake_moli(tmp_path, 'printf "# Access Denied\\n\\nReference #18.5"')
    seen = _use(monkeypatch, moli)

    result = await _run(_PAGE)

    assert result.strip() == _TWICE_ARTICLE.strip()
    assert len(seen) == 1


@pytest.mark.asyncio
async def test_nonzero_exit_falls_through_to_twice(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    # What moli prints for a PDF: "raw download output only supports ... --dump json".
    moli = _fake_moli(tmp_path, 'echo "Error: failed to fetch" >&2; exit 1')
    seen = _use(monkeypatch, moli)

    result = await _run(_PAGE)

    assert result.strip() == _TWICE_ARTICLE.strip()
    assert len(seen) == 1


@pytest.mark.asyncio
async def test_challenge_page_falls_through_to_twice(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    wall = "Please verify you are human. Checking your browser before accessing the site. " * 10
    assert len(wall) >= wf._MOLI_MIN_CHARS
    moli = _fake_moli(tmp_path, f'printf "%s" "{wall}"')
    seen = _use(monkeypatch, moli)

    result = await _run(_PAGE)

    assert result.strip() == _TWICE_ARTICLE.strip()
    assert len(seen) == 1


@pytest.mark.asyncio
async def test_hung_render_is_killed_then_twice(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(wf, "_MOLI_KILL_S", 0.3)
    moli = _fake_moli(tmp_path, 'sleep 30; printf "%s" "' + _ARTICLE + '"')
    seen = _use(monkeypatch, moli)

    result = await _run(_PAGE)

    assert result.strip() == _TWICE_ARTICLE.strip()
    assert len(seen) == 1


@pytest.mark.asyncio
async def test_missing_binary_goes_straight_to_twice(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    seen = _use(monkeypatch, str(tmp_path / "no-such-moli"))

    result = await _run(_PAGE)

    assert result.strip() == _TWICE_ARTICLE.strip()
    assert len(seen) == 1


@pytest.mark.asyncio
async def test_empty_moli_bin_disables_the_pass(monkeypatch: pytest.MonkeyPatch) -> None:
    seen = _use(monkeypatch, "")

    result = await _run(_PAGE)

    assert result.strip() == _TWICE_ARTICLE.strip()
    assert len(seen) == 1


@pytest.mark.asyncio
async def test_moli_failure_without_twice_key_names_the_variable(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    moli = _fake_moli(tmp_path, "exit 1")
    _use(monkeypatch, moli, twice_key="")

    result = await _run(_PAGE)

    assert "TWICE_API_KEY" in result


@pytest.mark.skipif(not shutil.which(os.environ.get("MOLI_BIN", "moli")), reason="moli not installed")
@pytest.mark.asyncio
async def test_live_moli_renders_a_local_page(monkeypatch: pytest.MonkeyPatch) -> None:
    """End-to-end with the real binary against a page served from this
    process: the text is written by JavaScript, so a plain HTTP fetch would
    return the empty shell. twice stays untouched."""
    import threading
    from http.server import BaseHTTPRequestHandler, HTTPServer

    body = b"""<!doctype html><title>Local page</title><main id="m"></main>
<script>
  const lines = [];
  for (let i = 0; i < 40; i++) lines.push("<p>Paragraph " + i + " rendered by page script.</p>");
  document.getElementById("m").innerHTML = "<h1>Local JS page</h1>" + lines.join("");
</script>"""

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *_args: object) -> None:
            return None

    server = HTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        seen = _use(monkeypatch, os.environ.get("MOLI_BIN", "moli"))
        result = await _run(f"http://127.0.0.1:{server.server_port}/page")
    finally:
        server.shutdown()

    assert "Local JS page" in result
    assert "Paragraph 39 rendered by page script." in result
    assert seen == []
