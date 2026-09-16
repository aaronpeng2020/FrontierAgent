"""Web scraping tool with academic URL routing.

Generic pages are rendered locally first by moli (https://github.com/lexmount/moli,
a lightweight headless browser: JavaScript apps render, anti-bot walls do not
pass). Whatever moli cannot read is sent to twice (https://twice.sh): the
service loads the URL in a real browser, passes anti-bot challenges where it
legitimately can, converts PDFs, and hands back markdown. PMC / PubMed /
bioRxiv / paywall URLs still go to the corresponding OA API first; the
moli→twice pair is the leaf those routes fall back to.
"""

from __future__ import annotations

import asyncio
import logging
import shutil
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlsplit

import httpx

from frontier_agent.core.tool import tool
from frontier_agent.infra.config import FrontierAgentConfig, get_config
from frontier_agent.infra.summary_llm import summarize as _summary_llm_summarize
from frontier_agent.infra.usage_meter import record_api_request
from plugins.tools._academic_fetch import (
    biorxiv_to_pdf,
    extract_pmcid,
    fetch_pmc_fulltext,
    fetch_unpaywall_oa_url,
    is_garbage_content,
    pubmed_to_pmc,
    resolve_doi,
    route_url,
)
from plugins.tools._bounded_fetch import (
    blocked_download_url,
    non_public_url_error,
    read_bounded,
)
from plugins.tools._scrape_cache import ScrapeUnavailable, scrape_result_cache

logger = logging.getLogger(__name__)

_MAX_RETRIES = 3
_SHORT_CONTENT_THRESHOLD = 500       # re-try via Unpaywall when content is this short
_PAYWALL_SHORT_THRESHOLD = 500       # tighter check used for known paywall domains
# A CAPTCHA / login-wall page is short; above this the keyword heuristic in
# ``is_garbage_content`` only triggers the Unpaywall detour, never a failure.
_GARBAGE_MAX_CHARS = 2_000

# Below this many chars the raw page is returned verbatim instead of being
# routed through an extraction LLM, even when ``info_to_extract`` is set. The
# extraction step exists purely to keep 50KB+ pages from blowing the agent's
# context; a short page (a policy paragraph, a financial snippet, a small
# table) costs nothing to carry whole, and paraphrasing it through a
# temperature=1.0 extractor risks drift.
#
# Threshold picked from an offline A/B on real pages: long policy/legal
# text (13-24K chars) summarised faithfully, but a short, number-dense
# page fabricated a derived percentage and back-filled figures that were
# never on the page. Drift tracks number density, not length, so the line
# sits high enough to route short number-dense pages to raw (a few K
# tokens whole — no real context cost) while long-form still gets
# compressed. Tune upward if headroom allows.
_SUMMARY_MIN_CHARS = 12_000

# ── moli tunables ─────────────────────────────────────────────────────────
# moli's own readiness deadline (page lifecycle + network idle), and the hard
# cap on the subprocess after which it is killed. A page moli cannot finish in
# this budget is exactly the kind twice's real browser exists for.
_MOLI_TIMEOUT_MS = 15_000
_MOLI_KILL_S = 25.0
# Below this many chars the render is treated as an app shell / error page
# and the URL goes on to twice. Matches ``_SHORT_CONTENT_THRESHOLD``.
_MOLI_MIN_CHARS = 500
# Peak RSS per render was 50-400 MB on real pages; the agent fans out
# ``web_fetch`` over URL lists, so the local renders are capped.
_MOLI_MAX_CONCURRENCY = 3
_moli_slots: asyncio.Semaphore | None = None
# Binary name → resolved path, or "" once a lookup failed.
_moli_resolved: dict[str, str] = {}

# ── twice tunables ────────────────────────────────────────────────────────
_TWICE_FETCH_PATH = "/v1/fetch"
# Server-side wait before ``POST /v1/fetch`` answers 202 with a run id
# instead of the page (service default 90, max 120).
_TWICE_WAIT_S = 90
# First slice of page text requested. Overflow trimming / spill handles
# anything the agent cannot carry; the service caps a slice at 500k.
_TWICE_MAX_CHARS = 200_000
# Long-poll length per ``GET /v1/runs/<id>/wait`` call.
_TWICE_POLL_S = 60
# Hard ceiling on one fetch, polling included: a challenge page or PDF takes
# 10-40 s, a cold runner adds 1-2 min.
_TWICE_DEADLINE_S = 360
# ``question`` is answered by a small model; keep the prompt bounded.
_TWICE_MAX_QUESTION_CHARS = 2_000
_TWICE_RUNNING_STATES = frozenset({"queued", "pending", "claimed", "running"})

# URLs twice reported as ``challenge: blocked`` during the current
# ``_fetch_one`` call. The academic routes try several URLs and only return
# text, so the leaf notes the block here for the caller-facing message.
_blocked_urls: ContextVar[list[str] | None] = ContextVar(
    "web_fetch_blocked_urls", default=None,
)


@dataclass(frozen=True)
class _TwicePage:
    """One page as twice returned it."""

    content: str
    title: str = ""
    answer: str = ""
    status: int = 0
    challenge: str = "none"

    @property
    def blocked(self) -> bool:
        return self.challenge == "blocked"


@tool
async def web_fetch(
    url: str | list[str],
    info_to_extract: str | list[str] = "",
) -> str:
    """Scrape and extract content from one or more web pages.

    Automatic backend selection based on URL domain: PMC / PubMed / bioRxiv /
    medRxiv URLs go to the corresponding OA API; known paywall domains are
    routed via Unpaywall; everything else is rendered by a headless browser
    (moli locally, then twice — a real browser — for JavaScript apps behind
    challenge pages and for PDFs). Retry and arXiv PDF→HTML redirect are
    applied automatically.

    A non-empty ``info_to_extract`` routes a long page through a cheap
    extraction LLM that returns only the information requested. With it
    omitted the raw extracted text is returned (subject to overflow trim).

    Args:
        url: A URL string, or a list of URLs for parallel fetch.
        info_to_extract: Optional focus for the extraction LLM. A single
            string applies to every URL; a list pairs with ``url`` 1:1.

    Returns:
        For a single URL, the extracted content directly. For a list, a
        numbered block per URL: ``[i] URL: …\\n    Info: …``.
    """
    urls, focuses = _normalise_inputs(url, info_to_extract)
    if not urls:
        return "Error: URL is required."

    if len(urls) == 1:
        return await _fetch_one(urls[0], focuses[0])

    results = await asyncio.gather(
        *(_fetch_one(u, f) for u, f in zip(urls, focuses, strict=False))
    )
    return "\n\n".join(
        f"[{i}] URL: {u}\n    Info: {r}"
        for i, (u, r) in enumerate(zip(urls, results, strict=False), 1)
    )


def _normalise_inputs(
    url: str | list[str], info_to_extract: str | list[str],
) -> tuple[list[str], list[str]]:
    """Coerce the LangChain payload into paired URL + focus lists."""
    from plugins.tools._coerce import coerce_json_list
    urls = coerce_json_list(url) if isinstance(url, str) else url
    if isinstance(urls, str):
        urls = [urls]
    elif not isinstance(urls, list):
        urls = []
    urls = [u.strip() for u in urls if isinstance(u, str) and u.strip()]

    focuses = (
        coerce_json_list(info_to_extract) if isinstance(info_to_extract, str)
        else info_to_extract
    )
    if isinstance(focuses, str):
        focuses = [focuses] * len(urls)
    elif isinstance(focuses, list):
        focuses = [str(f) for f in focuses]
        if len(focuses) < len(urls):
            focuses = focuses + [""] * (len(urls) - len(focuses))
    else:
        focuses = [""] * len(urls)
    return urls, focuses


async def _fetch_one(url: str, info_to_extract: str) -> str:
    """Run the full fetch pipeline for a single URL."""
    if not url or not url.strip():
        return "Error: URL is required."

    url = url.strip()
    if not url.startswith(("http://", "https://")):
        url = f"https://{url}"

    # Vet the target before ANYTHING leaves the process — including the
    # request to twice, which would otherwise be handed an internal URL.
    non_public = await non_public_url_error(url)
    if non_public:
        return (
            f"[BLOCKED] {non_public}. Only public http(s) endpoints may be "
            "fetched. Use a public source."
        )

    # Operator hard-block (WEB_DOMAIN_BLACKLIST_EXTRA) — same list that
    # filters web_search results, enforced here so a direct URL from page
    # content cannot bypass it.
    from plugins.tools.web_search import is_domain_blocked
    if is_domain_blocked(url):
        return (
            f"[BLOCKED] The domain of {url} is on the operator blocklist "
            f"and must not be accessed. Use a different source."
        )

    blocked_ext = blocked_download_url(url)
    if blocked_ext:
        return (
            f"[BLOCKED] This URL is a dataset/archive download ({blocked_ext}), "
            "not a web page. Do not download data files — read the dataset's "
            "documentation/landing page instead, or use aggregate API queries."
        )

    # arXiv PDFs consistently fail extraction — redirect to HTML abstract page.
    if "arxiv.org/pdf/" in url:
        html_url = url.replace("/pdf/", "/abs/").split(".pdf")[0]
        logger.info("Redirecting arxiv PDF → HTML: %s", html_url)
        url = html_url
    elif "arxiv.org/pdf" in url:
        html_url = url.replace("/pdf", "/abs")
        logger.info("Redirecting arxiv PDF → HTML: %s", html_url)
        url = html_url

    config = get_config()
    route = route_url(url)
    question = (info_to_extract or "").strip()

    # The leader's twice call may already carry the extraction answer. It is
    # held outside the cache on purpose: sharing raw page text between sibling
    # agents is fine, sharing one agent's focus is not (see ``_scrape_cache``).
    answer: dict[str, str] = {}
    blocked: list[str] = []
    token = _blocked_urls.set(blocked)

    # Single-flight cross-run cache: sibling agents fetching the same URL share
    # one round-trip. Only validated, non-garbage content is cached; the empty
    # / garbage branches raise so the failure is neither stored nor shared.
    async def _scrape() -> str:
        content, page_answer = await _fetch_via_route(
            url, route, config, question=question,
        )
        # Post-fetch quality check: an un-listed paywall can slip past the
        # route table. Try one Unpaywall-driven retry (cheap — fails fast when
        # no DOI resolves).
        recovered = await _maybe_recover_via_unpaywall(url, route, content, config)
        if not recovered:
            raise ScrapeUnavailable("blocked" if blocked else "empty")
        # Final garbage check — a login wall / "access denied" page can still
        # come back as a 200 with no challenge flag. twice reports real
        # challenges explicitly, so the keyword heuristic only decides for a
        # short body: a full article that merely *mentions* Cloudflare or
        # CAPTCHAs must not be thrown away.
        if len(recovered) < _GARBAGE_MAX_CHARS and is_garbage_content(recovered):
            raise ScrapeUnavailable("garbage")
        if page_answer and recovered is content:
            answer["text"] = page_answer
        return recovered

    try:
        content = await scrape_result_cache.get_or_scrape(url, _scrape)
    except ScrapeUnavailable as exc:
        if str(exc) == "blocked" or blocked:
            return (
                f"Could not extract content from {url}: the site blocks "
                "automated access (anti-bot challenge, HTTP 403) even for a "
                "real browser. Fetching it again — with this tool or with "
                "curl — will not help: look for the same material at another "
                "source (an archive copy, the publisher's API, a mirror), or "
                "search for the page title instead."
            )
        if str(exc) == "garbage":
            return (
                f"[ACCESS BLOCKED] The page at {url} is behind a paywall or "
                f"anti-bot protection. Please try searching for an open-access "
                f"version (arxiv.org, PMC, institutional repositories)."
            )
        if not config.twice_api_key:
            return (
                f"Could not extract content from {url}: TWICE_API_KEY is not "
                "set, so web_fetch cannot read web pages. Configure it "
                "(https://twice.sh) or answer from other sources."
            )
        return f"Could not extract content from {url}"
    finally:
        _blocked_urls.reset(token)

    # LLM extraction if requested. Skipped for short pages
    # (< _SUMMARY_MIN_CHARS): they carry whole at no context cost, and
    # returning them verbatim avoids paraphrase drift on the exact numbers /
    # scope qualifiers the agent asked for. The focus is still served — the
    # agent reads ``info_to_extract`` straight from the raw page.
    #
    # For a long page twice already answered ``question`` while it had the
    # page open; the dedicated SUMMARY_LLM_* endpoint only runs when there is
    # no such answer (academic API text, a cache hit, twice left it empty).
    if question and len(content) >= _SUMMARY_MIN_CHARS:
        if answer.get("text"):
            return answer["text"]
        return await _summary_llm_summarize(content, info_to_extract)

    from plugins.tools._overflow import maybe_overflow
    return maybe_overflow("web_fetch", content)


# ── Routing ───────────────────────────────────────────────────────────────

async def _fetch_via_route(
    url: str, route: str, config: FrontierAgentConfig, *, question: str = "",
) -> tuple[str, str]:
    """Dispatch to the domain-specific backend; always falls back to a browser.

    Returns ``(content, answer)``. ``answer`` is twice's reply to
    ``question`` and is only ever set on the generic route when twice did
    the render — the academic backends return text from an OA API that
    answered no question, and moli answers none either (the summary LLM
    covers a long moli page in ``_fetch_one``).
    """
    if route == "pmc":
        return await _fetch_pmc(url, config), ""
    if route == "pubmed":
        return await _fetch_pubmed(url, config), ""
    if route == "biorxiv":
        return await _fetch_biorxiv(url, config), ""
    if route == "paywall":
        return await _fetch_paywall(url, config), ""
    content = await _moli_text(url, config)
    if content:
        return content, ""
    return await _twice_text(url, config, question=question)


async def _fetch_pmc(url: str, config: FrontierAgentConfig) -> str:
    pmcid = extract_pmcid(url)
    if pmcid:
        text = await fetch_pmc_fulltext(pmcid)
        if text:
            return text
    return await _twice_or_empty(url, config)


async def _fetch_pubmed(url: str, config: FrontierAgentConfig) -> str:
    pmcid = await pubmed_to_pmc(url)
    if pmcid:
        text = await fetch_pmc_fulltext(pmcid)
        if text:
            return text
    return await _twice_or_empty(url, config)


async def _fetch_biorxiv(url: str, config: FrontierAgentConfig) -> str:
    pdf_url = biorxiv_to_pdf(url)
    text = ""
    if pdf_url:
        logger.info("[bioRxiv] Auto PDF: %s", pdf_url)
        text = await _twice_or_empty(pdf_url, config)
    if not text or len(text) < _SHORT_CONTENT_THRESHOLD:
        fallback = await _twice_or_empty(url, config)
        if fallback and len(fallback) > len(text):
            text = fallback
    return text


async def _fetch_paywall(url: str, config: FrontierAgentConfig) -> str:
    doi = await resolve_doi(url)
    text = ""
    if doi:
        oa_url = await fetch_unpaywall_oa_url(doi)
        if oa_url:
            logger.info("[Paywall bypass] %s → OA PDF: %s", url[:60], oa_url)
            text = await _twice_or_empty(oa_url, config)
    if not text or len(text) < _PAYWALL_SHORT_THRESHOLD:
        fallback = await _twice_or_empty(url, config)
        if fallback and len(fallback) > len(text):
            text = fallback
    return text


async def _twice_or_empty(url: str, config: FrontierAgentConfig) -> str:
    """Render ``url`` (moli, then twice); empty string on any failure."""
    content = await _moli_text(url, config)
    if content:
        return content
    content, _ = await _twice_text(url, config)
    return content


# ── moli (local render) ───────────────────────────────────────────────────

def _moli_binary(config: FrontierAgentConfig) -> str:
    """Absolute path of the moli binary, or "" when the pass is off/absent."""
    name = getattr(config, "moli_bin", "") or ""
    if not name:
        return ""
    if name not in _moli_resolved:
        path = shutil.which(name)
        if not path:
            logger.info("moli binary %r not found; web_fetch goes straight to twice", name)
        _moli_resolved[name] = path or ""
    return _moli_resolved[name] or ""


def _moli_usable(content: str) -> bool:
    """Does a moli render look like page content rather than a wall/shell?"""
    if len(content) < _MOLI_MIN_CHARS:
        return False
    return not (len(content) < _GARBAGE_MAX_CHARS and is_garbage_content(content))


async def _moli_text(url: str, config: FrontierAgentConfig) -> str:
    """Render ``url`` with the local moli browser; "" when twice should try.

    Anything short of a clean, non-trivial markdown render — binary missing,
    non-zero exit (PDFs, transport errors, HTTP errors), the deadline, an
    access-denied / challenge page, an empty app shell — yields "" so the
    caller falls through to twice. moli never raises into the route chains.
    """
    binary = _moli_binary(config)
    if not binary:
        return ""
    global _moli_slots
    if _moli_slots is None:
        _moli_slots = asyncio.Semaphore(_MOLI_MAX_CONCURRENCY)
    args = [
        binary, "fetch", "--dump", "markdown",
        "--timeout", str(_MOLI_TIMEOUT_MS), url,
    ]
    async with _moli_slots:
        try:
            proc = await asyncio.create_subprocess_exec(
                *args,
                stdin=asyncio.subprocess.DEVNULL,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
        except OSError as exc:
            record_api_request("moli", requests=0, errors=1)
            logger.warning("moli could not start for %s: %s", url[:60], exc)
            return ""
        try:
            out, err = await asyncio.wait_for(proc.communicate(), _MOLI_KILL_S)
        except asyncio.TimeoutError:
            proc.kill()
            await proc.wait()
            record_api_request("moli", requests=0, errors=1)
            logger.info("moli killed after %.0fs for %s", _MOLI_KILL_S, url[:60])
            return ""
    if proc.returncode != 0:
        record_api_request("moli", requests=0, errors=1)
        logger.info(
            "moli exit %s for %s: %s",
            proc.returncode, url[:60], err.decode("utf-8", "replace").strip()[-200:],
        )
        return ""
    content = out.decode("utf-8", "replace").strip()[:_TWICE_MAX_CHARS]
    if not _moli_usable(content):
        record_api_request("moli", errors=1)
        logger.info("moli render of %s unusable (%d chars); trying twice", url[:60], len(content))
        return ""
    record_api_request("moli")
    return content


async def _twice_text(
    url: str, config: FrontierAgentConfig, *, question: str = "",
) -> tuple[str, str]:
    """``(content, answer)`` from twice; ``("", "")`` on any failure.

    A page twice could not get past (``challenge: blocked``) is noted in the
    per-fetch context so the caller can say so instead of "could not
    extract"; it never raises into the academic fallback chains.
    """
    if not config.twice_api_key:
        return "", ""
    try:
        page = await _twice_fetch(url, config, question=question)
    except Exception as exc:
        logger.warning("twice fetch failed for %s: %s", url[:60], exc)
        return "", ""
    if page is None:
        return "", ""
    if page.blocked:
        seen = _blocked_urls.get()
        if seen is not None:
            seen.append(url)
        return "", ""
    return page.content, page.answer


async def _maybe_recover_via_unpaywall(
    url: str, route: str, content: str, config: FrontierAgentConfig,
) -> str:
    """If the routed fetch returned garbage or was suspiciously short on a
    paywall domain, attempt one Unpaywall-driven retry.

    Skipped for ``pmc``/``pubmed`` — if the BioC API couldn't resolve the
    article there's no DOI detour worth trying that twice wouldn't already hit.
    """
    if route in ("pmc", "pubmed"):
        return content

    garbage = is_garbage_content(content)
    short_on_paywall = (
        route == "paywall"
        and content
        and len(content) < _PAYWALL_SHORT_THRESHOLD
    )
    if not garbage and not short_on_paywall:
        return content

    reason = "empty" if not content else ("garbage" if garbage else "short")
    logger.warning(
        "[Quality] %s content detected for %s — trying Unpaywall fallback",
        reason, url[:60],
    )
    doi = await resolve_doi(url)
    if not doi:
        return content
    oa_url = await fetch_unpaywall_oa_url(doi)
    if not oa_url:
        return content

    alt = await _twice_or_empty(oa_url, config)
    if alt and not is_garbage_content(alt) and len(alt) > len(content):
        logger.info("[Quality] Unpaywall fallback succeeded: %d chars", len(alt))
        return alt
    return content


# ── twice client ──────────────────────────────────────────────────────────

def _twice_headers(config: FrontierAgentConfig) -> dict[str, str]:
    return {
        "Authorization": f"Bearer {config.twice_api_key}",
        "Content-Type": "application/json",
        "Accept": "application/json",
    }


async def _twice_fetch(
    url: str, config: FrontierAgentConfig, *, question: str = "",
) -> _TwicePage | None:
    """One page through ``POST /v1/fetch``; ``None`` when it could not be read.

    A 202 (challenge page / PDF still rendering after ``wait``) is followed
    through ``GET /v1/runs/<id>/wait`` until the run is terminal, then the
    text is downloaded from ``verdict.page.content_url``. 429/5xx/transport
    errors retry with backoff; anything else from the service (bad key, plan
    limit) is a configuration problem and aborts without retry.
    """
    base = (config.twice_base_url or "https://twice.sh").rstrip("/")
    headers = _twice_headers(config)
    body: dict[str, Any] = {
        "url": url,
        "format": "markdown",
        "wait": _TWICE_WAIT_S,
        "max_chars": _TWICE_MAX_CHARS,
    }
    if question:
        body["question"] = question[:_TWICE_MAX_QUESTION_CHARS]

    loop = asyncio.get_running_loop()
    deadline = loop.time() + _TWICE_DEADLINE_S
    timeout = httpx.Timeout(_TWICE_WAIT_S + 30, connect=20)
    async with httpx.AsyncClient(timeout=timeout) as client:
        data = await _twice_post(client, base, headers, body, url)
        if data is None:
            return None
        if data.get("status") in _TWICE_RUNNING_STATES:
            run_id = str(data.get("run_id") or "")
            if not run_id:
                logger.error("twice answered 202 without a run_id for %s", url[:60])
                return None
            data = await _twice_wait(client, base, headers, run_id, url, deadline)
            if data is None:
                return None
    return _twice_page_from(data, url)


async def _twice_post(
    client: httpx.AsyncClient,
    base: str,
    headers: dict[str, str],
    body: dict[str, Any],
    url: str,
) -> dict[str, Any] | None:
    for attempt in range(_MAX_RETRIES):
        try:
            resp = await client.post(
                f"{base}{_TWICE_FETCH_PATH}", headers=headers, json=body,
            )
        except httpx.TimeoutException:
            record_api_request("twice", requests=0, errors=1)
            logger.warning(
                "twice timeout for %s (attempt %d)", url[:60], attempt + 1,
            )
            resp = None
        except httpx.HTTPError as exc:
            record_api_request("twice", requests=0, errors=1)
            logger.warning(
                "twice transport error for %s: %s (attempt %d)",
                url[:60], exc, attempt + 1,
            )
            resp = None

        if resp is not None:
            if resp.status_code in (200, 202):
                record_api_request("twice")
                try:
                    data = resp.json()
                except ValueError:
                    logger.error("twice returned non-JSON for %s", url[:60])
                    return None
                return data if isinstance(data, dict) else None
            record_api_request("twice", errors=1)
            if resp.status_code != 429 and resp.status_code < 500:
                # Bad key, plan limit, malformed request: retrying cannot help.
                logger.error(
                    "twice HTTP %d for %s (body: %s)",
                    resp.status_code, url[:60], resp.text[:200],
                )
                return None
            logger.warning(
                "twice HTTP %d for %s (attempt %d)",
                resp.status_code, url[:60], attempt + 1,
            )

        if attempt < _MAX_RETRIES - 1:
            await asyncio.sleep(2 ** attempt)
    logger.error("twice fetch exhausted retries for %s", url[:60])
    return None


async def _twice_wait(
    client: httpx.AsyncClient,
    base: str,
    headers: dict[str, str],
    run_id: str,
    url: str,
    deadline: float,
) -> dict[str, Any] | None:
    """Long-poll a 202 run to a terminal state; return a ``/v1/fetch``-shaped dict."""
    loop = asyncio.get_running_loop()
    transport_failures = 0
    while True:
        remaining = deadline - loop.time()
        if remaining <= 0:
            logger.warning(
                "twice run %s for %s still running after %ds — giving up",
                run_id, url[:60], _TWICE_DEADLINE_S,
            )
            return None
        poll = int(max(1, min(_TWICE_POLL_S, remaining)))
        try:
            resp = await client.get(
                f"{base}/v1/runs/{run_id}/wait",
                params={"timeout": poll},
                headers=headers,
                timeout=httpx.Timeout(poll + 30, connect=20),
            )
        except httpx.HTTPError as exc:
            transport_failures += 1
            record_api_request("twice", requests=0, errors=1)
            logger.warning("twice poll error for %s: %s", url[:60], exc)
            if transport_failures >= _MAX_RETRIES:
                return None
            await asyncio.sleep(2 ** transport_failures)
            continue
        if resp.status_code != 200:
            record_api_request("twice", errors=1)
            logger.error(
                "twice poll HTTP %d for run %s (%s)", resp.status_code, run_id, url[:60],
            )
            return None
        try:
            data = resp.json()
        except ValueError:
            logger.error("twice poll returned non-JSON for run %s", run_id)
            return None
        if not isinstance(data, dict):
            return None
        status = str(data.get("status") or "")
        if status in _TWICE_RUNNING_STATES:
            continue
        if status != "completed" or data.get("error"):
            logger.warning(
                "twice run %s for %s ended %s: %s",
                run_id, url[:60], status or "unknown", data.get("error"),
            )
            return None
        verdict = data.get("verdict") or {}
        page = verdict.get("page") if isinstance(verdict, dict) else None
        if not isinstance(page, dict):
            page = {}
        content = ""
        content_url = page.get("content_url")
        if isinstance(content_url, str) and content_url:
            content = await _twice_download(client, base, headers, content_url, url)
        return {"status": "completed", "page": page, "content": content}


async def _twice_download(
    client: httpx.AsyncClient,
    base: str,
    headers: dict[str, str],
    content_url: str,
    url: str,
) -> str:
    """Fetch the finished run's text. Only the service's own origin gets our key."""
    if urlsplit(content_url)[:2] != urlsplit(base)[:2]:
        logger.error(
            "twice content_url %s is not on %s — refusing to send credentials",
            content_url[:80], base,
        )
        return ""
    try:
        async with client.stream(
            "GET", content_url, headers={"Authorization": headers["Authorization"]},
        ) as resp:
            if resp.status_code != 200:
                record_api_request("twice", errors=1)
                logger.error(
                    "twice content download HTTP %d for %s", resp.status_code, url[:60],
                )
                return ""
            body, _ = await read_bounded(resp)
    except httpx.HTTPError as exc:
        record_api_request("twice", requests=0, errors=1)
        logger.warning("twice content download failed for %s: %s", url[:60], exc)
        return ""
    # The service writes UTF-8 markdown; skip charset sniffing (chardet
    # mis-classifies CJK-heavy bodies with ASCII headers as Windows-1252).
    return body.decode("utf-8", errors="replace")


def _twice_page_from(data: dict[str, Any], url: str) -> _TwicePage | None:
    page = data.get("page")
    if not isinstance(page, dict):
        page = {}
    content = data.get("content")
    if not isinstance(content, str):
        content = ""
    try:
        status = int(page.get("status") or 0)
    except (TypeError, ValueError):
        status = 0
    challenge = str(page.get("challenge") or "none")
    title = str(page.get("title") or "")
    answer = page.get("answer")
    answer = answer.strip() if isinstance(answer, str) else ""

    if challenge == "blocked":
        logger.warning(
            "twice: %s blocks automated access (HTTP %d, challenge blocked)",
            url[:60], status,
        )
        return _TwicePage(content="", title=title, status=status, challenge=challenge)
    if status >= 400:
        # The origin answered with an error page; its body is not the page.
        logger.warning("twice: HTTP %d from origin for %s", status, url[:60])
        return None
    return _TwicePage(
        content=content, title=title, answer=answer, status=status, challenge=challenge,
    )
