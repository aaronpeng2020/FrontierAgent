"""Web scraping tool with academic URL routing.

Generic pages are read through the searchs.io gateway (``GET /v1/extract``,
repo ~/code/searchs.io): a plain fetch first, then twice.sh — a real browser
that passes anti-bot challenges where it legitimately can and converts PDFs —
then Jina Reader, one markdown document either way. PMC / PubMed / bioRxiv /
paywall URLs still go to the corresponding OA API first; the gateway is the
leaf those routes fall back to. (The ``_twice_*`` names below predate the
gateway hop; twice is still the engine that does the heavy lifting.)
"""

from __future__ import annotations

import asyncio
import logging
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Any

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

# ── searchs.io extract tunables ───────────────────────────────────────────
_EXTRACT_PATH = "/v1/extract"
_SEARCHS_DEFAULT_BASE = "https://api.searchs.io"
# How long the gateway may keep twice on a page before answering 502/504
# (gateway cap 120). A challenge page or PDF takes 10-40 s; a cold runner
# adds 1-2 min, which then surfaces as a failed fetch the agent retries later.
_TWICE_WAIT_S = 90
# Page text requested. Overflow trimming / spill handles anything the agent
# cannot carry; the gateway caps a document at 200k.
_TWICE_MAX_CHARS = 200_000
# ``question`` is answered by a small model while twice has the page open;
# the gateway forwards up to 2000 chars.
_TWICE_MAX_QUESTION_CHARS = 2_000

# URLs twice reported as ``challenge: blocked`` during the current
# ``_fetch_one`` call. The academic routes try several URLs and only return
# text, so the leaf notes the block here for the caller-facing message.
_blocked_urls: ContextVar[list[str] | None] = ContextVar(
    "web_fetch_blocked_urls", default=None,
)


@dataclass(frozen=True)
class _TwicePage:
    """One page as the gateway (twice underneath) returned it."""

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
    routed via Unpaywall; everything else goes through the searchs.io
    gateway (plain fetch, then twice — a real browser — for JavaScript apps,
    challenge pages and PDFs). Retry and arXiv PDF→HTML redirect are applied
    automatically.

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
        if not config.searchs_api_key:
            return (
                f"Could not extract content from {url}: SEARCHS_API_KEY is not "
                "set, so web_fetch cannot read web pages. Configure it "
                "(https://searchs.io) or answer from other sources."
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
    """Dispatch to the domain-specific backend; always falls back to twice.

    Returns ``(content, answer)``. ``answer`` is twice's reply to
    ``question`` and is only ever set on the generic route — the academic
    backends return text from an OA API that answered no question.
    """
    if route == "pmc":
        return await _fetch_pmc(url, config), ""
    if route == "pubmed":
        return await _fetch_pubmed(url, config), ""
    if route == "biorxiv":
        return await _fetch_biorxiv(url, config), ""
    if route == "paywall":
        return await _fetch_paywall(url, config), ""
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
    """Read ``url`` through twice; empty string on any failure."""
    content, _ = await _twice_text(url, config)
    return content


async def _twice_text(
    url: str, config: FrontierAgentConfig, *, question: str = "",
) -> tuple[str, str]:
    """``(content, answer)`` from twice; ``("", "")`` on any failure.

    A page twice could not get past (``challenge: blocked``) is noted in the
    per-fetch context so the caller can say so instead of "could not
    extract"; it never raises into the academic fallback chains.
    """
    if not config.searchs_api_key:
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


# ── searchs.io extract client ─────────────────────────────────────────────
# ``GET /v1/extract?engine=auto`` on the gateway: a plain fetch first (free,
# sub-second), twice.sh's real browser when the page is bot-walled / JS-only
# / too thin / a PDF, then Jina Reader — one JSON document either way, so
# there is no 202 / run polling on this side any more. ``question`` forces
# twice (the only engine that answers it) and comes back as ``answer``.

def _twice_headers(config: FrontierAgentConfig) -> dict[str, str]:
    return {
        "Authorization": f"Bearer {config.searchs_api_key}",
        "Accept": "application/json",
    }


async def _twice_fetch(
    url: str, config: FrontierAgentConfig, *, question: str = "",
) -> _TwicePage | None:
    """One page through the gateway; ``None`` when it could not be read.

    429 / 5xx / transport errors retry with backoff, except the gateway's own
    ``fetch_failed`` (every engine already gave up on the page — a retry
    would only re-run the whole chain). Anything else from the service (bad
    key, plan limit) is a configuration problem and aborts without retry.
    """
    base = (config.searchs_base_url or _SEARCHS_DEFAULT_BASE).rstrip("/")
    headers = _twice_headers(config)
    params: dict[str, Any] = {
        "url": url,
        "engine": "auto",
        "wait": _TWICE_WAIT_S,
        "max_chars": _TWICE_MAX_CHARS,
    }
    if question:
        params["question"] = question[:_TWICE_MAX_QUESTION_CHARS]

    # The gateway aborts twice at wait+15 s; leave room for its own hop.
    timeout = httpx.Timeout(_TWICE_WAIT_S + 45, connect=20)
    async with httpx.AsyncClient(timeout=timeout) as client:
        data = await _twice_get(client, base, headers, params, url)
    if data is None:
        return None
    return _twice_page_from(data, url)


def _gateway_error(resp: httpx.Response) -> tuple[str, str]:
    """``(error code, message)`` from a gateway error envelope; "" when absent."""
    try:
        data = resp.json()
    except ValueError:
        return "", ""
    if not isinstance(data, dict):
        return "", ""
    return str(data.get("error") or ""), str(data.get("message") or "")


async def _twice_get(
    client: httpx.AsyncClient,
    base: str,
    headers: dict[str, str],
    params: dict[str, Any],
    url: str,
) -> dict[str, Any] | None:
    for attempt in range(_MAX_RETRIES):
        try:
            resp = await client.get(f"{base}{_EXTRACT_PATH}", headers=headers, params=params)
        except httpx.TimeoutException:
            record_api_request("searchs", requests=0, errors=1)
            logger.warning(
                "searchs.io extract timeout for %s (attempt %d)", url[:60], attempt + 1,
            )
            resp = None
        except httpx.HTTPError as exc:
            record_api_request("searchs", requests=0, errors=1)
            logger.warning(
                "searchs.io extract transport error for %s: %s (attempt %d)",
                url[:60], exc, attempt + 1,
            )
            resp = None

        if resp is not None:
            if resp.status_code == 200:
                record_api_request("searchs")
                try:
                    data = resp.json()
                except ValueError:
                    logger.error("searchs.io extract returned non-JSON for %s", url[:60])
                    return None
                return data if isinstance(data, dict) else None
            record_api_request("searchs", errors=1)
            code, message = _gateway_error(resp)
            if code == "fetch_failed":
                # Every engine (fetch → twice → jina) gave up on this page.
                logger.warning(
                    "searchs.io could not read %s: %s", url[:60], message[:200],
                )
                return None
            if resp.status_code != 429 and resp.status_code < 500:
                # Bad key, plan limit, malformed request: retrying cannot help.
                logger.error(
                    "searchs.io extract HTTP %d for %s (body: %s)",
                    resp.status_code, url[:60], resp.text[:200],
                )
                return None
            logger.warning(
                "searchs.io extract HTTP %d for %s (attempt %d)",
                resp.status_code, url[:60], attempt + 1,
            )

        if attempt < _MAX_RETRIES - 1:
            await asyncio.sleep(2 ** attempt)
    logger.error("searchs.io extract exhausted retries for %s", url[:60])
    return None


def _twice_page_from(data: dict[str, Any], url: str) -> _TwicePage | None:
    """Shape the gateway's ExtractResult into a ``_TwicePage``."""
    content = data.get("markdown")
    if not isinstance(content, str) or not content:
        content = data.get("content")
    if not isinstance(content, str):
        content = ""
    try:
        status = int(data.get("status") or 0)
    except (TypeError, ValueError):
        status = 0
    challenge = str(data.get("challenge") or "none")
    title = str(data.get("title") or "")
    answer = data.get("answer")
    answer = answer.strip() if isinstance(answer, str) else ""

    if challenge == "blocked":
        logger.warning(
            "searchs.io/twice: %s blocks automated access (HTTP %d, challenge blocked)",
            url[:60], status,
        )
        return _TwicePage(content="", title=title, status=status, challenge=challenge)
    if status >= 400:
        # The origin answered with an error page; its body is not the page.
        logger.warning("searchs.io: HTTP %d from origin for %s", status, url[:60])
        return None
    return _TwicePage(
        content=content, title=title, answer=answer, status=status, challenge=challenge,
    )
