"""In-process positive cache for web scrape content (single-flight)."""

from __future__ import annotations

import asyncio
import os
from collections.abc import Awaitable, Callable

from frontier_agent.infra.usage_meter import record_api_request

# ── Positive scrape cache (cross-run, single-flight) ───────────────────────
#
# The negative cache above only suppresses re-hammering KNOWN-BAD URLs. This
# positive cache stores SUCCESSFUL scrape content so the same URL fetched by
# many sibling agents costs one upstream round-trip, not N.
#
# Why it pays: sibling agents researching one question converge on the same
# canonical pages, so unique URLs run far below total fetches (typically a
# 3-5x ratio on a fan-out run). Agents dispatched with asyncio.gather share
# one process and event loop, so a module-level singleton reaches all of them
# with zero plumbing.
#
# Scope = SCRAPE ONLY. The per-call SUMMARY_LLM extraction is deliberately NOT
# cached: sharing raw page bytes can't homogenise the runs' reasoning (same URL
# is the same content for everyone), but sharing extraction would leak one
# run's info_to_extract focus into another and couple the otherwise-independent
# trajectories. Successes only — a failed scrape is never stored, and a waiter
# whose leader failed/timed-out falls back to its own fetch, so the cache never
# converts one transient failure into a correlated N-run failure.

# Follower wait cap before falling back to an independent scrape. Bounds the
# "leader stalled → everyone blocks" failure mode.
_SINGLE_FLIGHT_TIMEOUT_S = 90.0
# Cap on distinct cached pages (FIFO eviction) — guards memory on long runs.
_MAX_CACHE_ENTRIES = 1024


def _positive_cache_enabled() -> bool:
    """On by default; set ``WEB_FETCH_SCRAPE_CACHE=0`` (or false/off) to disable."""
    raw = (os.environ.get("WEB_FETCH_SCRAPE_CACHE") or "").strip().lower()
    return raw not in {"0", "false", "off", "no"}


class ScrapeUnavailable(Exception):
    """Raised by a scrape callable when content could not be obtained.

    Signals the cache to NOT store a result and lets waiters fall back to an
    independent fetch. Carries the original error string for the caller's
    user-facing message.
    """


class ScrapeResultCache:
    """Process-global positive cache for successful scrapes, with single-flight.

    ``get_or_scrape(url, scrape_fn)`` returns cached content on a hit; on a miss
    the first caller (the *leader*) runs ``scrape_fn`` while concurrent callers
    for the same URL (*followers*) await its result instead of duplicating the
    fetch. The leader caches only successful, non-empty content.
    """

    def __init__(self) -> None:
        self._content: dict[str, str] = {}
        self._inflight: dict[str, asyncio.Future[str]] = {}
        self._lock = asyncio.Lock()
        self.hits = 0
        self.misses = 0
        self.coalesced = 0

    async def get_or_scrape(
        self,
        url: str,
        scrape_fn: Callable[[], Awaitable[str]],
        *,
        single_flight_timeout: float = _SINGLE_FLIGHT_TIMEOUT_S,
        should_cache: Callable[[str], bool] | None = None,
    ) -> str:
        """Return cached content for ``url`` or run ``scrape_fn`` to produce it.

        ``scrape_fn`` must return the scraped content string on success and
        raise :class:`ScrapeUnavailable` (or any exception) on failure — only
        successful, non-empty returns accepted by ``should_cache`` are cached.
        ``should_cache`` defaults to accepting every non-empty string; callers
        can return a low-confidence result to the current request without
        poisoning later requests with it. Raises whatever ``scrape_fn`` raises
        for the leader; followers fall back to their own ``scrape_fn`` on the
        leader's failure or a wait timeout.
        """
        if not url or not _positive_cache_enabled():
            return await scrape_fn()

        async with self._lock:
            if url in self._content:
                self.hits += 1
                # A cache hit is one upstream round-trip saved.
                record_api_request("jina", requests=0, cache_hits=1)
                return self._content[url]
            fut = self._inflight.get(url)
            leader = fut is None
            if leader:
                self.misses += 1
                fut = asyncio.get_event_loop().create_future()
                self._inflight[url] = fut
            else:
                self.coalesced += 1
                # Coalesced follower shares the leader's round-trip —
                # also a saved upstream request.
                record_api_request("jina", requests=0, cache_hits=1)

        if not leader:
            try:
                # ``shield`` so a waiter's timeout can't cancel the shared work.
                return await asyncio.wait_for(
                    asyncio.shield(fut), timeout=single_flight_timeout,
                )
            except Exception:
                # Leader failed / timed out → independent fetch (resilience).
                return await scrape_fn()

        # Leader path.
        try:
            content = await scrape_fn()
        except BaseException as exc:
            async with self._lock:
                self._inflight.pop(url, None)
            if not fut.done():
                fut.set_exception(exc)
                # Retrieve eagerly so a leader failure with no waiting follower
                # doesn't log "Future exception was never retrieved". Waiters
                # (if any) still see it via their own ``await``.
                fut.exception()
            raise
        cacheable = isinstance(content, str) and bool(content.strip())
        if cacheable and should_cache is not None:
            try:
                cacheable = bool(should_cache(content))
            except Exception:
                # Cache policy is an optimization boundary: a buggy predicate
                # must not fail a successful scrape or strand coalesced waiters.
                cacheable = False
        async with self._lock:
            self._inflight.pop(url, None)
            if cacheable:
                if len(self._content) >= _MAX_CACHE_ENTRIES:
                    self._content.pop(next(iter(self._content)), None)
                self._content[url] = content
        if not fut.done():
            fut.set_result(content)
        return content

    def clear(self) -> None:
        """Drop all entries + counters. Primarily for tests."""
        self._content.clear()
        self._inflight.clear()
        self.hits = self.misses = self.coalesced = 0

    def stats(self) -> dict[str, int]:
        return {
            "hits": self.hits,
            "misses": self.misses,
            "coalesced": self.coalesced,
            "size": len(self._content),
        }


# Module-level singleton — shared across all sibling agents in the process.
scrape_result_cache = ScrapeResultCache()
