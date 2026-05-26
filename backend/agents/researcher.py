import asyncio
import re
import httpx
from bs4 import BeautifulSoup
from duckduckgo_search import DDGS

from backend.models import ResearchFinding, Source

MAX_CONTENT_CHARS = 1500
RESULTS_PER_QUERY = 3
SCRAPE_TIMEOUT = 15.0
SEARCH_DELAY = 2.5

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    )
}

# Question words and auxiliaries that make terrible DDG search terms
_QUESTION_PREFIX = re.compile(
    r"^(what|how|why|when|where|who|which|can|could|should|would|will|does|do|is|are|has|have)\s+",
    re.IGNORECASE,
)
_FILLER = re.compile(
    r"\b(please|effectively|efficiently|successfully|organizations|companies|businesses|be implemented|be integrated|be used)\b",
    re.IGNORECASE,
)


def _clean_query(subquestion: str) -> str:
    q = subquestion.strip().rstrip("?")
    # strip leading question/aux words repeatedly until none remain
    prev = None
    while prev != q:
        prev = q
        q = _QUESTION_PREFIX.sub("", q).strip()
    q = _FILLER.sub("", q)
    # collapse whitespace
    q = re.sub(r"^(the|a|an)\s+", "", q, flags=re.IGNORECASE)
    q = " ".join(q.split())
    return q or subquestion  # fall back to original if we wiped everything


class ResearcherAgent:
    async def research(self, subquestions: list[str]) -> list[ResearchFinding]:
        findings = []
        for i, subquestion in enumerate(subquestions):
            if i > 0:
                await asyncio.sleep(SEARCH_DELAY)
            sources = await self._research_subquestion(subquestion)
            findings.append(ResearchFinding(subquestion=subquestion, sources=sources))
        return findings

    async def _research_subquestion(self, subquestion: str) -> list[Source]:
        search_query = _clean_query(subquestion)
        results = self._search(search_query)
        sources = []
        async with httpx.AsyncClient(
            timeout=SCRAPE_TIMEOUT,
            follow_redirects=True,
            headers=HEADERS,
        ) as client:
            for result in results[:RESULTS_PER_QUERY]:
                url = result.get("href", "")
                title = result.get("title", "No title")
                if not url:
                    continue
                content = await self._scrape(client, url)
                sources.append(Source(url=url, title=title, content=content))
        return sources

    def _search(self, query: str, retries: int = 2) -> list[dict]:
        for attempt in range(retries + 1):
            try:
                with DDGS() as ddgs:
                    results = list(ddgs.text(query, max_results=RESULTS_PER_QUERY + 2))
                if results:
                    return results
            except Exception:
                pass
            if attempt < retries:
                import time
                time.sleep(2.0)
        return []

    async def _scrape(self, client: httpx.AsyncClient, url: str) -> str:
        try:
            response = await client.get(url)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, "html.parser")
            for tag in soup(["script", "style", "nav", "footer", "header"]):
                tag.decompose()
            paragraphs = soup.find_all("p")
            text = " ".join(p.get_text(separator=" ", strip=True) for p in paragraphs)
            text = " ".join(text.split())
            return text[:MAX_CONTENT_CHARS]
        except Exception:
            return ""
