import httpx
from bs4 import BeautifulSoup
from duckduckgo_search import DDGS

from backend.models import ResearchFinding, Source

MAX_CONTENT_CHARS = 1500
RESULTS_PER_QUERY = 3
SCRAPE_TIMEOUT = 15.0


class ResearcherAgent:
    async def research(self, subquestions: list[str]) -> list[ResearchFinding]:
        findings = []
        for subquestion in subquestions:
            sources = await self._research_subquestion(subquestion)
            findings.append(ResearchFinding(subquestion=subquestion, sources=sources))
        return findings

    async def _research_subquestion(self, subquestion: str) -> list[Source]:
        results = self._search(subquestion)
        sources = []
        async with httpx.AsyncClient(
            timeout=SCRAPE_TIMEOUT,
            follow_redirects=True,
            headers={"User-Agent": "Mozilla/5.0 (compatible; ResearchBot/1.0)"},
        ) as client:
            for result in results[:RESULTS_PER_QUERY]:
                url = result.get("href", "")
                title = result.get("title", "No title")
                if not url:
                    continue
                content = await self._scrape(client, url)
                sources.append(Source(url=url, title=title, content=content))
        return sources

    def _search(self, query: str) -> list[dict]:
        try:
            with DDGS() as ddgs:
                results = list(ddgs.text(query, max_results=RESULTS_PER_QUERY + 2))
            return results
        except Exception:
            return []

    async def _scrape(self, client: httpx.AsyncClient, url: str) -> str:
        try:
            response = await client.get(url)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, "html.parser")
            paragraphs = soup.find_all("p")
            text = " ".join(p.get_text(separator=" ", strip=True) for p in paragraphs)
            text = " ".join(text.split())  # normalize whitespace
            return text[:MAX_CONTENT_CHARS]
        except Exception:
            return ""
