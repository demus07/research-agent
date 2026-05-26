import asyncio
import os
import re
import httpx
from tavily import TavilyClient

from backend.models import ResearchFinding, Source

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
MODEL = os.getenv("MODEL", "qwen2.5:7b")

RESULTS_PER_QUERY = 3
SEARCH_DELAY = 1.0

_BLOCKED_DOMAINS = {
    "instagram.com", "facebook.com", "twitter.com", "x.com",
    "tiktok.com", "linkedin.com", "pinterest.com", "snapchat.com",
    "reddit.com", "quora.com",
    "apps.apple.com", "play.google.com", "youtube.com", "vimeo.com",
    "merriam-webster.com", "dictionary.com", "cambridge.org",
    "amazon.com", "amazon.in", "flipkart.com",
    "jio.com", "hotstar.com", "indiatimes.com", "ndtv.com", "timesofindia.com",
}


def _get_tavily_client() -> TavilyClient:
    api_key = os.getenv("TAVILY_API_KEY")
    if not api_key:
        raise RuntimeError("TAVILY_API_KEY is not set. Add it to your .env file.")
    return TavilyClient(api_key=api_key)


def _search_web(query: str, max_results: int = RESULTS_PER_QUERY + 2) -> list[dict]:
    client = _get_tavily_client()
    response = client.search(query, max_results=max_results, search_depth="basic")
    return [
        {
            "href": r["url"],
            "title": r.get("title", r["url"]),
            "body": r.get("content", ""),
        }
        for r in response.get("results", [])
    ]


def _is_blocked(url: str) -> bool:
    try:
        from urllib.parse import urlparse
        host = urlparse(url).netloc.lower().lstrip("www.")
        return any(host == d or host.endswith("." + d) for d in _BLOCKED_DOMAINS)
    except Exception:
        return False


class ResearcherAgent:
    def __init__(self):
        self.model = MODEL
        self.ollama_url = OLLAMA_URL

    async def research(self, subquestions: list[str]) -> list[ResearchFinding]:
        findings = []
        for i, subquestion in enumerate(subquestions):
            if i > 0:
                await asyncio.sleep(SEARCH_DELAY)
            sources = await self._research_subquestion(subquestion)
            findings.append(ResearchFinding(subquestion=subquestion, sources=sources))
        return findings

    async def _research_subquestion(self, subquestion: str) -> list[Source]:
        search_query = await self._to_search_query(subquestion)
        results = _search_web(search_query)
        sources = []
        for result in results:
            if len(sources) >= RESULTS_PER_QUERY:
                break
            url = result.get("href", "")
            title = result.get("title", "")
            content = result.get("body", "")
            if not url or _is_blocked(url):
                continue
            if not content:
                continue
            sources.append(Source(url=url, title=title, content=content))
        return sources

    async def _to_search_query(self, subquestion: str) -> str:
        prompt = (
            "Convert this research question into a short Google search query "
            "(4-6 words max, no punctuation). Return only the search query, "
            f"nothing else: {subquestion}"
        )
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(
                    f"{self.ollama_url}/api/generate",
                    json={"model": self.model, "prompt": prompt, "stream": False},
                )
                response.raise_for_status()
                raw = response.json().get("response", "").strip()
                query = raw.splitlines()[0].strip().strip('"').strip("'").strip(".")
                if query and len(query.split()) <= 10:
                    return query
        except Exception:
            pass
        return _clean_query(subquestion)


# Fallback query cleaner used when Ollama call fails
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
    prev = None
    while prev != q:
        prev = q
        q = _QUESTION_PREFIX.sub("", q).strip()
    q = _FILLER.sub("", q)
    q = re.sub(r"^(the|a|an)\s+", "", q, flags=re.IGNORECASE)
    q = " ".join(q.split())
    return q or subquestion
