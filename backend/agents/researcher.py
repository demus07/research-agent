import asyncio
import os
import re
import time
import httpx
from bs4 import BeautifulSoup
from googlesearch import search as google_search

from backend.models import ResearchFinding, Source

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
MODEL = os.getenv("MODEL", "qwen2.5:7b")

MAX_CONTENT_CHARS = 1500
RESULTS_PER_QUERY = 3
SCRAPE_TIMEOUT = 15.0
SEARCH_DELAY = 2.5
FETCH_EXTRA = 4

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    )
}

_BLOCKED_DOMAINS = {
    "instagram.com", "facebook.com", "twitter.com", "x.com",
    "tiktok.com", "linkedin.com", "pinterest.com", "snapchat.com",
    "reddit.com", "quora.com",
    "apps.apple.com", "play.google.com", "youtube.com", "vimeo.com",
    "merriam-webster.com", "dictionary.com", "cambridge.org",
    "amazon.com", "amazon.in", "flipkart.com",
    "jio.com", "hotstar.com", "indiatimes.com", "ndtv.com", "timesofindia.com",
}


def _search_web(query: str, max_results: int = RESULTS_PER_QUERY + FETCH_EXTRA) -> list[dict]:
    results = []
    try:
        for url in google_search(query, num_results=max_results, lang="en", sleep_interval=1):
            results.append({"href": url, "title": "", "body": ""})
    except Exception:
        pass
    return results


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
        async with httpx.AsyncClient(
            timeout=SCRAPE_TIMEOUT,
            follow_redirects=True,
            headers=HEADERS,
        ) as client:
            for result in results:
                if len(sources) >= RESULTS_PER_QUERY:
                    break
                url = result.get("href", "")
                if not url or self._is_blocked(url) or not self._has_content_path(url):
                    continue
                title, content = await self._scrape(client, url)
                if not content:
                    continue
                if len(title) < 25:
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

    def _is_blocked(self, url: str) -> bool:
        try:
            from urllib.parse import urlparse
            host = urlparse(url).netloc.lower().lstrip("www.")
            return any(host == d or host.endswith("." + d) for d in _BLOCKED_DOMAINS)
        except Exception:
            return False

    def _has_content_path(self, url: str) -> bool:
        try:
            from urllib.parse import urlparse
            path = urlparse(url).path.rstrip("/")
            return len(path) > 1
        except Exception:
            return True

    async def _scrape(self, client: httpx.AsyncClient, url: str) -> tuple[str, str]:
        try:
            response = await client.get(url)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, "html.parser")
            # title from <title> tag
            title_tag = soup.find("title")
            title = title_tag.get_text(strip=True) if title_tag else url
            # clean up common title suffixes like " | Site Name"
            title = re.split(r"\s[\|\-—]\s", title)[0].strip()
            # content from <p> tags
            for tag in soup(["script", "style", "nav", "footer", "header"]):
                tag.decompose()
            paragraphs = soup.find_all("p")
            text = " ".join(p.get_text(separator=" ", strip=True) for p in paragraphs)
            text = " ".join(text.split())
            return title, text[:MAX_CONTENT_CHARS]
        except Exception:
            return url, ""


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
