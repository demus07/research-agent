import json
import os
import httpx


OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
MODEL = os.getenv("MODEL", "qwen2.5:7b")


class PlannerAgent:
    def __init__(self):
        self.model = MODEL
        self.ollama_url = OLLAMA_URL

    async def plan(self, query: str) -> list[str]:
        prompt = f"""You are a research planning assistant. Given a research query, write 5-6 research questions that together form a complete outline of the topic.

Rules:
- Each question should cover a distinct aspect of the topic
- Questions should read like a research outline, not web search queries
- They should be broad enough to find multiple sources, but focused on one aspect
- Use natural research question phrasing: "What content types...", "How to...", "Which strategies..."
- NEVER ask for ranked lists of specific people, follower counts, or named influencers — these cannot be answered via web scraping. Instead rephrase as characteristics or patterns: not "Who are the top 5 fitness influencers" but "What content style do leading fitness Instagram accounts use"

Example for "grow Instagram personal brand":
- "What content types drive follower growth on Instagram"
- "How to build a consistent Instagram brand identity"
- "Instagram engagement strategies for organic growth"
- "How to use Instagram analytics to improve performance"
- "Hashtag and posting schedule strategies for Instagram"

Research query: {query}

Respond ONLY with a valid JSON object in this exact format, no other text:
{{
  "subquestions": [
    "research question 1",
    "research question 2",
    "research question 3",
    "research question 4",
    "research question 5"
  ]
}}"""

        async with httpx.AsyncClient(timeout=120.0) as client:
            try:
                response = await client.post(
                    f"{self.ollama_url}/api/generate",
                    json={
                        "model": self.model,
                        "prompt": prompt,
                        "stream": False,
                    },
                )
                response.raise_for_status()
            except httpx.ConnectError:
                raise RuntimeError(
                    f"Cannot connect to Ollama at {self.ollama_url}. "
                    "Make sure Ollama is running: ollama serve"
                )
            except httpx.HTTPStatusError as e:
                raise RuntimeError(f"Ollama returned error: {e.response.status_code}")

        data = response.json()
        raw = data.get("response", "")

        subquestions = self._parse_subquestions(raw, query)
        return subquestions

    def _parse_subquestions(self, raw: str, original_query: str) -> list[str]:
        # Try to extract JSON from the response
        start = raw.find("{")
        end = raw.rfind("}") + 1
        if start != -1 and end > start:
            try:
                parsed = json.loads(raw[start:end])
                subquestions = parsed.get("subquestions", [])
                if isinstance(subquestions, list) and len(subquestions) >= 2:
                    return [str(q) for q in subquestions[:6]]
            except json.JSONDecodeError:
                pass

        # Fallback: extract numbered lines
        lines = [line.strip() for line in raw.split("\n") if line.strip()]
        subquestions = []
        for line in lines:
            # Strip common list prefixes like "1.", "- ", "* "
            for prefix in ["1.", "2.", "3.", "4.", "5.", "6.", "-", "*", "•"]:
                if line.startswith(prefix):
                    line = line[len(prefix):].strip()
                    break
            if len(line) > 10 and "?" in line or len(line) > 20:
                subquestions.append(line)
            if len(subquestions) >= 6:
                break

        if len(subquestions) >= 2:
            return subquestions

        # Last resort: generate generic subquestions
        return [
            f"What is the definition and background of {original_query}?",
            f"What are the key aspects and components of {original_query}?",
            f"What are the main challenges or limitations related to {original_query}?",
            f"What are the latest developments or trends in {original_query}?",
            f"What are the practical applications and use cases of {original_query}?",
        ]
