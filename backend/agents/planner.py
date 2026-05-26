import json
import os
import httpx


OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
MODEL = os.getenv("MODEL", "llama3")


class PlannerAgent:
    def __init__(self):
        self.model = MODEL
        self.ollama_url = OLLAMA_URL

    async def plan(self, query: str) -> list[str]:
        prompt = f"""You are a research planning assistant. Given a research query, generate 4-6 specific, searchable subtopics that together cover the topic comprehensively.

Rules:
- Each subtopic must be a concrete, specific phrase — NOT an abstract question
- Write them like Google search queries a journalist would type, not academic questions
- Include specifics: tactics, comparisons, examples, tools, stats, trends
- BAD: "What are the key components of X?" or "How can organizations achieve Y?"
- GOOD: "Instagram personal brand growth tactics 2025" or "RAG vs fine-tuning LLM tradeoffs"

Research query: {query}

Respond ONLY with a valid JSON object in this exact format, no other text:
{{
  "subquestions": [
    "specific searchable subtopic 1",
    "specific searchable subtopic 2",
    "specific searchable subtopic 3",
    "specific searchable subtopic 4"
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
