import json
import os
import httpx

from backend.models import ResearchFinding


OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
MODEL = os.getenv("MODEL", "llama3")


class SynthesizerAgent:
    def __init__(self):
        self.model = MODEL
        self.ollama_url = OLLAMA_URL

    async def synthesize(self, findings: list[ResearchFinding]) -> dict:
        research_text = self._format_findings(findings)

        prompt = f"""You are a research synthesis expert. Analyze the following research findings and produce a structured synthesis.

RESEARCH FINDINGS:
{research_text}

Produce a thorough synthesis in valid JSON format. Respond ONLY with the JSON object, no preamble or explanation:
{{
  "executive_summary": "2-3 sentence overview of the entire research topic",
  "key_findings": ["finding 1", "finding 2", "finding 3", "finding 4"],
  "sections": [
    {{
      "subquestion": "the subquestion",
      "analysis": "detailed 2-4 paragraph analysis for this subquestion based on the sources"
    }}
  ],
  "tradeoffs": ["tradeoff or comparison point 1", "tradeoff 2", "tradeoff 3"],
  "conclusion": "3-5 sentence conclusion synthesizing everything",
  "comparison_table": [
    {{"aspect": "aspect name", "details": "comparison details"}}
  ]
}}"""

        async with httpx.AsyncClient(timeout=180.0) as client:
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

        data = response.json()
        raw = data.get("response", "")
        return self._parse_synthesis(raw, findings)

    def _format_findings(self, findings: list[ResearchFinding]) -> str:
        parts = []
        for i, finding in enumerate(findings, 1):
            parts.append(f"\n### Subquestion {i}: {finding.subquestion}")
            for j, source in enumerate(finding.sources, 1):
                parts.append(f"\nSource {j}: {source.title}")
                parts.append(f"URL: {source.url}")
                if source.content:
                    parts.append(f"Content: {source.content[:800]}")
                else:
                    parts.append("Content: (no content retrieved)")
        return "\n".join(parts)

    def _parse_synthesis(self, raw: str, findings: list[ResearchFinding]) -> dict:
        start = raw.find("{")
        end = raw.rfind("}") + 1
        if start != -1 and end > start:
            try:
                parsed = json.loads(raw[start:end])
                if "sections" in parsed and "conclusion" in parsed:
                    return parsed
            except json.JSONDecodeError:
                pass

        # Fallback: build a basic structure from findings
        sections = []
        for finding in findings:
            combined = " ".join(s.content for s in finding.sources if s.content)
            sections.append({
                "subquestion": finding.subquestion,
                "analysis": combined[:600] if combined else "No data retrieved for this subquestion.",
            })

        return {
            "executive_summary": "Research synthesis could not be fully parsed from the model response.",
            "key_findings": ["See individual sections for details."],
            "sections": sections,
            "tradeoffs": [],
            "conclusion": raw[:500] if raw else "No synthesis available.",
            "comparison_table": [],
        }
