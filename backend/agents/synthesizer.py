import json
import os
import re
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

        prompt = f"""You are a research synthesizer. Output ONLY a valid JSON object — no markdown, no explanation, no code fences.

Use exactly these keys:
- "executive_summary": one paragraph string
- "key_findings": array of 3-5 short strings
- "sections": array of objects, each with "subquestion" (string) and "analysis" (string)
- "tradeoffs": array of short strings (can be empty array)
- "conclusion": one paragraph string
- "comparison_table": array of objects with "aspect" and "details" keys (can be empty array)

RESEARCH DATA:
{research_text}

JSON output (start with {{ and end with }}):"""

        async with httpx.AsyncClient(timeout=180.0) as client:
            try:
                response = await client.post(
                    f"{self.ollama_url}/api/generate",
                    json={"model": self.model, "prompt": prompt, "stream": False},
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
        # strip markdown code fences if model wrapped output
        cleaned = re.sub(r"```json|```", "", raw).strip()

        # attempt 1: parse the whole cleaned string
        try:
            parsed = json.loads(cleaned)
            if "conclusion" in parsed:
                return parsed
        except json.JSONDecodeError:
            pass

        # attempt 2: extract first {...} block (handles preamble/postamble text)
        match = re.search(r"\{.*\}", cleaned, re.DOTALL)
        if match:
            try:
                parsed = json.loads(match.group())
                if "conclusion" in parsed:
                    return parsed
            except json.JSONDecodeError:
                pass

        # fallback: build structure from raw findings
        sections = []
        for finding in findings:
            combined = " ".join(s.content for s in finding.sources if s.content)
            sections.append({
                "subquestion": finding.subquestion,
                "analysis": combined[:600] if combined else "No data retrieved for this subquestion.",
            })

        return {
            "executive_summary": raw[:300] if raw else "Synthesis unavailable.",
            "key_findings": ["See individual sections for details."],
            "sections": sections,
            "tradeoffs": [],
            "conclusion": raw[300:800] if len(raw) > 300 else "See executive summary.",
            "comparison_table": [],
        }
