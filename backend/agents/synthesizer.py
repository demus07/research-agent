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

        # attempt 3: JSON was truncated — rescue individual fields via regex
        rescued = self._rescue_fields(cleaned, findings)
        if rescued:
            return rescued

        # final fallback: build entirely from scraped content, no raw LLM text
        return self._fallback_from_findings(findings)

    def _rescue_fields(self, text: str, findings: list[ResearchFinding]) -> dict | None:
        """Extract whatever fields are present from a truncated JSON string."""
        if "{" not in text:
            return None

        def get_str(field: str) -> str:
            m = re.search(rf'"{field}"\s*:\s*"((?:[^"\\]|\\.)*)"', text, re.DOTALL)
            return m.group(1).replace('\\"', '"').replace("\\n", "\n") if m else ""

        def get_list(field: str) -> list[str]:
            m = re.search(rf'"{field}"\s*:\s*\[(.*?)\]', text, re.DOTALL)
            if not m:
                return []
            return [i.replace('\\"', '"') for i in re.findall(r'"((?:[^"\\]|\\.)*)"', m.group(1))]

        summary = get_str("executive_summary")
        conclusion = get_str("conclusion")
        key_findings = get_list("key_findings")

        # only use rescue if we got at least something meaningful
        if not summary and not conclusion:
            return None

        # rebuild sections from findings if the sections field is unusable
        sections = []
        for finding in findings:
            combined = " ".join(s.content for s in finding.sources if s.content)
            sections.append({
                "subquestion": finding.subquestion,
                "analysis": combined[:600] if combined else "No data retrieved for this subquestion.",
            })

        return {
            "executive_summary": summary or "See detailed sections below.",
            "key_findings": key_findings or ["See individual sections for details."],
            "sections": sections,
            "tradeoffs": get_list("tradeoffs"),
            "conclusion": conclusion or summary,
            "comparison_table": [],
        }

    def _fallback_from_findings(self, findings: list[ResearchFinding]) -> dict:
        """Build a plain synthesis directly from scraped content."""
        sections = []
        all_content = []
        for finding in findings:
            combined = " ".join(s.content for s in finding.sources if s.content)
            all_content.append(combined)
            sections.append({
                "subquestion": finding.subquestion,
                "analysis": combined[:600] if combined else "No data retrieved for this subquestion.",
            })

        summary = " ".join(all_content)[:300] if all_content else "No data was retrieved."
        return {
            "executive_summary": summary,
            "key_findings": ["See individual sections for details."],
            "sections": sections,
            "tradeoffs": [],
            "conclusion": "See the detailed analysis sections above for full findings.",
            "comparison_table": [],
        }
