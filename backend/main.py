import asyncio
import json
import os
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from sse_starlette.sse import EventSourceResponse

from backend.agents import PlannerAgent, ResearcherAgent, SynthesizerAgent, WriterAgent
from backend.models import ResearchRequest

OUTPUTS_DIR = Path(__file__).resolve().parent.parent / "outputs"
OUTPUTS_DIR.mkdir(exist_ok=True)

app = FastAPI(title="Autonomous Research Agent")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/research")
async def research(request: ResearchRequest):
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty.")

    async def event_stream():
        planner = PlannerAgent()
        researcher = ResearcherAgent()
        synthesizer = SynthesizerAgent()
        writer = WriterAgent()

        def send(msg: str, data: dict | None = None):
            payload = {"message": msg}
            if data:
                payload.update(data)
            return json.dumps(payload)

        try:
            yield {"data": send("Planning research...")}
            await asyncio.sleep(0)

            subquestions = await planner.plan(request.query)

            yield {"data": send(f"Planned {len(subquestions)} subquestions.", {"subquestions": subquestions})}
            await asyncio.sleep(0)

            findings = []
            for i, subq in enumerate(subquestions, 1):
                yield {"data": send(f"Researching subquestion {i}/{len(subquestions)}: {subq[:80]}...")}
                await asyncio.sleep(0)
                partial = await researcher.research([subq])
                findings.extend(partial)

            yield {"data": send("Synthesizing findings...")}
            await asyncio.sleep(0)

            synthesis = await synthesizer.synthesize(findings)

            yield {"data": send("Generating report...")}
            await asyncio.sleep(0)

            md_file, pdf_file, md_content = writer.write(
                request.query, subquestions, findings, synthesis
            )

            yield {
                "data": send(
                    "Done.",
                    {
                        "done": True,
                        "markdown_file": md_file,
                        "pdf_file": pdf_file,
                        "markdown_content": md_content,
                    },
                )
            }

        except RuntimeError as e:
            yield {"data": send(f"Error: {e}", {"error": True})}
        except Exception as e:
            yield {"data": send(f"Unexpected error: {e}", {"error": True})}

    return EventSourceResponse(event_stream())


@app.get("/download/{filename}")
async def download(filename: str):
    # Prevent path traversal
    safe_name = Path(filename).name
    file_path = OUTPUTS_DIR / safe_name
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="File not found.")
    media_type = "text/markdown" if safe_name.endswith(".md") else "application/pdf"
    return FileResponse(str(file_path), media_type=media_type, filename=safe_name)


@app.get("/health")
async def health():
    return {"status": "ok"}
