# Autonomous Research Agent

A web app where you type a research query and an AI system autonomously plans, researches, synthesizes, and generates a full Markdown + PDF report — all locally, no API keys required.

## Requirements

- Python 3.10+
- [Ollama](https://ollama.ai) installed and running locally
- The `llama3` model pulled: `ollama pull llama3`

## Setup

```bash
# 1. Install dependencies
cd backend
pip install -r requirements.txt

# 2. Make sure Ollama is running
ollama serve   # in a separate terminal if not already running

# 3. Start the backend
cd ..
uvicorn backend.main:app --reload --port 8000

# 4. Open the frontend
open frontend/index.html
# or just double-click frontend/index.html in your file manager
```

## Changing the Model

Set the `MODEL` environment variable before starting the server:

```bash
MODEL=mistral uvicorn backend.main:app --reload --port 8000
MODEL=llama3:70b uvicorn backend.main:app --reload --port 8000
```

You can also change the Ollama endpoint:

```bash
OLLAMA_URL=http://192.168.1.10:11434 uvicorn backend.main:app --reload --port 8000
```

## Example Queries to Try

- "What are the tradeoffs between microservices and monolithic architectures?"
- "Impact of large language models on software development productivity"
- "Pros and cons of nuclear energy vs renewable energy"
- "State of quantum computing in 2024: what works and what doesn't"
- "Best practices for building RAG systems with vector databases"
- "History and future of the Rust programming language"

## How It Works

1. **Planner** — sends your query to Ollama and gets back 4–6 focused subquestions
2. **Researcher** — searches DuckDuckGo for each subquestion, fetches and scrapes the top 3 URLs
3. **Synthesizer** — feeds all scraped content back to Ollama for structured synthesis
4. **Writer** — formats everything into a professional Markdown report and renders it to PDF via WeasyPrint

Reports are saved to `/outputs` and available for download from the frontend.
