\# Data-Analyst Telegram Bot



An LLM-powered Telegram bot that answers data-analysis questions with a single JSON reply.



\## Architecture

\- FastAPI web server (`/health`, `/run.jsonl`)

\- Telegram long-polling loop (background thread)

\- Gemini API agent loop with a `run\_python` tool for real computation/data-fetching

\- Per-chat conversation history (multi-turn support)

\- JSONL run log served publicly at `/run.jsonl`



\## Stack

\- Python, FastAPI, Uvicorn

\- Google Gemini API (`gemini-flash-lite-latest`)

\- Deployed on Render (free tier), kept warm via self-ping + UptimeRobot



\## Environment variables

\- `BOT\_TOKEN` — Telegram bot token

\- `GEMINI\_API\_KEY` — Gemini API key

\- `BASE\_URL` — public base URL of the deployment

