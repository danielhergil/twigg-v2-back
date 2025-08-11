# Course Generator API (FastAPI + OpenRouter + Supabase)

Generate courses **module-by-module** with near-real-time feedback via **SSE** and publish to Supabase.

## Run locally

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env  # fill it
uvicorn main:app --reload --port 8000