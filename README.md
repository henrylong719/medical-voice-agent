# Medical Voice Agent

Medical Voice Agent is a backend-first prototype of a university health clinic scheduling assistant. It combines a FastAPI API, Supabase/Postgres, LangGraph, Anthropic, and OpenAI embeddings to identify patients by UIN, triage symptoms, find appointment availability, and manage bookings through a conversational interface.

The active implementation lives in [`backend/`](backend/). The planning history and future-phase notes live in [`docs/`](docs/).

## Current Status

The repository is currently in a Phase 3-style state:

- Phase 1 foundations are in place: schema, seed data, admin APIs, scheduling engine, and time utilities.
- Phase 2 agent work is in place: LangGraph agent, tool calling, patient identification, booking/rescheduling/cancellation, and chat endpoints.
- Phase 3 hybrid triage work is in place: pgvector-backed medical knowledge retrieval plus keyword symptom matching.

There is not a frontend or realtime voice client in this repo yet. The project is currently centered on the backend API and supporting SQL/docs.

## What The Project Does Today

- Identifies returning patients by 9-digit UIN.
- Registers new patients when a UIN is not found.
- Matches symptoms to specialties with hybrid triage using keyword search over `symptom_specialty_map` plus semantic search over `medical_knowledge`.
- Finds open slots across doctors or for a specific doctor.
- Books, reschedules, and cancels appointments.
- Streams chat responses over Server-Sent Events.
- Persists conversation state in Postgres using LangGraph's `AsyncPostgresSaver`.
- Exposes admin APIs for specialties, doctors, patients, appointments, blocks, and slots.

## Architecture At A Glance

1. FastAPI exposes admin endpoints and agent-facing chat endpoints.
2. The chat API calls a LangGraph-based agent configured in `backend/app/agent/graph.py`.
3. The agent uses tool calls defined in `backend/app/agent/tools.py`.
4. Tools talk directly to Supabase for patient, doctor, appointment, and specialty data.
5. The slot engine computes availability from recurring doctor schedules minus booked appointments and doctor blocks.
6. Hybrid triage combines keyword lookup in `symptom_specialty_map` with semantic retrieval through pgvector via `match_medical_knowledge`.
7. Conversation memory is persisted in Supabase Postgres via `SUPABASE_DB_URI`.

## Project Layout

```text
.
├── backend/
│   ├── app/
│   │   ├── agent/          # system prompt, tools, and LangGraph agent wiring
│   │   ├── api/            # admin and chat FastAPI routes
│   │   ├── models/         # Pydantic models and typed DB row shapes
│   │   ├── services/       # scheduling, time parsing, RAG retrieval, ingestion
│   │   ├── config.py       # environment-driven settings
│   │   ├── main.py         # FastAPI entry point
│   │   └── supabase_client.py
│   ├── sql/
│   │   ├── schema.sql
│   │   ├── seed.sql
│   │   ├── create_doctor_with_details.sql
│   │   └── rag.sql
│   ├── .env.example
│   ├── pyproject.toml
│   └── uv.lock
├── docs/                   # phase notes, planning prompts, flow diagram
├── LICENSE
└── todo.md
```

## Tech Stack

- Python 3.12
- `uv` for environment and dependency management
- FastAPI + Uvicorn
- Supabase + PostgreSQL
- pgvector for semantic retrieval
- LangChain + LangGraph
- Anthropic Claude for agent responses
- OpenAI embeddings for semantic search
- LangSmith for tracing and observability

## Setup

### 1. Prerequisites

- Python 3.12+
- [`uv`](https://docs.astral.sh/uv/)
- A Supabase project
- An Anthropic API key
- An OpenAI API key for embedding ingestion and semantic retrieval
- A LangSmith key if you want tracing

### 2. Configure Environment Variables

```bash
cd backend
cp .env.example .env
```

Fill in the important values in `backend/.env`:

| Variable | Required | Purpose |
| --- | --- | --- |
| `SUPABASE_URL` | Yes | Supabase project URL |
| `SUPABASE_SERVICE_KEY` | Yes | Backend service-role key |
| `SUPABASE_DB_URI` | Yes for chat | Direct Postgres URI used by `AsyncPostgresSaver` |
| `ANTHROPIC_API_KEY` | Yes for chat | Claude API key |
| `ANTHROPIC_MODEL` | No | Defaults to `claude-haiku-4-5-20251001` |
| `OPENAI_API_KEY` | Yes for Phase 3 RAG | Embeddings for ingestion and semantic triage |
| `LANGSMITH_API_KEY` | Optional | LangSmith tracing |
| `LANGSMITH_TRACING` | Optional | Usually `true` |
| `LANGSMITH_PROJECT` | Optional | Trace grouping/project name |
| `TIMEZONE` | No | Clinic timezone, defaults to `America/Chicago` |
| `SCHEDULING_HORIZON_DAYS` | No | How far ahead to search for slots |
| `DEFAULT_SLOT_DURATION_MIN` | No | Fallback slot duration |

Keep the service-role key local to the backend only. Do not expose it to a browser client.

### 3. Initialize Supabase

Run these SQL files in your Supabase SQL editor, in this order:

1. `backend/sql/schema.sql`
2. `backend/sql/seed.sql`
3. `backend/sql/create_doctor_with_details.sql`
4. `backend/sql/rag.sql`

Notes:

- The first three files get the clinic data model, sample data, and doctor-creation RPC in place.
- `rag.sql` enables `pgvector`, creates `medical_knowledge`, and adds the `match_medical_knowledge` RPC used by hybrid triage.

### 4. Install Dependencies

```bash
cd backend
uv sync
```

### 5. Ingest Medical Knowledge Chunks

This populates the `medical_knowledge` table with embedded symptom-cluster passages used by semantic triage.

```bash
cd backend
uv run python -m app.services.ingest_knowledge
```

### 6. Run The API

```bash
cd backend
uv run uvicorn app.main:app --reload
```

Open:

- Swagger UI: `http://localhost:8000/docs`
- Health check: `http://localhost:8000/health`

## API Overview

### Admin Endpoints

Mounted under `/api/v1/admin`:

- `GET /specialties`
- `GET /specialties/{specialty_id}`
- `GET /doctors`
- `GET /doctors/{doctor_id}`
- `POST /doctors`
- `GET /patients`
- `GET /patients/uin/{uin}`
- `POST /patients`
- `GET /appointments`
- `GET /appointments/{appointment_id}`
- `GET /blocks`
- `POST /blocks`
- `GET /slots/by-specialty`
- `GET /slots/by-doctor`

### Chat Endpoints

Mounted under `/api/v1`:

- `POST /chat`
  Streaming SSE response.
- `POST /chat/invoke`
  Full JSON response for easier testing.

## Example Requests

### Streaming Chat

Reuse the same `thread_id` across messages if you want conversation memory.

```bash
curl -N -X POST http://localhost:8000/api/v1/chat \
  -H "Content-Type: application/json" \
  -d '{
    "message": "I need help booking an appointment for recurring headaches.",
    "thread_id": "demo-thread-1"
  }'
```

### Non-Streaming Chat

```bash
curl -X POST http://localhost:8000/api/v1/chat/invoke \
  -H "Content-Type: application/json" \
  -d '{
    "message": "My UIN is 123456789 and I need to reschedule my appointment.",
    "thread_id": "demo-thread-1"
  }'
```

### Slot Search

```bash
curl "http://localhost:8000/api/v1/admin/slots/by-specialty?specialty_id=a1000000-0000-0000-0000-000000000001&preferred_day=next%20monday&preferred_time=morning"
```

## Key Implementation Notes

- Patient identity is a first-class part of the workflow. The agent is prompted to ask for a UIN before doing anything else.
- Scheduling is computed from recurring availability templates, then filtered by booked appointments and doctor time-off blocks.
- RAG retrieval uses OpenAI embeddings and a Supabase RPC rather than embedding logic inside the agent tool layer.
- If semantic retrieval is unavailable, the triage tool falls back to keyword results instead of crashing the whole flow.
- The seed data includes 10 specialties, 8 doctors, symptom mappings, patients, and appointment data for development.

## Validation And Smoke Tests

Interactive API docs are available at `http://localhost:8000/docs`.

For the RAG retriever smoke test:

```bash
cd backend
uv run python -m app.services.test_retriever
```

This script expects:

- `OPENAI_API_KEY` to be configured
- `backend/sql/rag.sql` to have been applied
- `medical_knowledge` to be populated via the ingestion script

## Roadmap And Docs

The docs folder contains both the current phase notes and the forward-looking project plan.

- [Current Phase 3 notes](<docs/Medical Voice Agent — Project Instruction Prompt_phase3.md>)
- [Combined phase status notes](<docs/Medical Voice Agent — Project Instruction Prompt_phases.md>)
- [Original project planning prompt](<docs/Medical Voice Agent — Project Instruction Prompt.md>)
- [Phase 3 hybrid triage flow diagram](docs/phase3_hybrid_rag_triage_flow.svg)
- [Long-form working notes](docs/record.md)

Planned next stages referenced in the docs:

- Phase 4: multi-agent LangGraph supervisor with intake, triage, and scheduling sub-agents
- Phase 5: guardrails and safety boundaries
- Phase 6: realtime voice pipeline
- Phase 7: evaluation and prompt optimization

## License

[MIT](LICENSE)
