# Medical Voice Agent Backend

FastAPI backend for the medical scheduling assistant. The current implementation is a Phase 4-style backend with:

- admin APIs for specialties, doctors, patients, patient identifiers, appointments, blocks, and slot search
- a LangGraph-based multi-agent chat system with Supervisor, Intake, Triage, and Scheduling nodes
- demographic-first patient verification with fallback to strong identifiers
- hybrid triage using keyword matching plus pgvector retrieval
- persistent conversation memory via `AsyncPostgresSaver`

For the broader roadmap and phase notes, see [../README.md](../README.md) and [../docs/medical_voice_agent_plan_v3.md](../docs/medical_voice_agent_plan_v3.md).

## Quick Start

### 1. Configure Environment

```bash
cd backend
cp .env.example .env
```

Fill in `backend/.env`:

| Variable | Required | Purpose |
| --- | --- | --- |
| `SUPABASE_URL` | Yes | Supabase project URL |
| `SUPABASE_SERVICE_KEY` | Yes | Service-role key for backend DB access |
| `SUPABASE_DB_URI` | Yes for chat | Direct Postgres URI used by `AsyncPostgresSaver` |
| `ANTHROPIC_API_KEY` | Yes for chat | Claude API key for Supervisor and sub-agents |
| `ANTHROPIC_MODEL` | No | Defaults to `claude-haiku-4-5-20251001` |
| `OPENAI_API_KEY` | Yes for RAG ingestion/retrieval | Embeddings for `medical_knowledge` |
| `LANGSMITH_API_KEY` | Optional | LangSmith tracing |
| `LANGSMITH_TRACING` | Optional | Usually `true` |
| `LANGSMITH_PROJECT` | Optional | Trace project name |
| `TIMEZONE` | No | Clinic timezone, defaults to `America/Chicago` |
| `SCHEDULING_HORIZON_DAYS` | No | Booking horizon for slot search |
| `DEFAULT_SLOT_DURATION_MIN` | No | Fallback slot duration |

### 2. Initialize Supabase

Run these SQL files in order:

1. `sql/001_schema.sql`
2. `sql/002_seed.sql`
3. `sql/003_create_doctor_with_details.sql`
4. `sql/004_finalize_reschedule_appointment.sql`
5. `sql/005_rag.sql`

`005_rag.sql` enables `pgvector`, creates `medical_knowledge`, and adds the `match_medical_knowledge` RPC used by semantic retrieval.

### 3. Install Dependencies

```bash
cd backend
uv sync
```

### 4. Ingest Medical Knowledge

Required if you want semantic retrieval instead of keyword-only fallback.

```bash
cd backend
uv run python -m app.services.ingest_knowledge
```

### 5. Run The API

```bash
cd backend
uv run uvicorn app.main:app --reload
```

Useful local URLs:

- Swagger UI: `http://localhost:8000/docs`
- Health check: `http://localhost:8000/health`

## What The Backend Does Today

- `GET /api/v1/admin/...` routes manage clinic data and slot search
- `POST /api/v1/chat` streams SSE chat responses
- `POST /api/v1/chat/invoke` returns full JSON chat responses for testing
- booking flows ask whether the patient is new or returning
- returning-patient verification starts with full name + date of birth, then phone, then strong identifiers like `MRN`, passport, driver's license, or clinic patient number
- new-patient registration collects full name, date of birth, and phone
- triage combines `symptom_specialty_map` keyword matches with `medical_knowledge` retrieval
- scheduling supports booking, rescheduling, and cancellation

## Verify The Setup

Try a few admin endpoints:

```text
GET  /health
GET  /api/v1/admin/specialties
GET  /api/v1/admin/doctors
GET  /api/v1/admin/patients
POST /api/v1/admin/patients/search
POST /api/v1/admin/patients/{patient_id}/identifiers
GET  /api/v1/admin/slots/by-specialty?specialty_id=a1000000-0000-0000-0000-000000000001
```

Try a non-streaming chat request:

```bash
curl -X POST http://localhost:8000/api/v1/chat/invoke \
  -H "Content-Type: application/json" \
  -d '{
    "message": "I need to reschedule my appointment. My name is Sarah Connor and my birthday is October 26, 1985.",
    "thread_id": "demo-thread-1"
  }'
```

## Testing

Run the main backend test suite:

```bash
cd backend
uv run pytest tests
```

Run focused workflow coverage:

```bash
cd backend
uv run pytest tests/test_workflows.py
```

Run one opt-in eval:

```bash
cd backend
RUN_EVALS=1 uv run pytest evals/test_eval_identity_ambiguity.py -s
```

Notes:

- The eval suite uses real external services and should run against a test Supabase project.
- Some eval fixtures expect nullable `eval_tag` columns on `patients` and `appointments`.

## Project Structure

```text
backend/
├── .env.example
├── pyproject.toml
├── uv.lock
├── sql/
│   ├── 001_schema.sql
│   ├── 002_seed.sql
│   ├── 003_create_doctor_with_details.sql
│   ├── 004_finalize_reschedule_appointment.sql
│   └── 005_rag.sql
└── app/
    ├── agent/
    │   ├── agents.py
    │   ├── graph.py
    │   ├── state.py
    │   ├── supervisor.py
    │   └── tools.py
    ├── api/
    │   ├── admin/
    │   └── chat/
    ├── models/
    ├── services/
    │   ├── ingest_knowledge.py
    │   ├── rag_retriever.py
    │   ├── slot_engine.py
    │   ├── test_retriever.py
    │   └── time_utils.py
    ├── config.py
    ├── main.py
    └── supabase_client.py
```

## Architecture Notes

- The outer LangGraph is compiled in `app/agent/graph.py`.
- The Supervisor routes between Intake, Triage, and Scheduling based on shared `AgentState`.
- Tool handlers in `app/agent/tools.py` talk directly to Supabase and scheduling services.
- Conversation persistence uses Supabase Postgres through `AsyncPostgresSaver`, not an in-memory checkpointer.
- The current transport is SSE for text chat. Voice/WebSocket work is planned for a later phase.
