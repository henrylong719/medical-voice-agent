# Medical Voice Agent

An AI-powered voice agent that lets patients call a medical clinic, describe symptoms, get triaged to the right specialist, and book appointments — entirely through natural voice conversation.

Built from scratch as a self-study project covering RAG, agent orchestration, real-time voice pipelines, guardrails, evaluation, and MCP.

## Current Status

Phases 1 through 5 are complete. Phase 6 (voice pipeline) is in progress.

At the start of Phase 6, the multi-agent system (supervisor + 3 sub-agents) was simplified to a single voice-optimized agent. The multi-agent code is preserved on disk but not active. This lets voice development proceed without multi-agent routing complexity — the two systems solve orthogonal problems and will be recombined later.

| Phase | Focus | Status |
|---|---|---|
| 1. Database & Backend | Schema, slot engine, time utils, admin API, seed data | Complete |
| 2. LangChain Agent | Tool calling, system prompt, streaming, conversation memory | Complete |
| 3. RAG Triage | pgvector, embeddings, hybrid keyword + semantic search | Complete |
| 4. Multi-Agent System | Supervisor + intake/triage/scheduling sub-agents | Complete (preserved, temporarily replaced by single agent) |
| 5. Guardrails & Safety | Input/output filtering, emergency detection, PII redaction | Complete |
| 6. Voice Pipeline | STT + TTS streaming, WebSocket, barge-in, voice UX | **In progress** |
| 7. Evals & Optimization | LangSmith datasets, automated scoring, prompt iteration | Upcoming |
| 8. MCP Integration | MCP server, tools/resources/prompts, Claude Desktop | Upcoming |

## What It Does Today

- Greets patients and determines intent (book, reschedule, cancel, or general question).
- Identifies returning patients by demographics (name + DOB), with progressive fallback to phone, then strong identifiers (MRN, passport, driver's license, clinic patient number).
- Registers new patients with name, DOB, and phone.
- Triages symptoms using hybrid search: keyword matching on `symptom_specialty_map` plus semantic retrieval over `medical_knowledge` via pgvector. Falls back gracefully if the embedding API is unavailable.
- Computes available appointment slots on the fly from doctor weekly availability templates, filtering out booked slots and time-off blocks.
- Books, reschedules, and cancels appointments with explicit patient confirmation at each step.
- Detects emergency symptoms (cardiac, stroke, anaphylaxis, etc.) and immediately directs patients to call 911.
- Blocks prompt injection attempts, medical advice requests, and off-topic queries before they reach the agent.
- Scans agent responses for medical advice violations and rewrites them if detected.
- Redacts PII (names, DOBs, phone numbers, identifiers) from LangSmith traces automatically.
- Streams chat responses over Server-Sent Events.
- Persists conversation state in Postgres via LangGraph's `AsyncPostgresSaver`.
- Streams audio to AssemblyAI for real-time speech-to-text (STT client built and tested).

## Architecture

```
Browser / Client
    │
    ├── POST /api/v1/chat         (text: SSE streaming)
    ├── POST /api/v1/chat/invoke   (text: full JSON response)
    └── WS /ws/voice               (voice: bidirectional audio — Phase 6, to build)
          │
          ▼
    ┌─────────────────────────────────────┐
    │  FastAPI                            │
    │                                     │
    │  screen_input()  ← input guardrails │
    │       │                             │
    │       ▼                             │
    │  Single Agent (Claude Haiku 4.5)    │
    │  ┌──────────────────────────────┐   │
    │  │ 10 tools:                    │   │
    │  │  find_patients_by_demo...    │   │
    │  │  find_patient_by_identifier  │   │
    │  │  register_patient            │   │
    │  │  triage_symptoms (RAG)       │   │
    │  │  find_slots                  │   │
    │  │  book_appointment            │   │
    │  │  find_appointment            │   │
    │  │  reschedule_appointment      │   │
    │  │  cancel_appointment          │   │
    │  │  list_specialties            │   │
    │  └──────────────────────────────┘   │
    │       │                             │
    │       ▼                             │
    │  sanitize_output() ← output guard.  │
    └─────────┬───────────────────────────┘
              │
              ▼
    ┌─────────────────────┐    ┌──────────────┐
    │  Supabase Postgres  │    │  pgvector     │
    │  9 tables + RPCs    │    │  medical_     │
    │  + checkpointer     │    │  knowledge    │
    └─────────────────────┘    └──────────────┘
```

For the voice pipeline (Phase 6, in progress):

```
Browser Mic → WebSocket → AssemblyAI STT → Transcript → Agent → Text → Cartesia TTS → Audio → WebSocket → Browser Speaker
```

## Project Layout

```
backend/
├── app/
│   ├── agent/
│   │   ├── voice_prompt.py       # Active: voice-optimized single-agent prompt
│   │   ├── graph.py              # Agent graph + public API (stream_agent_response, invoke_agent)
│   │   ├── tools.py              # 10 tools with Pydantic schemas
│   │   ├── guardrails.py         # Input + output guardrails (~900 lines)
│   │   ├── pii_redactor.py       # PII masking for LangSmith traces
│   │   ├── supervisor.py         # Preserved: Phase 4 supervisor routing
│   │   ├── agents.py             # Preserved: Phase 4 sub-agent definitions
│   │   └── state.py              # Preserved: Phase 4 multi-agent state
│   ├── voice/
│   │   ├── stt_client.py         # AssemblyAI v3 streaming (built, tested)
│   │   └── test_stt.py           # Standalone STT test script
│   ├── api/
│   │   ├── admin/                # CRUD endpoints for all domain entities
│   │   └── chat/routes.py        # SSE streaming + invoke endpoints
│   ├── services/
│   │   ├── slot_engine.py        # On-the-fly availability computation
│   │   ├── time_utils.py         # NLP date parsing, timezone handling
│   │   ├── rag_retriever.py      # Embedding + vector similarity search
│   │   ├── knowledge_chunks.py   # Medical knowledge content
│   │   └── ingest_knowledge.py   # Batch embed + insert into pgvector
│   ├── models/                   # Pydantic models + TypedDicts
│   ├── config.py                 # Pydantic settings from .env
│   ├── main.py                   # FastAPI app entry point
│   └── supabase_client.py        # Supabase client singleton
├── tests/                        # ~600 tests across 20+ test files
├── evals/                        # 13 end-to-end eval scripts
├── sql/
│   ├── 001_schema.sql            # 9 tables with constraints and indexes
│   ├── 002_seed.sql              # Specialties, doctors, patients, mappings
│   ├── 003_create_doctor_with_details.sql  # Transactional RPC
│   ├── 004_finalize_reschedule_appointment.sql
│   └── 005_rag.sql               # pgvector extension + medical_knowledge table
└── pyproject.toml
```

## Tech Stack

| Layer | Technology | Purpose |
|---|---|---|
| Language | Python 3.12+ | All backend code |
| Backend | FastAPI + uvicorn | REST API, WebSocket, validation |
| Database | Supabase (PostgreSQL) | Relational data, RPC functions |
| Vector Store | pgvector (in Supabase) | Embeddings for RAG triage |
| LLM | Claude Haiku 4.5 | Agent reasoning and tool calling |
| Agent Framework | LangChain + LangGraph | Tool binding, agent graph, checkpointer |
| Observability | LangSmith | Tracing, evals, prompt debugging |
| Embeddings | OpenAI text-embedding-3-small | Symptom text → vectors for RAG |
| STT | AssemblyAI Streaming v3 | Real-time speech-to-text |
| TTS | Cartesia Sonic 3 | Real-time text-to-speech (Phase 6) |
| Package Manager | uv | Fast Python dependency management |

## Setup

### Prerequisites

- Python 3.12+
- [uv](https://docs.astral.sh/uv/)
- A Supabase project
- An Anthropic API key
- An OpenAI API key (for embeddings)
- An AssemblyAI API key (for voice, Phase 6)
- A Cartesia API key (for voice, Phase 6)
- A LangSmith API key (optional, for tracing)

### 1. Configure Environment

```bash
cd backend
cp .env.example .env
```

Fill in `backend/.env`:

| Variable | Required | Purpose |
|---|---|---|
| `SUPABASE_URL` | Yes | Supabase project URL |
| `SUPABASE_SERVICE_KEY` | Yes | Backend service-role key (never expose to clients) |
| `SUPABASE_DB_URI` | Yes for chat | Direct Postgres URI for `AsyncPostgresSaver` |
| `ANTHROPIC_API_KEY` | Yes for chat | Claude API key |
| `ANTHROPIC_MODEL` | No | Defaults to `claude-haiku-4-5-20251001` |
| `OPENAI_API_KEY` | Yes for RAG | Embeddings for ingestion and semantic triage |
| `ASSEMBLYAI_API_KEY` | Phase 6 | Streaming speech-to-text |
| `CARTESIA_API_KEY` | Phase 6 | Streaming text-to-speech |
| `LANGSMITH_API_KEY` | Optional | LangSmith tracing |
| `LANGSMITH_TRACING` | Optional | Usually `true` |
| `LANGSMITH_PROJECT` | Optional | Trace grouping name |
| `TIMEZONE` | No | Clinic timezone, defaults to `America/Chicago` |
| `SCHEDULING_HORIZON_DAYS` | No | How far ahead to search for slots |
| `DEFAULT_SLOT_DURATION_MIN` | No | Fallback slot length in minutes |

### 2. Initialize Database

Run these SQL files in your Supabase SQL editor, in order:

1. `backend/sql/001_schema.sql`
2. `backend/sql/002_seed.sql`
3. `backend/sql/003_create_doctor_with_details.sql`
4. `backend/sql/004_finalize_reschedule_appointment.sql`
5. `backend/sql/005_rag.sql`

If you reset the `public` schema, restore `service_role` privileges:

```sql
grant usage on schema public to service_role;
grant all privileges on all tables in schema public to service_role;
grant all privileges on all sequences in schema public to service_role;
grant all privileges on all routines in schema public to service_role;

alter default privileges in schema public
  grant all privileges on tables to service_role;
alter default privileges in schema public
  grant all privileges on sequences to service_role;
alter default privileges in schema public
  grant all privileges on routines to service_role;
```

### 3. Install Dependencies

```bash
cd backend
uv sync
```

### 4. Ingest Medical Knowledge

Populates the `medical_knowledge` table with embedded symptom-cluster passages for RAG triage:

```bash
cd backend
uv run python -m app.services.ingest_knowledge
```

### 5. Run the API

```bash
cd backend
uv run uvicorn app.main:app --reload
```

- Swagger UI: http://localhost:8000/docs
- Health check: http://localhost:8000/health

## API Endpoints

### Admin (under `/api/v1/admin`)

| Method | Endpoint | Description |
|---|---|---|
| GET | `/specialties` | List all specialties |
| GET | `/specialties/{id}` | Get a specialty |
| GET | `/doctors` | List all doctors |
| GET | `/doctors/{id}` | Get a doctor with schedule |
| POST | `/doctors` | Create a doctor with availability |
| GET | `/patients` | List patients |
| POST | `/patients/search` | Search by demographics |
| GET | `/patients/{id}` | Get a patient |
| POST | `/patients` | Register a patient |
| POST | `/patients/{id}/identifiers` | Attach a strong identifier |
| GET | `/appointments` | List appointments |
| GET | `/appointments/{id}` | Get an appointment |
| GET | `/blocks` | List doctor time-off blocks |
| POST | `/blocks` | Create a time-off block |
| GET | `/slots/by-specialty` | Find available slots by specialty |
| GET | `/slots/by-doctor` | Find available slots by doctor |

### Chat (under `/api/v1`)

| Method | Endpoint | Description |
|---|---|---|
| POST | `/chat` | Streaming SSE response |
| POST | `/chat/invoke` | Full JSON response (easier for testing) |

## Example Requests

### Streaming Chat

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
    "message": "I need to reschedule my appointment. My name is Sarah Connor and my birthday is October 26, 1985.",
    "thread_id": "demo-thread-1"
  }'
```

### Slot Search

```bash
curl "http://localhost:8000/api/v1/admin/slots/by-specialty?specialty_id=a1000000-0000-0000-0000-000000000001&preferred_day=next%20monday&preferred_time=morning"
```

### STT Test

```bash
# Generate a test WAV (macOS):
say -o test.wav --data-format=LEI16@16000 "I have a headache and my vision is blurry"

# Stream it to AssemblyAI:
cd backend
uv run python -m app.voice.test_stt test.wav
```

## Testing

### Unit and Integration Tests

```bash
cd backend
uv run python -m pytest tests/
```

The test suite includes ~600 tests covering guardrails (172 tests), adversarial inputs (86 tests), PII redaction (40 tests), supervisor routing, sub-agent behavior, graph structure, end-to-end conversation workflows, tools, slot engine, time utils, and RAG retrieval.

### End-to-End Evals

The `evals/` directory contains 13 eval scripts that test complete multi-turn conversations against the real agent graph. These call external LLM APIs and cost money — run against a test Supabase project.

Before running evals, add the eval-only columns to your test database:

```sql
ALTER TABLE patients ADD COLUMN IF NOT EXISTS eval_tag text;
ALTER TABLE appointments ADD COLUMN IF NOT EXISTS eval_tag text;
```

Run a single eval:

```bash
cd backend
RUN_EVALS=1 uv run pytest evals/test_eval_identity_ambiguity.py -s
```

Run the full eval suite:

```bash
cd backend
RUN_EVALS=1 uv run pytest evals -s
```

The `-s` flag prints pass rates and failed transcripts.

For HTTP contract evals (`test_eval_admin_dob_validation.py`, `test_eval_streaming_contract.py`), start the API first. They default to `http://localhost:8000` or use `EVAL_BASE_URL`.

### RAG Retriever Smoke Test

```bash
cd backend
uv run python -m app.services.test_retriever
```

Requires `OPENAI_API_KEY`, applied `005_rag.sql`, and ingested knowledge chunks.

## Key Design Decisions

**Slot computation is on-the-fly.** Doctors define weekly availability templates, and the slot engine generates concrete slots for a requested date range, subtracting booked appointments and time-off blocks. No cron jobs or pre-generated slot tables.

**Hybrid triage combines keyword and semantic search.** The `triage_symptoms` tool runs keyword matching on `symptom_specialty_map` and vector similarity search on `medical_knowledge` in parallel, then merges and ranks results. If the embedding API is down, keyword results still work.

**Guardrails are deterministic, not prompt-based.** Emergency detection, prompt injection blocking, and medical advice filtering use regex and rule-based classifiers that run before and after the LLM. They cannot be bypassed by clever prompting.

**PII redaction happens at the tracing layer.** Patient names, dates of birth, phone numbers, and identifiers are masked before they reach LangSmith, using a pre-initialized anonymizer on the LangSmith client singleton.

**Single-agent for voice, multi-agent preserved.** The voice pipeline uses a simplified single-agent setup so voice UX can be developed without orchestration complexity. The multi-agent system (supervisor + 3 sub-agents, ~1,700 lines) is preserved on disk and will be reactivated with voice-adapted prompts once the pipeline is solid.

## Docs

- [Architecture and learning plan](docs/medical_voice_agent_plan_v3.md) — full 8-phase roadmap with concepts and implementation details.
- [QA notes](docs/QAs.md) — questions and answers from the learning process.
- [Known issues](docs/issues.md) — tracked bugs and edge cases.

## License

[MIT](LICENSE)
