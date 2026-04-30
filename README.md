# Medical Voice Agent

Medical Voice Agent is an in-progress medical clinic scheduling assistant. The current implementation centers on a FastAPI/Supabase backend, with a frontend dashboard and browser voice experience planned as the next layer. It uses FastAPI, Supabase/PostgreSQL, LangChain/LangGraph, Anthropic Claude, OpenAI embeddings, and deterministic safety guardrails to support patient intake, symptom triage, appointment search, booking, rescheduling, and cancellation through a conversational API.

This is an educational engineering prototype. It is not intended to provide medical advice, diagnosis, treatment, or production healthcare operations.

## Motivation

This project is a personal follow-up to a related end-to-end Medical Voice Agent project built with a team at the University of Illinois Urbana-Champaign. That earlier version used Vapi AI for the managed voice layer, plus a FastAPI backend, a Next.js staff dashboard, and Supabase/PostgreSQL.

Managed voice AI platforms are useful for shipping quickly, but they can become expensive and restrictive as call volume grows. This repo explores how much of the system can be built directly, including:

- speech-to-text
- LLM orchestration
- tool calling
- appointment scheduling
- semantic symptom triage
- backend safety guardrails
- workflow evaluation

The broader goal is to understand the engineering trade-offs between a managed voice platform and a custom pipeline with more control over cost, latency, observability, model choice, and safety boundaries.

## Current Status

The active implementation is a text-first FastAPI backend with early voice-pipeline work. A frontend is planned but not yet included in this repository.

Implemented:

- Supabase/PostgreSQL schema, seed data, stored procedures, and pgvector support.
- Admin APIs for specialties, doctors, patients, identifiers, appointments, doctor blocks, slot search, and custom auth.
- JWT access tokens plus refresh-token rotation stored in Postgres.
- A single voice-optimized LangChain agent wired through `langchain.agents.create_agent`.
- Persistent chat memory through LangGraph's `AsyncPostgresSaver`.
- Tool calling for patient lookup, patient registration, symptom triage, slot search, booking, rescheduling, cancellation, and specialty listing.
- Hybrid triage using `symptom_specialty_map` keyword matches plus pgvector-backed semantic retrieval over `medical_knowledge`.
- Input guardrails for emergencies, self-harm, medical advice requests, prompt injection, and off-topic messages.
- Output guardrails for unsafe medical advice in non-streaming responses, plus post-stream monitoring for SSE responses.
- LangSmith PII redaction when tracing is enabled.
- AssemblyAI streaming STT client and a local WAV-file smoke test.
- Unit tests for tools, scheduling, time parsing, guardrails, PII redaction, RAG retrieval, and support utilities.
- Opt-in eval files under `backend/evals`.

Not implemented yet:

- A browser or phone voice UI.
- A FastAPI voice WebSocket endpoint.
- Cartesia or other TTS streaming integration.
- Barge-in handling.
- MCP server integration.
- A frontend dashboard.

Important architecture note: older project docs and some legacy tests still refer to a Phase 4 Supervisor/Intake/Triage/Scheduling multi-agent graph. The live app currently uses the single voice-optimized agent in `backend/app/agent/graph.py`, with the same scheduling and triage tool surface preserved behind it.

## What It Does

The backend conversational workflow can:

- Ask whether a booking patient is new or returning.
- Look up returning patients by full name and date of birth.
- Use phone number as an ambiguity resolver.
- Fall back to stronger identifiers such as MRN, passport number, driver's license number, or clinic patient number.
- Register new patients with name, date of birth, phone, and optional email.
- Match symptoms to specialties with keyword and semantic retrieval.
- Detect red-flag emergency or self-harm messages before the LLM sees them.
- Refuse medical advice and keep the conversation scoped to scheduling.
- Find available appointment slots by specialty or doctor.
- Book, reschedule, and cancel appointments.
- Stream chat responses over Server-Sent Events.
- Persist conversation history by `thread_id`.
- Protect admin APIs with bearer-token auth.

## Architecture

```text
Patient text message
        |
        v
FastAPI chat endpoint
        |
        v
Input guardrails: emergency, self-harm, advice, prompt injection, off-topic
        |
        v
Single voice-optimized LangChain agent
        |
        v
Agent tools
        |
        v
Supabase/PostgreSQL
        |
        v
Patients, doctors, appointments, slots, symptom mappings, RAG knowledge
        |
        v
Output guardrails and response streaming
```

The agent decides what should happen next in the conversation. Backend tools perform concrete operations such as identity lookup, triage, slot computation, appointment mutation, and specialty listing.

## Project Layout

```text
.
├── backend/
│   ├── app/
│   │   ├── agent/
│   │   │   ├── graph.py          # Active single-agent orchestration
│   │   │   ├── guardrails.py     # Input/output safety checks
│   │   │   ├── pii_redactor.py   # LangSmith trace redaction
│   │   │   ├── tools.py          # LangChain tools backed by services/Supabase
│   │   │   └── voice_prompt.py   # Voice-optimized system prompt
│   │   ├── api/
│   │   │   ├── main.py           # API router assembly
│   │   │   └── routes/
│   │   │       ├── admin/        # Auth and admin resource APIs
│   │   │       └── chat/         # SSE and non-streaming chat APIs
│   │   ├── core/                 # Settings and JWT/password helpers
│   │   ├── db/sql/               # Supabase SQL setup files
│   │   ├── models/               # Pydantic models and typed DB rows
│   │   ├── services/             # Slot engine, time utils, RAG, auth, ingestion
│   │   ├── voice/                # AssemblyAI STT client and WAV smoke test
│   │   ├── main.py               # FastAPI application entry point
│   │   └── supabase_client.py
│   ├── evals/                    # Opt-in eval scenarios
│   ├── tests/                    # Unit and workflow-oriented tests
│   ├── .env.example
│   ├── pyproject.toml
│   └── uv.lock
├── docs/                         # Planning docs and architecture notes
├── MANUAL_TEST_PLAN.md
├── LICENSE
└── todo.md
```

## Tech Stack

- Python 3.12
- `uv` for dependency and virtual environment management
- FastAPI and Uvicorn
- Pydantic and pydantic-settings
- Supabase/PostgreSQL with pgvector
- LangChain, LangGraph, and LangSmith
- Anthropic Claude for the chat agent
- OpenAI `text-embedding-3-small` for embeddings, called through `httpx`
- AssemblyAI streaming STT for the current voice experiment
- PyJWT and `pwdlib` for custom admin auth
- pytest and pytest-asyncio

## Setup

### 1. Install Prerequisites

You need:

- Python 3.12+
- `uv`
- A Supabase project
- An Anthropic API key for chat
- An OpenAI API key for RAG ingestion and semantic retrieval
- An AssemblyAI API key only if you want to run the STT smoke test
- A LangSmith API key only if you want tracing

### 2. Configure Environment Variables

```bash
cd backend
cp .env.example .env
```

Fill in `backend/.env`.

| Variable | Required | Purpose |
| --- | --- | --- |
| `SUPABASE_URL` | Yes | Supabase project URL |
| `SUPABASE_SERVICE_KEY` | Yes | Backend service-role key for Supabase access |
| `SUPABASE_DB_URI` | Yes for chat | Direct Postgres URI used by `AsyncPostgresSaver` |
| `FRONTEND_HOST` | No | Allowed frontend origin, defaults to `http://localhost:5173` |
| `BACKEND_CORS_ORIGINS` | No | Optional extra CORS origins as JSON list or comma-separated string |
| `ENVIRONMENT` | No | `local`, `staging`, or `production` |
| `JWT_SECRET_KEY` | Yes for auth | Must be at least 32 characters |
| `JWT_ALGORITHM` | No | Defaults to `HS256` |
| `ACCESS_TOKEN_EXPIRE_MINUTES` | No | Access-token lifetime |
| `REFRESH_TOKEN_EXPIRE_DAYS` | No | Refresh-token lifetime |
| `COOKIE_SECURE` | No | Set `true` behind HTTPS |
| `ANTHROPIC_API_KEY` | Yes for chat | Claude API key |
| `ANTHROPIC_MODEL` | No | Defaults to `claude-haiku-4-5-20251001` |
| `OPENAI_API_KEY` | Yes for RAG | Embeddings for ingestion and semantic triage |
| `LANGSMITH_API_KEY` | Optional | Enables LangSmith tracing |
| `LANGSMITH_TRACING` | Optional | Usually `true` or `false` |
| `LANGSMITH_PROJECT` | Optional | LangSmith project name |
| `TIMEZONE` | No | Clinic timezone, defaults to `America/Chicago` |
| `SCHEDULING_HORIZON_DAYS` | No | How far ahead slot search looks |
| `DEFAULT_SLOT_DURATION_MIN` | No | Fallback appointment duration |
| `ASSEMBLYAI_API_KEY` | Optional | Required only for `app.voice.test_stt` |

Keep the Supabase service-role key on the backend only. Do not expose it to a browser or public client.

### 3. Initialize Supabase

Run these SQL files in the Supabase SQL editor, in order:

1. `backend/app/db/sql/001_schema.sql`
2. `backend/app/db/sql/002_seed.sql`
3. `backend/app/db/sql/003_create_doctor_with_details.sql`
4. `backend/app/db/sql/004_finalize_reschedule_appointment.sql`
5. `backend/app/db/sql/005_rag.sql`
6. `backend/app/db/sql/006_auth.sql`

`005_rag.sql` enables pgvector, creates `medical_knowledge`, and adds the `match_medical_knowledge` RPC used by semantic retrieval.

`006_auth.sql` creates the custom `users` and `refresh_sessions` tables. New users registered through the API are not superusers by default, so superuser-only admin routes require setting `users.is_superuser = true` for the relevant account in your development database.

### 4. Install Dependencies

```bash
cd backend
uv sync
```

### 5. Ingest Medical Knowledge

Run this after `005_rag.sql` if you want semantic retrieval instead of keyword-only triage fallback:

```bash
cd backend
uv run python -m app.services.ingest_knowledge
```

### 6. Run The API

```bash
cd backend
uv run uvicorn app.main:app --reload
```

Useful local URLs:

- Swagger UI: `http://localhost:8000/docs`
- Health check: `http://localhost:8000/health`

## API Overview

All versioned endpoints are mounted under `/api/v1`.

### Chat

The chat endpoints are currently not protected by admin auth.

| Method | Endpoint | Description |
| --- | --- | --- |
| `POST` | `/api/v1/chat` | Streams an SSE response |
| `POST` | `/api/v1/chat/invoke` | Returns a complete JSON response |

Example:

```bash
curl -X POST http://localhost:8000/api/v1/chat/invoke \
  -H "Content-Type: application/json" \
  -d '{
    "message": "I need to book an appointment for recurring headaches.",
    "thread_id": "demo-thread-1"
  }'
```

### Admin Auth

| Method | Endpoint | Description |
| --- | --- | --- |
| `POST` | `/api/v1/admin/auth/register` | Create an admin user and return an access token |
| `POST` | `/api/v1/admin/auth/login` | Login and return an access token |
| `POST` | `/api/v1/admin/auth/refresh` | Rotate refresh token and return a new access token |
| `POST` | `/api/v1/admin/auth/logout` | Revoke the refresh token |
| `GET` | `/api/v1/admin/auth/me` | Return the current user |

Register or login, then pass the returned access token to protected admin routes:

```bash
curl http://localhost:8000/api/v1/admin/specialties \
  -H "Authorization: Bearer $ACCESS_TOKEN"
```

### Admin Resources

These routes are mounted under `/api/v1/admin`. `specialties` require an authenticated user. Doctors, patients, appointments, blocks, and slot search require a superuser.

| Method | Endpoint | Description |
| --- | --- | --- |
| `GET` | `/api/v1/admin/specialties` | List specialties |
| `GET` | `/api/v1/admin/specialties/{specialty_id}` | Get one specialty |
| `GET` | `/api/v1/admin/doctors` | List doctors, optionally filtered by specialty |
| `GET` | `/api/v1/admin/doctors/{doctor_id}` | Get one doctor with specialties and availability |
| `POST` | `/api/v1/admin/doctors` | Create a doctor with specialties and availability |
| `GET` | `/api/v1/admin/patients` | List patients |
| `POST` | `/api/v1/admin/patients/search` | Search patients by demographics |
| `GET` | `/api/v1/admin/patients/{patient_id}` | Get one patient |
| `POST` | `/api/v1/admin/patients` | Register a patient |
| `POST` | `/api/v1/admin/patients/{patient_id}/identifiers` | Add a patient identifier |
| `GET` | `/api/v1/admin/appointments` | List appointments with optional filters |
| `GET` | `/api/v1/admin/appointments/{appointment_id}` | Get one appointment |
| `GET` | `/api/v1/admin/blocks` | List doctor time-off blocks |
| `POST` | `/api/v1/admin/blocks` | Create a doctor time-off block |
| `GET` | `/api/v1/admin/slots/by-specialty` | Find slots across doctors in a specialty |
| `GET` | `/api/v1/admin/slots/by-doctor` | Find slots for a specific doctor |

Slot search example:

```bash
curl "http://localhost:8000/api/v1/admin/slots/by-specialty?specialty_id=a1000000-0000-0000-0000-000000000001&preferred_day=next%20monday&preferred_time=morning" \
  -H "Authorization: Bearer $ACCESS_TOKEN"
```

## Voice Work

The current voice implementation is an AssemblyAI streaming STT client plus a standalone test script. It does not yet expose a production voice endpoint.

To stream a WAV file to AssemblyAI:

```bash
cd backend
uv run python -m app.voice.test_stt test.wav
```

Expected audio format for the STT client is 16 kHz, mono, 16-bit PCM. The script can also read WAV files and feed them in paced chunks.

## Testing

The current useful local checks are focused unit tests:

```bash
cd backend
uv run pytest tests/test_time_utils.py tests/test_time_utils_extended.py
uv run pytest tests/test_slot_engine.py tests/test_slot_engine_extended.py
uv run pytest tests/test_tools.py
uv run pytest tests/test_guardrails.py tests/test_adversarial.py tests/test_pii_redactor.py
uv run pytest tests/test_rag_retriever.py tests/test_supabase_client.py
```

The full `tests/` directory still includes legacy workflow tests that target the older multi-agent graph test harness. `tests/test_eval_helpers.py` also has package-path assumptions that do not work from `backend/` as currently written. These should be migrated before treating `uv run pytest tests` as the canonical all-green command for the current single-agent architecture.

## Evaluation Suite

Opt-in evals live in:

```text
backend/evals
```

They can call real external services and should run against a disposable test Supabase project.

Some eval fixtures expect nullable `eval_tag` columns:

```sql
ALTER TABLE patients ADD COLUMN IF NOT EXISTS eval_tag text;
ALTER TABLE appointments ADD COLUMN IF NOT EXISTS eval_tag text;
```

Run one eval:

```bash
cd backend
RUN_EVALS=1 uv run pytest evals/test_eval_identity_ambiguity.py -s
```

Run all evals:

```bash
cd backend
RUN_EVALS=1 uv run pytest evals -s
```

HTTP contract evals, such as `evals/test_eval_admin_dob_validation.py` and `evals/test_eval_streaming_contract.py`, require the API to be running first. They default to `http://localhost:8000` and can be pointed elsewhere with `EVAL_BASE_URL`.

## Key Design Decisions

Hybrid triage combines keyword and semantic search. The `triage_symptoms` tool queries `symptom_specialty_map`, retrieves relevant `medical_knowledge` chunks through pgvector, then merges specialty scores. If semantic retrieval is unavailable, keyword matches can still keep the scheduling flow alive.

Guardrails are deterministic. Emergency detection, self-harm detection, advice refusal, prompt-injection checks, off-topic checks, and output sanitation are implemented as code-level rules rather than relying only on the LLM prompt.

The active agent is intentionally simple. The previous multi-agent design was useful for exploring Supervisor/Intake/Triage/Scheduling responsibilities, but the current code consolidates the workflow into one voice-optimized agent so Phase 6 voice work can proceed with fewer orchestration moving parts.

Conversation memory is durable. Chat endpoints pass a `thread_id` into the compiled agent, and `AsyncPostgresSaver` stores graph state in Supabase Postgres so conversations can continue across requests.

Admin auth is backend-owned. This project does not use Supabase Auth. FastAPI owns password hashing, access-token creation, refresh-token rotation, and authorization checks.

## Known Gaps

- The README now reflects the live backend, but `backend/README.md` and some planning docs may still describe older phases.
- Some legacy workflow tests still reference removed multi-agent modules such as `app.agent.state` and `app.agent.supervisor`.
- `tests/test_eval_helpers.py` currently fails from `backend/` because it imports the package through the outer `medical_voice_agent` namespace.
- The chat API is currently open; only admin resource routes use JWT auth.
- Output guardrails can rewrite non-streaming responses, but streaming SSE responses are scanned after the stream completes.
- STT exists as a standalone client and smoke test; end-to-end browser voice is still in progress.
- The project is not hardened for production healthcare use.

## License

[MIT](LICENSE)
