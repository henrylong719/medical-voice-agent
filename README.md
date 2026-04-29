# Medical Voice Agent

Medical Voice Agent is a backend-first prototype of a medical clinic scheduling assistant. It combines **FastAPI**, **Supabase/PostgreSQL**, **LangGraph**, **Anthropic Claude**, and **OpenAI embeddings** to support patient intake, symptom triage, appointment availability search, and booking workflows through a conversational interface.

The active implementation lives in [`backend/`](backend/). The planning history and future-phase notes live in [`docs/`](docs/).

> This project is for learning and engineering exploration. It is not intended to provide medical advice, diagnosis, or treatment.

---

## Motivation

This project was created as a personal follow-up to a related end-to-end Medical Voice Agent project that I developed with a team at the University of Illinois.

The original project focuses on a phone-call AI assistant for a university health center appointment-booking use case. It uses **Vapi AI** for the voice layer, including speech-to-text, text-to-speech, LLM-driven conversations, and tool calling. It also includes a **FastAPI backend**, a **Next.js staff dashboard**, and **Supabase/PostgreSQL** for storing patients, appointments, call logs, and transcripts.

Vapi AI is very useful for quickly building and testing a working voice agent. However, managed voice AI platforms usually charge based on call duration, so the cost can become significant when call volume increases. Because of this, I started this project to explore whether a more custom architecture could reduce cost and provide more control over the voice-agent pipeline.

The goal of this version is to investigate how much of the system can be built directly, including:

- speech-to-text
- LLM orchestration
- text-to-speech
- tool calling
- appointment scheduling
- semantic symptom triage
- backend safety guardrails
- workflow evaluation

This project is not only about recreating the same medical voice agent. It is also an engineering exploration of the trade-offs between using a managed voice AI platform and building a more custom system.

A managed platform can help ship faster and simplify voice infrastructure, while a custom pipeline may provide more control over **cost, latency, model choice, observability, and scalability**.

Related project:

```text
https://github.com/Agentic-AI-UIUC/Agentic-Medical-Voice-Agent
```

---

## Current Status

The repository is currently in a Phase 4-style state:

- **Phase 1 foundations are in place:** schema, seed data, admin APIs, scheduling engine, and time utilities.
- **Phase 2 agent work is in place:** LangGraph agent, tool calling, patient identification, booking, rescheduling, cancellation, and chat endpoints.
- **Phase 3 hybrid triage work is in place:** pgvector-backed medical knowledge retrieval plus keyword symptom matching.
- **Phase 4 multi-agent routing is in place:** Supervisor, Intake, Triage, and Scheduling agents.

---

## Upcoming Milestones

- **Phase 5:** Emergency symptoms trigger an immediate escalation response, and the assistant refuses medical advice while staying in scheduling mode.
- **Phase 6:** A patient can complete an end-to-end appointment workflow entirely by voice through the browser.
- **Phase 7:** The project has measurable eval coverage for quality and safety, with regression checks for prompt changes.
- **Phase 8:** The medical tools are available through an MCP server and usable from an MCP-compatible client.

---

## What The Project Does Today

The current implementation supports a backend conversational workflow for medical appointment scheduling.

It can:

- Ask booking patients whether they are new or returning.
- Look up returning patients by full name and date of birth first.
- Use phone number as an optional disambiguator if demographic lookup is ambiguous.
- Fall back to stronger identifiers like MRN, passport number, driver's license number, or clinic patient number if demographics still do not resolve one record.
- Register new patients with full name, date of birth, and phone number.
- Match symptoms to specialties with hybrid triage using keyword search over `symptom_specialty_map` plus semantic search over `medical_knowledge`.
- Find open appointment slots across doctors or for a specific doctor.
- Book, reschedule, and cancel appointments.
- Stream chat responses over Server-Sent Events.
- Persist conversation state in Postgres using LangGraph's `AsyncPostgresSaver`.
- Expose admin APIs for specialties, doctors, patients, patient identifiers, appointments, blocks, and slots.
- Guide ambiguous identity cases toward staff help, although a dedicated human-handoff implementation is not yet complete.

---

## Architecture At A Glance

At a high level, the system works like this:

```text
User message or future voice input
        ↓
FastAPI chat endpoint
        ↓
LangGraph Supervisor Agent
        ↓
Intake Agent / Triage Agent / Scheduling Agent
        ↓
Tool calls
        ↓
Supabase/PostgreSQL
        ↓
Patient, doctor, appointment, and triage data
```

The project separates the AI conversation layer from backend business logic.

The agent decides what step should happen next, while backend tools handle concrete actions such as:

- patient lookup
- patient registration
- symptom triage
- slot search
- appointment booking
- rescheduling
- cancellation

---

## Multi-Agent Workflow

The workflow is designed around multiple agent responsibilities.

### Supervisor Agent

Routes the conversation to the correct sub-agent based on the user's intent.

Example intents include:

- new patient intake
- returning patient lookup
- symptom triage
- booking
- rescheduling
- cancellation

### Intake Agent

Handles patient identification and registration.

For returning patients, the system starts with demographic information such as:

- full name
- date of birth

If the result is ambiguous, it can ask for additional identifiers such as phone number or patient ID.

### Triage Agent

Handles symptom-to-specialty matching.

The triage system combines:

- keyword-based symptom matching
- semantic search using OpenAI embeddings
- pgvector retrieval from medical knowledge chunks

This makes the system more flexible than simple keyword matching.

For example, instead of only matching the exact keyword:

```text
headache
```

the system can better handle descriptions such as:

```text
my head hurts
I feel pressure around my forehead
I have pain behind my eyes
```

### Scheduling Agent

Handles appointment-related workflows.

It can:

- search available slots
- book an appointment
- reschedule an appointment
- cancel an appointment

The scheduling logic uses doctor availability, booked appointments, and doctor blocks to compute available slots.

---

## Hybrid Triage

The triage system is one of the key parts of this project.

In an early version, symptom matching was based mainly on keywords. This worked for simple examples, but it was not practical enough because real patients may describe symptoms in many different ways.

To improve this, the project uses hybrid triage:

```text
Keyword matching
+ Semantic search
+ Specialty scoring
= Better specialty recommendation
```

### Keyword Matching

The system checks structured symptom-to-specialty mappings.

For example:

```text
headache → Neurology
skin rash → Dermatology
```

### Semantic Search

The system also embeds the patient's symptom description and compares it with medical knowledge chunks stored in Postgres using pgvector.

This allows the system to understand meaning, not just exact words.

### Specialty Scoring

Possible specialties receive scores based on keyword and semantic matches.

If one specialty has a clear confidence score, the system can recommend it.

If multiple specialties are close, the assistant can ask follow-up questions instead of guessing too early.

---

## Safety Guardrails

Because this project is related to healthcare, the assistant should not treat every conversation as a normal appointment-booking flow.

Some symptoms may require urgent help.

The project is designed to support emergency guardrails before normal triage.

Examples of red-flag symptoms include:

- chest pain
- trouble breathing
- stroke-like symptoms
- severe bleeding
- severe allergic reaction
- mental health crisis

The intended safety flow is:

```text
Patient describes symptoms
        ↓
Check emergency red flags
        ↓
If urgent: stop normal booking flow and advise emergency help
        ↓
If not urgent: continue normal triage and scheduling
```

This reflects an important design principle:

> Normal triage can use semantic search and scoring, but emergency cases need stricter rule-based guardrails.

---

## Project Layout

```text
.
├── backend/
│   ├── app/
│   │   ├── agent/          # System prompt, tools, and LangGraph agent wiring
│   │   ├── api/            # Admin and chat FastAPI routes
│   │   ├── models/         # Pydantic models and typed DB row shapes
│   │   ├── services/       # Scheduling, time parsing, RAG retrieval, ingestion
│   │   ├── config.py       # Environment-driven settings
│   │   ├── main.py         # FastAPI entry point
│   │   └── supabase_client.py
│   ├── sql/
│   │   ├── 001_schema.sql
│   │   ├── 002_seed.sql
│   │   ├── 003_create_doctor_with_details.sql
│   │   ├── 004_finalize_reschedule_appointment.sql
│   │   └── 005_rag.sql
│   ├── .env.example
│   ├── pyproject.toml
│   └── uv.lock
├── docs/                   # Phase notes, planning prompts, flow diagram
├── LICENSE
└── todo.md
```

---

## Tech Stack

### Backend

- Python 3.12
- `uv` for environment and dependency management
- FastAPI
- Uvicorn
- Pydantic

### Database

- Supabase
- PostgreSQL
- pgvector

### AI / Agent Layer

- LangChain
- LangGraph
- Anthropic Claude
- OpenAI embeddings
- LangSmith tracing

### Testing / Evaluation

- pytest
- workflow tests
- evaluation notes for quality and safety checks

---

## Setup

### 1. Prerequisites

You need:

- Python 3.12+
- [`uv`](https://docs.astral.sh/uv/)
- A Supabase project
- An Anthropic API key
- An OpenAI API key for embedding ingestion and semantic retrieval
- A LangSmith key if you want tracing

---

### 2. Configure Environment Variables

```bash
cd backend
cp .env.example .env
```

Fill in the important values in `backend/.env`:

| Variable                    | Required            | Purpose                                          |
| --------------------------- | ------------------- | ------------------------------------------------ |
| `SUPABASE_URL`              | Yes                 | Supabase project URL                             |
| `SUPABASE_SERVICE_KEY`      | Yes                 | Backend service-role key                         |
| `SUPABASE_DB_URI`           | Yes for chat        | Direct Postgres URI used by `AsyncPostgresSaver` |
| `ANTHROPIC_API_KEY`         | Yes for chat        | Claude API key                                   |
| `ANTHROPIC_MODEL`           | No                  | Defaults to `claude-haiku-4-5-20251001`          |
| `OPENAI_API_KEY`            | Yes for Phase 3 RAG | Embeddings for ingestion and semantic triage     |
| `LANGSMITH_API_KEY`         | Optional            | LangSmith tracing                                |
| `LANGSMITH_TRACING`         | Optional            | Usually `true`                                   |
| `LANGSMITH_PROJECT`         | Optional            | Trace grouping/project name                      |
| `TIMEZONE`                  | No                  | Clinic timezone, defaults to `America/Chicago`   |
| `SCHEDULING_HORIZON_DAYS`   | No                  | How far ahead to search for slots                |
| `DEFAULT_SLOT_DURATION_MIN` | No                  | Fallback slot duration                           |

Important:

```text
Keep the Supabase service-role key local to the backend only.
Do not expose it to a browser client, mobile app, or public frontend.
```

---

### 3. Initialize Supabase

Run these SQL files in your Supabase SQL editor, in this order:

1. `backend/sql/001_schema.sql`
2. `backend/sql/002_seed.sql`
3. `backend/sql/003_create_doctor_with_details.sql`
4. `backend/sql/004_finalize_reschedule_appointment.sql`
5. `backend/sql/005_rag.sql`

Notes:

- The first four files set up the clinic data model, sample data, doctor-creation RPC, and reschedule-finalization RPC.
- `005_rag.sql` enables `pgvector`, creates `medical_knowledge`, and adds the `match_medical_knowledge` RPC used by hybrid triage.
- If you reset the `public` schema in Supabase, make sure `service_role` privileges are restored before running ingestion scripts.

You can re-run the following SQL if needed:

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

If `python -m app.services.ingest_knowledge` fails with `permission denied for table medical_knowledge`, this is the first thing to check. Also confirm that `SUPABASE_SERVICE_KEY` in `backend/.env` is the service-role key, not the anon key.

---

### 4. Install Dependencies

```bash
cd backend
uv sync
```

---

### 5. Ingest Medical Knowledge Chunks

This populates the `medical_knowledge` table with embedded symptom-cluster passages used by semantic triage.

```bash
cd backend
uv run python -m app.services.ingest_knowledge
```

---

### 6. Run The API

```bash
cd backend
uv run uvicorn app.main:app --reload
```

Open:

- Swagger UI: `http://localhost:8000/docs`
- Health check: `http://localhost:8000/health`

---

## API Overview

### Admin Endpoints

Mounted under `/api/v1/admin`:

- `GET /specialties`
- `GET /specialties/{specialty_id}`
- `GET /doctors`
- `GET /doctors/{doctor_id}`
- `POST /doctors`
- `GET /patients`
- `POST /patients/search`
- `GET /patients/{patient_id}`
- `POST /patients`
- `POST /patients/{patient_id}/identifiers`
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

---

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
    "message": "I need to reschedule my appointment. My name is Sarah Connor and my birthday is October 26, 1985.",
    "thread_id": "demo-thread-1"
  }'
```

### Slot Search

```bash
curl "http://localhost:8000/api/v1/admin/slots/by-specialty?specialty_id=a1000000-0000-0000-0000-000000000001&preferred_day=next%20monday&preferred_time=morning"
```

---

## Key Implementation Notes

- Patient identity is a first-class part of the workflow. For booking, the agent asks whether the patient is new or returning, then starts returning-patient lookup with full name and date of birth.
- If demographic lookup is ambiguous, the agent asks for phone number next and only then falls back to stronger identifiers like MRN, passport number, driver's license number, or clinic patient number.
- If identity remains ambiguous, the current implementation guides the conversation toward staff help, but a dedicated human-handoff mechanism has not been implemented yet.
- Scheduling is computed from recurring availability templates, then filtered by booked appointments and doctor time-off blocks.
- RAG retrieval uses OpenAI embeddings and a Supabase RPC rather than embedding logic inside the agent tool layer.
- If semantic retrieval is unavailable, the triage tool falls back to keyword results instead of crashing the whole flow.
- The seed data includes specialties, doctors, symptom mappings, patients, patient identifiers, and appointment data for development.

---

## Validation And Smoke Tests

Interactive API docs are available at:

```text
http://localhost:8000/docs
```

For automated multi-turn workflow tests that exercise the real Supervisor and LangGraph routing without Postman:

```bash
cd backend
uv run python -m pytest tests/test_workflows.py
```

These tests use an in-memory checkpointer plus scripted sub-agent doubles, so they cover conversation flow and state transitions without calling external LLM APIs.

---

## Evaluation Suite

The opt-in eval suite lives under:

```text
backend/evals
```

Notes:

- `uv sync` is enough for the Python dependencies used by the eval suite.
- The eval fixtures currently tag seeded rows for cleanup, so your test Supabase project needs a nullable `eval_tag` column on both `patients` and `appointments`.
- Most evals call the real agent graph directly and use real external services, so they can cost money and should be run against a test Supabase project, not production.

Add the eval-only columns once in your test database:

```sql
ALTER TABLE patients ADD COLUMN IF NOT EXISTS eval_tag text;
ALTER TABLE appointments ADD COLUMN IF NOT EXISTS eval_tag text;
```

Run one eval and print the safety and quality report to stdout:

```bash
cd backend
RUN_EVALS=1 uv run pytest evals/test_eval_identity_ambiguity.py -s
```

The `-s` flag is important because it prints pass rates and failed transcripts.

Run the full eval suite:

```bash
cd backend
RUN_EVALS=1 uv run pytest evals -s
```

If you include the HTTP contract evals `evals/test_eval_admin_dob_validation.py` or `evals/test_eval_streaming_contract.py`, start the API first:

```bash
cd backend
uv run uvicorn app.main:app --reload
```

Those tests default to `http://localhost:8000`, or you can point them at a different server with `EVAL_BASE_URL`.

LR-11, RAG degradation, is not covered by the automated eval suite. It requires removing `OPENAI_API_KEY` at runtime to force semantic retrieval failure, which is environment-destructive and better tested manually in a dedicated test setup.

For the RAG retriever smoke test:

```bash
cd backend
uv run python -m app.services.test_retriever
```

This script expects:

- `OPENAI_API_KEY` to be configured
- `backend/sql/005_rag.sql` to have been applied
- `medical_knowledge` to be populated via the ingestion script

---

## Roadmap And Docs

The docs folder contains both earlier phase notes and the current long-form roadmap.

- [Current architecture and learning plan](docs/medical_voice_agent_plan_v3.md)
- [Phase 1 notes](<docs/Medical Voice Agent — Project Instruction Prompt_phase1.md>)
- [Combined phase status notes](<docs/Medical Voice Agent — Project Instruction Prompt_phases.md>)
- [Original project planning prompt](<docs/Medical Voice Agent — Project Instruction Prompt.md>)
- [Long-form working notes](docs/record.md)

Based on the current implementation, Phases 1 through 4 are in place. The remaining roadmap is:

| Phase                                            | Focus                                                        | Duration  | Goal                                                         |
| ------------------------------------------------ | ------------------------------------------------------------ | --------- | ------------------------------------------------------------ |
| **5. Guardrails & Medical Safety**               | Safety boundaries, emergency detection, scope control, PII hygiene | 1-2 weeks | Add input/output guardrails so the assistant stays in scheduling scope, catches red-flag symptoms, and handles sensitive data more safely. |
| **6. Real-Time Voice Pipeline**                  | Streaming STT/TTS, WebSockets, barge-in, spoken UX           | 2-4 weeks | Turn the text-based backend into a realtime voice experience with AssemblyAI STT, Cartesia TTS, and a FastAPI WebSocket pipeline. |
| **7. Evaluation, Testing & Prompt Optimization** | LangSmith datasets, automated scoring, regression coverage   | 2-3 weeks | Build a repeatable eval system for triage accuracy, safety, end-to-end flows, and prompt iteration. |
| **8. MCP Integration**                           | MCP server, tools/resources/prompts, stdio + SSE transports  | 1-2 weeks | Expose scheduling and triage capabilities through a standards-compliant MCP server for Claude Desktop and other MCP clients. |

---

## Why This Project Matters

This project helped me explore the difference between building a simple chatbot and building an AI workflow connected to real backend logic.

A chatbot can answer questions, but a medical scheduling assistant needs to:

- identify the patient
- understand the reason for visit
- check symptoms safely
- choose the correct next step
- call backend tools
- update the database
- handle ambiguous cases
- avoid unsafe medical advice

The main learning from this project is that practical AI systems need more than a good prompt. They need reliable backend tools, structured workflows, safety guardrails, database constraints, observability, and testing.

---

## Disclaimer

This project is for educational and prototype purposes only.

It does not provide medical diagnosis, treatment, or professional medical advice. Any real healthcare deployment would require clinical review, privacy controls, compliance checks, security hardening, and human oversight.

---

## License

[MIT](LICENSE)
