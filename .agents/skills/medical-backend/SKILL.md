---
name: medical-backend
description: Use when editing or reviewing the Medical Voice Agent backend, including FastAPI routes, LangChain agent flow, guardrails, scheduling, triage, Supabase services, backend tests, evals, or backend documentation.
---

# Medical Backend

Use this skill for work inside `medical_voice_agent/backend` or for repo docs that describe the backend.

## Orientation

- The active conversational implementation is the single voice-optimized agent in `app/agent/graph.py`.
- Older docs and some legacy tests may mention a Supervisor/Intake/Triage/Scheduling multi-agent graph. Treat that as historical unless the user explicitly asks to revive or migrate it.
- The project is an educational prototype for clinic scheduling. It must not provide medical advice, diagnosis, or treatment.
- Backend changes should preserve deterministic guardrails for emergencies, self-harm, medical-advice requests, prompt injection, and off-topic messages.

## Key Files

- `app/agent/graph.py`: active LangChain agent assembly.
- `app/agent/tools.py`: agent tools for patient lookup, registration, triage, slot search, booking, rescheduling, cancellation, and specialties.
- `app/agent/guardrails.py`: input and output safety checks.
- `app/agent/voice_prompt.py`: voice-optimized system prompt and conversation policy.
- `app/api/routes/chat/`: streaming and non-streaming chat endpoints.
- `app/api/routes/admin/`: protected admin APIs.
- `app/services/slot_engine.py`: slot search and scheduling logic.
- `app/services/rag_retriever.py` and `app/services/ingest_knowledge.py`: semantic medical knowledge retrieval.
- `app/db/sql/`: Supabase schema, seed data, stored procedures, pgvector, and auth setup.
- `tests/`: focused unit and workflow coverage.
- `evals/`: opt-in workflow evals that may call real services.

## Working Rules

- Prefer existing FastAPI, Pydantic, Supabase, LangChain, and pytest patterns.
- Keep service-layer business logic outside route handlers when the existing structure supports it.
- Keep patient identifiers, service-role keys, API keys, and LangSmith traces private. Do not add logs that expose PHI-like data or secrets.
- For scheduling and triage behavior, update tests or evals with the code change when practical.
- If a change touches user-facing conversation behavior, check the prompt, guardrails, tools, and tests together.
- For time logic, preserve the clinic timezone behavior. The default timezone is `America/Chicago`.

## Validation

Run commands from `medical_voice_agent/backend`.

Focused local checks from the README:

```bash
uv run pytest tests/test_time_utils.py tests/test_time_utils_extended.py
uv run pytest tests/test_slot_engine.py tests/test_slot_engine_extended.py
uv run pytest tests/test_tools.py
uv run pytest tests/test_guardrails.py tests/test_adversarial.py tests/test_pii_redactor.py
uv run pytest tests/test_rag_retriever.py tests/test_supabase_client.py
```

Lint when editing Python:

```bash
uv run ruff check .
```

Run the API locally:

```bash
uv run uvicorn app.main:app --reload
```

The full `uv run pytest tests` suite may include legacy workflow tests and package-path assumptions that are not currently canonical for the active single-agent architecture. Prefer focused tests unless the user asks for a full migration or full-suite cleanup.

Opt-in evals require `RUN_EVALS=1` and may use external services or a disposable Supabase project:

```bash
RUN_EVALS=1 uv run pytest evals/test_eval_identity_ambiguity.py -s
```
