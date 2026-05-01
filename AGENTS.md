# Medical Voice Agent Repo Guidance

This is an educational medical clinic scheduling prototype. Do not add behavior that provides medical advice, diagnosis, or treatment.

The active backend conversation flow is the single voice-optimized LangChain agent in `backend/app/agent/graph.py`. Older Supervisor/Intake/Triage/Scheduling multi-agent references are legacy unless explicitly requested.

For backend work, use existing FastAPI, Pydantic, Supabase, LangChain, and pytest patterns. Run backend commands from `backend`.

Protect secrets and PHI-like data. Do not log API keys, Supabase service keys, patient identifiers, or raw sensitive conversation content.

Prefer focused tests from the README or the `$medical-backend` skill. The full backend test suite includes some legacy assumptions and should not be treated as canonical all-green without migration work.

Use `$medical-backend` for backend, guardrail, triage, scheduling, Supabase, test, or eval work.
