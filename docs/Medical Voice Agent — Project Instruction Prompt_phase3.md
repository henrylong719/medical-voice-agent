

---

## Current Phase

**I am currently on: Phase 3 — RAG-Powered Triage with Vector Search**

### What's Been Built So Far

- **Phase 1 complete:**
  - Database schema (9 tables) with enums, CHECK constraints, indexes in Supabase
  - Seed data: 10 specialties, 50+ symptom-specialty mappings with weights and follow-up questions, 8 doctors with varied schedules, 5 sample patients
  - RPC function for atomic doctor creation (transactional)
  - FastAPI project with clean architecture: models/, services/, api/admin/
  - TypedDicts (db_rows.py) for typed Supabase query results
  - Pydantic models for API validation (doctor, patient, block, specialty, slot)
  - Slot engine with dual entry points: find_slots_for_specialty() and find_slots_for_doctor()
  - Time utils: NLP date parsing (abbreviations, numeric dates, month names, "next available" aliases), time bucket filtering, UTC conversion, voice formatting
  - Admin REST API: CRUD for specialties, doctors, patients, appointments, blocks, slots
  - All endpoints tested and working

- **Phase 2 complete:**
  - 9 agent tools with Pydantic input schemas: identify_patient, register_patient, triage_symptoms, find_slots, book_appointment, find_appointment, reschedule_appointment, cancel_appointment, list_specialties
  - System prompt with mandatory UIN-first workflow, response style rules, and hard safety boundaries
  - Agent graph using create_agent (LangChain v1.0 API) with Claude Haiku 4.5
  - Persistent conversation memory via AsyncPostgresSaver (Supabase Postgres), with AsyncExitStack lifecycle management and clean shutdown via FastAPI lifespan
  - Chat API: POST /api/v1/chat (SSE streaming) and POST /api/v1/chat/invoke (full response)
  - db_rows.py expanded with 9 new TypedDicts for tool query results, shared nested fragments (NestedDoctorName, NestedSpecialtyName)
  - LangSmith tracing configured for observability
  - Full booking flow tested end-to-end: identify → triage → follow-up questions → slot search → book

### What I'm Working On Now

- Starting Phase 3: RAG-powered triage with vector search

### Current Challenges

- Triage uses keyword matching (ilike) — brittle for natural language symptom descriptions
- Agent sometimes picks Cardiology over Neurology when scores are close (keyword limitation)
- Agent doesn't always ask for explicit booking confirmation before booking (prompt tuning needed)

