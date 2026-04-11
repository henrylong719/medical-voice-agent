

# Phase 2

---

## Current Phase

**I am currently on: Phase 2 — LangChain Agent with Tool Calling**

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

### What I'm Working On Now

- Starting Phase 2: building a text-based LangChain agent with tool calling

### Current Challenges

- (none yet)



# Phase 3

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



# Phase 4

## Current Phase

**I am currently on: Phase 4 — Multi-Agent System with LangGraph**

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
  - LangSmith tracing configured for observability (env vars pushed from Pydantic settings in main.py)
  - Full booking flow tested end-to-end: identify → triage → follow-up questions → slot search → book
- **Phase 3 complete:**
  - pgvector extension enabled in Supabase with HNSW index (cosine distance)
  - medical_knowledge table with vector(1536) column, md5(content) unique index, and JSONB metadata
  - match_medical_knowledge RPC function for similarity search with configurable threshold and count
  - 24 knowledge chunks written across 10 specialties + 1 emergency severity guide, organized by symptom cluster (200–500 tokens each)
  - Iterative chunk quality improvements: added focused migraine-with-aura chunk (fixed Neurology vs Ophthalmology ambiguity) and focused diabetes symptoms chunk (fixed colloquial symptom matching)
  - Ingestion pipeline (ingest_knowledge.py): batch embeds via OpenAI text-embedding-3-small, idempotent with skip-if-exists logic
  - rag_retriever.py as single source of truth for all embedding logic (constants, embed_texts batch, embed_query single), imported by both ingestion and retrieval
  - Hybrid triage_symptoms tool: combines keyword search (ilike on symptom_specialty_map) with semantic search (vector similarity on medical_knowledge), graceful fallback if embedding API fails
  - Tool input schema updated with description field for full natural language + symptoms list for keywords
  - RetrievedChunk TypedDict for typed retriever results
  - OpenAI API called directly via httpx (no SDK dependency)
  - Test suite (test_retriever.py): 20 test cases covering direct matches, colloquial language, ambiguous symptoms, and edge cases — 100% pass rate
  - End-to-end tested: "sharp pains behind my eyes with flashing lights" correctly routes to Neurology through the full agent flow, confirmed in LangSmith traces

### What I'm Working On Now

- Starting Phase 4: Multi-agent system with LangGraph

### Current Challenges

- Single monolithic agent handles all responsibilities (intake, triage, scheduling) — system prompt is getting complex
- Agent doesn't always ask for explicit booking confirmation before booking (prompt tuning needed)
- No structured handoff between conversation stages (identification → triage → scheduling)
