

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
