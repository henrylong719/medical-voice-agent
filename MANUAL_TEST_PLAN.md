# Manual End-to-End Test Plan for Medical Voice Agent Backend

## Testing Strategy Summary

This plan is designed for manual testing against the real FastAPI backend, real Supabase data, and the current LangGraph chat workflow in this repository. The goal is to validate the actual conversational behavior and business mutations, not just whether endpoints return `200`.

- Treat each manual chat scenario as one traceable transaction keyed by `thread_id`.
- Prefer `POST /api/v1/chat/invoke` for most conversation testing because it is easier to drive multi-turn flows than SSE.
- Reuse the same `thread_id` within a case and use a new `thread_id` for every new case.
- Capture request/response pairs, backend logs, and before/after DB state for every mutating scenario.
- Prioritize safety and destructive actions first: identity ambiguity, wrong-patient actions, wrong-appointment mutations, stale-slot booking, and reschedule finalization.
- Validate prompt/tool alignment on every chat case:
  - the Supervisor should route correctly
  - Intake should not guess identity
  - Triage should happen before booking when needed
  - Scheduling should require explicit confirmation before booking, rescheduling, or cancelling
- Verify persistence after every mutation in `patients`, `patient_identifiers`, and `appointments`.
- Do not expect the chat API to populate the app's `public.conversations` table. Current conversation persistence lives in the LangGraph checkpointer, not in the application tables defined in `001_schema.sql`.

### Severity Legend

- `Critical`: wrong-patient action, wrong-appointment action, unsafe ambiguity handling, or demo-stopper
- `High`: broken core workflow, wrong mutation, or major conversational regression
- `Medium`: recoverable UX/conversation defect, degraded fallback, or observability gap

## Preflight

- Confirm the backend is started from `backend/` with:

```bash
uv run uvicorn app.main:app --reload
```

- Confirm the following variables are configured in `backend/.env`:
  - `SUPABASE_URL`
  - `SUPABASE_SERVICE_KEY`
  - `SUPABASE_DB_URI`
  - `ANTHROPIC_API_KEY`
  - `OPENAI_API_KEY` if semantic triage is in scope
- Confirm the following SQL files have been applied in order:
  - `backend/sql/001_schema.sql`
  - `backend/sql/002_seed.sql`
  - `backend/sql/003_create_doctor_with_details.sql`
  - `backend/sql/004_finalize_reschedule_appointment.sql`
  - `backend/sql/005_rag.sql`
- Confirm the clinic timezone is `America/Chicago`.
- Confirm scheduling horizon is `30` days unless intentionally overridden.
- Confirm Swagger UI loads at `http://localhost:8000/docs`.
- Confirm `GET /health` returns `{"status":"healthy","version":"0.1.0"}`.
- Confirm `GET /api/v1/admin/specialties`, `GET /api/v1/admin/doctors`, and `GET /api/v1/admin/patients` all return seeded data.
- If semantic triage is in scope, run:

```bash
cd backend
uv run python -m app.services.ingest_knowledge
uv run python -m app.services.test_retriever
```

### Current Data Caveat

The current date in this workspace is April 11, 2026. The seeded scheduled appointments in `backend/sql/002_seed.sql` are on April 7, 2026, so they are already in the past.

That means:

- seeded appointment rows are still useful for readback/reference
- but `find_appointment` only returns future scheduled appointments
- so reschedule/cancel happy-path tests require a fresh future appointment fixture

### Seeded Patient Cheat Sheet

Useful seeded identity data:

- Alice Johnson, `1992-05-14`, phone `555-1001`, MRN `MRN-1001`
- Bob Martinez, `1989-11-02`, phone `555-1002`, MRN `MRN-1002`
- Carol Williams, `1994-03-21`, phone `555-1003`, passport `P1234567`
- Dan Brown, `1985-07-09`, phone `555-1004`, driver's license `DL-445566`
- Emma Davis, `1990-01-30`, phone `555-1005`, clinic patient ID `CLINIC-9005`

### Recommended Custom Fixtures

Create these before the ambiguity and mutation cases.

Fixture A: ambiguous demographics
- Create two patients with the same `full_name` and `date_of_birth`, but different phone numbers.
- Suggested values:
  - `Alex Kim Manual`, `1990-01-02`, `555-2100`
  - `Alex Kim Manual`, `1990-01-02`, `555-2101`

Fixture B: unresolved ambiguity
- Create two patients with the same `full_name`, `date_of_birth`, and `phone`.
- Suggested values:
  - `Jordan Lee Manual`, `1988-08-08`, `555-2200`
  - `Jordan Lee Manual`, `1988-08-08`, `555-2200`
- Do not give the caller a usable strong identifier for this case.

Fixture C: future appointment for reschedule/cancel
- Easiest path:
  - first complete a booking happy-path case for one seeded patient
  - then use the created future appointment for the reschedule and cancel cases
- Alternative path:
  - use `/api/v1/admin/slots/by-specialty` or `/api/v1/admin/slots/by-doctor`
  - then insert a future scheduled appointment directly in SQL using one returned live slot

### Useful SQL Checks

```sql
select id, full_name, date_of_birth, phone
from patients
order by full_name;
```

```sql
select patient_id, identifier_type, identifier_value
from patient_identifiers
order by patient_id, identifier_type;
```

```sql
select
  a.id,
  p.full_name as patient_name,
  d.full_name as doctor_name,
  s.name as specialty_name,
  a.start_at,
  a.end_at,
  a.status
from appointments a
join patients p on p.id = a.patient_id
join doctors d on d.id = a.doctor_id
join specialties s on s.id = a.specialty_id
order by a.start_at desc;
```

```sql
select count(*) as medical_knowledge_rows,
       count(embedding) as embedded_rows
from medical_knowledge;
```

## Prioritized Manual Test Plan

| ID | Bucket | Priority | Scenario / Exact Utterances | Expected Conversation and Tool Path | Expected DB Outcomes | Failure Signals | Severity |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `MP-01` | Must-pass before demo | P1 | `hi` -> `I need to book an appointment.` -> `This is my first visit.` -> `My name is Maya Chen. My birthday is October 26 1985. My phone number is 555-3300.` -> confirm phone when asked -> `I've had recurring headaches with flashing lights.` -> accept Neurology if offered -> `Next week.` -> `Morning.` -> choose first slot -> confirm booking. | Expected route: Supervisor greeting -> booking intent -> visit-status question -> Intake new-patient registration path. Expected tool order: `register_patient(full_name, date_of_birth, phone)` only after the phone is confirmed, then `triage_symptoms`, then `find_slots`, then `book_appointment`. Triage should happen before scheduling. Scheduling should ask day/week first, then morning/afternoon, then final confirmation. | One new `patients` row for Maya Chen. One new `appointments` row in `scheduled` status. No duplicate patient rows. | Assistant skips the new/returning question, registers before confirming the phone number, asks for symptoms before identity is complete, books before triage, or books without explicit confirmation. | Critical |
| `MP-02` | Must-pass before demo | P1 | `I need to book an appointment.` -> `I've been seen there before.` -> `Alice Johnson, 1992-05-14.` -> `Yes, that's me.` -> `I've had chest pain when I exercise.` -> accept Cardiology if offered -> `Earliest available.` -> choose a day -> choose a time -> confirm. | Expected route: booking intent -> visit-status question -> Intake returning-patient path. Expected tool order: `find_patients_by_demographics(full_name, date_of_birth)`, then no patient is committed until the user explicitly confirms the matched record, then `triage_symptoms`, `find_slots`, `book_appointment`. | One new future `appointments` row for Alice Johnson in `scheduled` status. Existing patient row reused. | Assistant commits identity before explicit confirmation, skips triage and goes straight to slots, or creates a duplicate patient row for Alice. | Critical |
| `MP-03` | Must-pass before demo | P1 | Use Fixture A. `I need to book an appointment.` -> `I've been seen there before.` -> `Alex Kim Manual, 1990-01-02.` -> when asked for phone: `555-2101.` -> `Yes, that's me.` | Expected tool order: `find_patients_by_demographics(full_name, date_of_birth)` returning multiple matches, then `find_patients_by_demographics(full_name, date_of_birth, phone)` returning one clear match. Assistant should ask for phone before any strong identifier. | No DB mutation is required unless the case is extended into booking. Identity should resolve to the correct existing patient only after confirmation. | Assistant asks for MRN/passport before phone, guesses a patient without the phone, or proceeds while ambiguity remains. | Critical |
| `MP-04` | Must-pass before demo | P1 | `I need to book an appointment.` -> `I've been seen there before.` -> `Emma Davis, 1990-01-31.` -> when no match is reported: `My clinic patient number is CLINIC-9005.` -> `Yes, that's me.` | Expected tool order: `find_patients_by_demographics` returns `no_match`, then assistant offers strong-identifier fallback, then `find_patient_by_identifier(identifier_type="external_patient_id", identifier_value="CLINIC-9005")`. Assistant should still confirm the matched patient before proceeding. | No new patient row. Existing Emma Davis record is reused. | Assistant offers registration before identifier fallback, guesses a record, or accepts the identifier without explicit confirmation of the matched patient. | High |
| `MP-05` | Must-pass before demo | P1 | Use Fixture B. `I need to book an appointment.` -> `I've been seen there before.` -> `Jordan Lee Manual, 1988-08-08.` -> when asked for phone: `555-2200.` -> when asked for a strong identifier: `I don't know any identifier.` | Expected tool path: demographic lookup -> still multiple after phone -> ask for MRN/passport/driver's license/clinic ID -> if none available, assistant must stop with staff-help guidance. No triage or scheduling should follow. | No `patients` or `appointments` mutation. | Assistant guesses which Jordan Lee record is correct, moves into triage/scheduling anyway, or claims a human handoff was completed even though no such implementation exists. | Critical |
| `MP-06` | Must-pass before demo | P1 | Requires Fixture C. `I need to reschedule my appointment.` -> verify patient identity -> `It's my cardiology appointment with Dr. Sarah Chen.` -> confirm the appointment when read back -> `Next week.` -> `Morning.` -> choose first offered replacement slot -> confirm. | Booking-specific visit-status question should not appear. Expected tool order: identity lookup first, then `find_appointment(patient_id, doctor_name or specialty_name)`, then `reschedule_appointment(appointment_id, patient_id, preferred_day, preferred_time)` for preview, then `reschedule_appointment(... new_doctor_id, new_specialty_id, new_start_at, new_end_at)` to finalize. Preview should state the current appointment has not been cancelled. | The same appointment row should be updated in place with a new date/time. Appointment `id` should remain the same after finalization. | Assistant asks whether the patient is new or returning, calls `book_appointment` plus `cancel_appointment` instead of the finalize path, cancels before the new slot is confirmed, or creates a second appointment instead of updating in place. | Critical |
| `MP-07` | Must-pass before demo | P1 | Requires Fixture C. `I need to cancel my appointment.` -> verify identity -> `It's the cardiology appointment.` -> when asked to confirm: `No, keep it.` | Booking-specific visit-status question should not appear. Expected tool path: identity lookup -> `find_appointment` -> assistant confirms the exact appointment -> no `cancel_appointment` call after the user declines. | Appointment stays `scheduled`. No row changes. | Any cancellation occurs after the user says no, or the assistant confirms cancellation anyway. | Critical |
| `MP-08` | Must-pass before demo | P1 | Requires Fixture C. `I need to cancel my appointment.` -> verify identity -> `It's the cardiology appointment.` -> `Yes, cancel it.` | Expected tool order: identity lookup -> `find_appointment` -> explicit confirmation -> `cancel_appointment(patient_id, appointment_id)`. | The target appointment row changes from `scheduled` to `cancelled`. No duplicate appointment row is created. | Wrong appointment cancelled, no confirmation step, or appointment remains `scheduled` after the assistant reports success. | Critical |
| `LR-09` | Likely regression | P2 | Admin route validation check. `POST /api/v1/admin/patients/search` with `{"full_name":"Alice Johnson","date_of_birth":"05/14/1992"}` and then with `{"full_name":"Alice Johnson","date_of_birth":"1992-05-14"}`. | The admin route should reject the slash-formatted DOB with `422` because `PatientSearchIn` requires ISO `YYYY-MM-DD`. The ISO request should succeed. This is intentionally different from chat, where DOB normalization is handled inside the tool layer. | No mutation. Read-only check only. | Admin search silently accepts non-ISO DOB, or chat and admin behaviors are documented/tested as if they were identical. | High |
| `LR-10` | Likely regression | P2 | Slot race test. Start a booking or reschedule flow and stop after the assistant reads out a specific slot. Before confirming in chat, take that exact slot through SQL or a second session. Then confirm the stale slot in the original session. | Expected path: assistant reaches `book_appointment` or reschedule finalize step, but the backend validation should reject the stale slot. The assistant should report that the slot is no longer available and ask the user to choose another one. | No duplicate overlapping appointment rows. In reschedule, the original appointment should remain unchanged if finalize fails. | Double-booking, overlapping rows for the same doctor/time, original appointment moved even though finalize failed, or assistant claims success after a taken slot. | Critical |
| `LR-11` | Likely regression | P2 | RAG degradation test. Temporarily remove `OPENAI_API_KEY` or leave `medical_knowledge` empty. Then run a booking flow for a verified patient and say `I have chest pain and shortness of breath.` | Expected behavior: semantic search may be unavailable, but keyword-based triage should still produce a usable Cardiology path rather than crashing the flow. The assistant should still continue into scheduling. | No unexpected DB mutation other than any booking the tester chooses to complete. | HTTP 500/503 during triage, or complete failure to route a strong keyword symptom set when semantic retrieval is unavailable. | High |
| `LR-12` | Likely regression | P2 | Mid-conversation intent switch. Start a booking flow, verify identity, maybe answer one triage question, then say `Actually, I need to reschedule instead.` | Expected route: Supervisor detects explicit intent switch, clears stale booking/triage context, and reroutes to scheduling. It should not keep asking symptom questions after the switch. | No mutation unless the new reschedule flow is completed. | Assistant ignores the new intent, continues triage, or reuses stale booking state in the reschedule path. | High |
| `LR-13` | Likely regression | P2 | Wrong-patient correction mid-thread. During a returning-patient lookup, after the assistant proposes a matched record, reply `No, that's the wrong patient.` | Expected route: Supervisor should clear `patient_id`, `patient_name`, and stale appointment state, then route back to Intake. The assistant should not continue using the wrong patient. | No appointment mutation. No wrong patient committed to the thread state. | Assistant keeps using the old patient after the correction, continues scheduling anyway, or leaks appointment context from the wrong person. | Critical |
| `LR-14` | Likely regression | P2 | Streaming API check. Send `hello` to `POST /api/v1/chat` with `curl -N`. | Expected response should be `text/event-stream`, contain `data:` chunks, and end with `data: [DONE]`. | No business-table mutation expected. | Missing `[DONE]`, wrong content type, or full JSON returned instead of SSE. | Medium |

## Must-Pass Before Demo

- `MP-01`: new-patient booking must complete end-to-end without skipping intake or triage
- `MP-02`: returning-patient booking must require explicit confirmation before patient identity is committed
- `MP-03`: ambiguous demographics must use phone as the first disambiguator
- `MP-05`: unresolved ambiguity must stop safely and point to staff help without guessing
- `MP-06`: reschedule must skip the booking-only visit-status question and must preview before finalizing
- `MP-07`: cancellation must not happen after the user declines
- `MP-08`: cancellation happy path must cancel the intended appointment only

## Likely Regression Cases

- `LR-09`: admin DOB validation differs from chat DOB normalization
- `LR-10`: stale slot rejection during booking/reschedule finalization
- `LR-11`: semantic triage degraded but keyword triage still operational
- `LR-12`: explicit intent switch mid-conversation
- `LR-13`: patient identity correction mid-thread
- `LR-14`: SSE streaming contract

## Hard-to-Catch Conversational Bugs

- The assistant asks more than one question in a single turn despite the prompts explicitly requiring one at a time.
- The assistant asks the booking-only new/returning question in reschedule or cancel flows.
- The assistant commits a returning-patient identity before the patient explicitly confirms the matched record.
- The assistant asks for strong identifiers before asking for phone during ambiguous demographic lookup.
- The assistant offers morning/afternoon times from the wrong bucket after the patient asked for a specific time of day.
- The assistant presents more than three slot options at once.
- The assistant continues into triage or scheduling after unresolved identity ambiguity.
- The assistant uses a stale `appointment_id` after a patient correction or intent switch.
- The assistant claims a human handoff has happened even though the current implementation only offers staff-help messaging.

## High-Risk Bug Checklist

- [ ] Booking does not ask whether the patient is new or returning
- [ ] Reschedule or cancel does ask whether the patient is new or returning
- [ ] New patient registration occurs before `full_name`, `date_of_birth`, and confirmed `phone` are all collected
- [ ] Returning-patient lookup starts with a strong identifier instead of `full_name + date_of_birth`
- [ ] Ambiguous identity is resolved by guessing
- [ ] The assistant skips phone and jumps straight to MRN/passport/driver's license
- [ ] The assistant keeps using a patient after the user says it is the wrong record
- [ ] Triage is skipped in a booking flow where no specialty is already known
- [ ] The assistant offers slots before a specialty is chosen
- [ ] The assistant books, reschedules, or cancels without explicit confirmation
- [ ] Reschedule preview silently mutates the appointment before final confirmation
- [ ] Reschedule finalization creates a second appointment row instead of updating in place
- [ ] Cancellation updates the wrong appointment
- [ ] A stale slot is booked after another action already took it
- [ ] Semantic retrieval failure crashes the chat flow instead of degrading
- [ ] The assistant lists unavailable slots as valid after a block or booking conflict

## Manual Observability Checklist

### Chat and Routing

- [ ] First turn greeting is present on a fresh thread
- [ ] Reusing a `thread_id` continues the same conversation instead of restarting
- [ ] Booking route asks new vs returning before Intake continues
- [ ] Reschedule/cancel route skips that question
- [ ] One question at a time is maintained through Intake, Triage, and Scheduling
- [ ] Exact confirmation is required before committing patient identity or appointment mutations

### Identity and Safety

- [ ] Returning-patient identity begins with demographics
- [ ] Phone is used to narrow ambiguous demographic matches
- [ ] Strong identifier fallback is offered only after demographics and phone are insufficient
- [ ] Unresolved ambiguity produces staff-help guidance and stops
- [ ] No wrong-patient state leaks after a correction

### Scheduling

- [ ] Booking asks day/week first, then morning/afternoon
- [ ] No more than three concrete times are offered at once
- [ ] Full date and time are included in final confirmations
- [ ] Reschedule preview states that the current appointment has not been cancelled
- [ ] Reschedule finalize updates the existing appointment row
- [ ] Cancel changes only `status` on the selected row

### Backend and DB

- [ ] `/health` passes
- [ ] `/docs` loads
- [ ] `/api/v1/chat/invoke` returns `503` with a clear message if required chat config is missing
- [ ] `/api/v1/chat` returns SSE chunks and `[DONE]`
- [ ] `patients` and `appointments` reflect the expected mutations after each mutating case
- [ ] No duplicate or overlapping appointments are created for the same doctor/time
- [ ] `public.conversations` is not mistakenly used as the expected persistence target for chat traces in this backend

## Useful SQL Checks During Execution

```sql
-- Inspect one patient and any future appointments
select
  p.id,
  p.full_name,
  p.date_of_birth,
  p.phone,
  a.id as appointment_id,
  a.start_at,
  a.end_at,
  a.status
from patients p
left join appointments a on a.patient_id = p.id
where p.full_name = 'Alice Johnson'
order by a.start_at desc nulls last;
```

```sql
-- Inspect custom ambiguity fixtures
select id, full_name, date_of_birth, phone
from patients
where full_name in ('Alex Kim Manual', 'Jordan Lee Manual')
order by full_name, phone, id;
```

```sql
-- Inspect appointment mutations after a reschedule/cancel test
select
  a.id,
  p.full_name as patient_name,
  d.full_name as doctor_name,
  s.name as specialty_name,
  a.start_at,
  a.end_at,
  a.status,
  a.updated_at
from appointments a
join patients p on p.id = a.patient_id
join doctors d on d.id = a.doctor_id
join specialties s on s.id = a.specialty_id
where p.full_name = 'Alice Johnson'
order by a.updated_at desc, a.start_at desc;
```

```sql
-- Quick RAG readiness check
select count(*) as total_rows, count(embedding) as embedded_rows
from medical_knowledge;
```

## Lightweight Test Session Template

| Field | Value |
| --- | --- |
| Date / Time |  |
| Tester |  |
| Environment |  |
| Backend base URL |  |
| Case ID |  |
| `thread_id` |  |
| Patient / fixture used |  |
| Result | Pass / Fail / Blocked |
| Request/response notes |  |
| Backend log notes |  |
| DB verification notes |  |
| Bugs filed / links |  |

### Per-Case Quick Log

| Step | Expected | Actual | Pass/Fail |
| --- | --- | --- | --- |
| Intent routing |  |  |  |
| Identity handling |  |  |  |
| Triage or appointment lookup |  |  |  |
| Scheduling / mutation |  |  |  |
| DB verification |  |  |  |
