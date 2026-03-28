# Medical Voice Agent — Phase 1

## Quick Start

### 1. Database Setup (Supabase)

Go to your Supabase project → SQL Editor and run these in order:

1. `schema.sql` — creates all 9 tables, enums, indexes, and triggers
2. `seed.sql` — loads specialties, symptom mappings, doctors, patients, sample appointments
3. `rpc_functions.sql` — creates the `create_doctor_with_details` transactional function

### 2. Environment Setup

```bash
cd backend
cp .env.example .env
```

Edit `.env` with your Supabase credentials (found in Supabase → Settings → API):

```
SUPABASE_URL=https://your-project-id.supabase.co
SUPABASE_SERVICE_KEY=your-service-role-key-here
TIMEZONE=America/Chicago
SCHEDULING_HORIZON_DAYS=30
DEFAULT_SLOT_DURATION_MIN=30
```

### 3. Install & Run

```bash
uv sync
uv run uvicorn app.main:app --reload
```

### 4. Verify

Open http://localhost:8000/docs for the interactive API docs.

Test these endpoints:

```
GET  /health
GET  /api/v1/admin/specialties
GET  /api/v1/admin/doctors
GET  /api/v1/admin/patients
GET  /api/v1/admin/appointments
GET  /api/v1/admin/slots/by-specialty?specialty_id=a1000000-0000-0000-0000-000000000001
GET  /api/v1/admin/slots/by-specialty?specialty_id=a1000000-0000-0000-0000-000000000001&preferred_day=next monday&preferred_time=morning
GET  /api/v1/admin/slots/by-specialty?specialty_id=a1000000-0000-0000-0000-000000000001&preferred_day=earliest
```

## Project Structure

```
backend/
├── .env.example              ← template for environment variables
├── pyproject.toml            ← uv project config and dependencies
├── schema.sql                ← database schema (9 tables)
├── seed.sql                  ← realistic test data
├── rpc_functions.sql         ← transactional Postgres functions
└── app/
    ├── main.py               ← FastAPI entry point, router mounting
    ├── config.py             ← Pydantic settings from .env
    ├── supabase_client.py    ← database client singleton
    ├── models/
    │   ├── block.py          ← BlockIn
    │   ├── db_rows.py        ← TypedDicts for Supabase results
    │   ├── doctor.py         ← DoctorIn, AvailabilityIn, DoctorCreateIn
    │   ├── patient.py        ← PatientIn
    │   ├── slot.py           ← SlotDict
    │   └── specialty.py      ← SpecialtyOut
    ├── services/
    │   ├── time_utils.py     ← NLP date parsing + voice formatting
    │   └── slot_engine.py    ← core scheduling algorithm
    └── api/
        └── admin/
            ├── specialty_routes.py
            ├── doctor_routes.py
            ├── patient_routes.py
            ├── appointment_routes.py
            ├── block_routes.py
            └── slot_routes.py
```
