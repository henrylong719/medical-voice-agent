-- ============================================================
-- Medical Voice Agent — Database Schema
-- Phase 1: Full schema for Supabase (PostgreSQL)
-- ============================================================

-- Enable UUID generation (Supabase usually has this, but just in case)
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "btree_gist";

-- ============================================================
-- CUSTOM TYPES (Enums)
-- ============================================================
-- We use Postgres enums for small, fixed sets of values.
-- They give us free validation — the DB rejects bad data.

CREATE TYPE day_of_week AS ENUM (
    'monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday'
);

CREATE TYPE appointment_status AS ENUM (
    'scheduled', 'completed', 'cancelled', 'no_show'
);

-- ============================================================
-- GROUP 1: Medical Domain (specialties + symptom mapping)
-- These define WHAT the clinic can treat.
-- ============================================================

-- Specialties: the departments/areas of medicine
-- Examples: Cardiology, Neurology, Dermatology
CREATE TABLE specialties (
    id          UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    name        TEXT NOT NULL UNIQUE,          -- e.g. "Cardiology"
    description TEXT,                          -- what this specialty covers
    created_at  TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Symptom-to-specialty mapping with weights and follow-up questions
-- This powers the keyword-based triage in Phase 2.
-- In Phase 3, we'll ADD semantic search alongside this.
CREATE TABLE symptom_specialty_map (
    id                  UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    specialty_id        UUID NOT NULL REFERENCES specialties(id) ON DELETE CASCADE,
    symptom             TEXT NOT NULL,              -- e.g. "chest pain"
    weight              NUMERIC(3,2) NOT NULL DEFAULT 1.0,  -- relevance score (0.00–1.00 scale, but allows up to 9.99)
    follow_up_questions TEXT[],                     -- array of follow-up questions to ask
    created_at          TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    -- Prevent duplicate symptom entries for the same specialty
    UNIQUE (specialty_id, symptom)
);

-- ============================================================
-- GROUP 2: Doctors (providers + availability)
-- These define WHO can treat patients and WHEN.
-- ============================================================

-- Doctors: the providers
CREATE TABLE doctors (
    id          UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    full_name   TEXT NOT NULL,
    email       TEXT UNIQUE,
    phone       TEXT,
    image_url   TEXT,                            -- profile photo URL
    is_active   BOOLEAN NOT NULL DEFAULT TRUE,   -- soft delete / deactivation
    created_at  TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Junction table: which doctors practice which specialties
-- This is the many-to-many relationship we discussed!
CREATE TABLE doctor_specialties (
    id            UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    doctor_id     UUID NOT NULL REFERENCES doctors(id) ON DELETE CASCADE,
    specialty_id  UUID NOT NULL REFERENCES specialties(id) ON DELETE CASCADE,

    -- A doctor can't be linked to the same specialty twice
    UNIQUE (doctor_id, specialty_id)
);

-- Doctor weekly availability templates
-- These are RECURRING patterns, not specific dates.
-- e.g. "Dr. Smith works every Monday from 9:00 to 14:00, 30-min slots"
CREATE TABLE doctor_availability (
    id                UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    doctor_id         UUID NOT NULL REFERENCES doctors(id) ON DELETE CASCADE,
    day_of_week       day_of_week NOT NULL,          -- our custom enum
    start_time        TIME NOT NULL,                 -- e.g. 09:00
    end_time          TIME NOT NULL,                 -- e.g. 14:00
    slot_duration_min INTEGER NOT NULL DEFAULT 30,   -- appointment length in minutes

    -- end_time must be after start_time
    CONSTRAINT valid_time_range CHECK (end_time > start_time),

    -- slot duration must be reasonable (15 min to 4 hours)
    CONSTRAINT valid_slot_duration CHECK (slot_duration_min BETWEEN 15 AND 240),

    -- A doctor can't have overlapping templates on the same day
    -- (simplified: one entry per doctor per day. If you need multiple
    -- blocks per day like 9-12 and 14-17, you'd create two rows.)
    UNIQUE (doctor_id, day_of_week, start_time)
);

-- Doctor blocks: specific dates/times when a doctor is UNAVAILABLE
-- e.g. vacation, conference, sick day
-- The slot engine subtracts these from the weekly templates.
CREATE TABLE doctor_blocks (
    id          UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    doctor_id   UUID NOT NULL REFERENCES doctors(id) ON DELETE CASCADE,
    start_at    TIMESTAMPTZ NOT NULL,          -- block start (specific datetime)
    end_at      TIMESTAMPTZ NOT NULL,          -- block end
    reason      TEXT,                          -- optional: "Vacation", "Conference"
    created_at  TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    -- end must be after start
    CONSTRAINT valid_block_range CHECK (end_at > start_at)
);

-- ============================================================
-- GROUP 3: Patients
-- Core demographic record plus optional external identifiers.
-- ============================================================

CREATE TABLE patients (
    id          UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    full_name   TEXT NOT NULL,
    date_of_birth DATE NOT NULL,
    phone       TEXT,
    email       TEXT,
    allergies   TEXT[] DEFAULT '{}',            -- e.g. {'penicillin', 'peanuts'}
    created_at  TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE patient_identifiers (
    id               UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    patient_id       UUID NOT NULL REFERENCES patients(id) ON DELETE CASCADE,
    identifier_type  TEXT NOT NULL,
    identifier_value TEXT NOT NULL,
    issuing_country  TEXT,
    is_primary       BOOLEAN NOT NULL DEFAULT FALSE,
    created_at       TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    CONSTRAINT valid_identifier_type CHECK (
        identifier_type IN ('mrn', 'passport', 'drivers_license', 'external_patient_id')
    ),
    CONSTRAINT unique_patient_identifier UNIQUE (identifier_type, identifier_value)
);

-- ============================================================
-- GROUP 4: Appointments
-- Where patients and doctors meet at a specific time.
-- ============================================================

CREATE TABLE appointments (
    id              UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    patient_id      UUID NOT NULL REFERENCES patients(id) ON DELETE CASCADE,
    doctor_id       UUID NOT NULL REFERENCES doctors(id) ON DELETE CASCADE,
    specialty_id    UUID NOT NULL REFERENCES specialties(id),
    start_at        TIMESTAMPTZ NOT NULL,
    end_at          TIMESTAMPTZ NOT NULL,
    status          appointment_status NOT NULL DEFAULT 'scheduled',
    reason          TEXT,                       -- why the patient is coming in
    notes           TEXT,                       -- post-appointment notes
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    -- appointment must have positive duration
    CONSTRAINT valid_appointment_range CHECK (end_at > start_at),

    -- appointment specialty must actually belong to the selected doctor
    CONSTRAINT valid_appointment_doctor_specialty
        FOREIGN KEY (doctor_id, specialty_id)
        REFERENCES doctor_specialties(doctor_id, specialty_id),

    -- severity rating from triage (1 = minor, 10 = critical)
    severity_rating INTEGER,
    CONSTRAINT valid_severity CHECK (severity_rating IS NULL OR severity_rating BETWEEN 1 AND 10)
);

-- Index for common queries: "find appointments for this patient"
CREATE INDEX idx_appointments_patient ON appointments(patient_id);

-- Index for slot engine: "find booked appointments for this doctor on this date range"
CREATE INDEX idx_appointments_doctor_time ON appointments(doctor_id, start_at, end_at);

-- Index for status filtering: "find all scheduled appointments"
CREATE INDEX idx_appointments_status ON appointments(status);

-- Prevent overlapping live appointments for the same doctor.
ALTER TABLE appointments
    ADD CONSTRAINT appointments_doctor_no_overlap
    EXCLUDE USING GIST (
        doctor_id WITH =,
        tstzrange(start_at, end_at, '[)') WITH &&
    )
    WHERE (status <> 'cancelled');

-- ============================================================
-- GROUP 5: Conversations
-- Logs of voice/chat interactions for debugging and review.
-- We'll populate these in Phase 6 (voice) but create the table now.
-- ============================================================

CREATE TABLE conversations (
    id              UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    patient_id      UUID REFERENCES patients(id),  -- nullable: patient might not be identified
    channel         TEXT NOT NULL DEFAULT 'chat',   -- 'chat' or 'voice'
    transcript      JSONB,                         -- full conversation log
    summary         TEXT,                          -- AI-generated summary
    started_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    ended_at        TIMESTAMPTZ,
    metadata        JSONB                          -- any extra info (duration, agent versions, etc.)
);

-- ============================================================
-- UPDATED_AT TRIGGER
-- Automatically update the updated_at column on row changes.
-- ============================================================

CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER set_appointments_updated_at
    BEFORE UPDATE ON appointments
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

-- ============================================================
-- PRIVILEGES FOR BACKEND SERVICE ACCESS
-- The backend uses the Supabase service_role key. After recreating
-- the public schema from scratch, we need to grant table/routine
-- privileges again so PostgREST can read and write these objects.
-- ============================================================

GRANT USAGE ON SCHEMA public TO service_role;
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO service_role;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO service_role;
GRANT ALL PRIVILEGES ON ALL ROUTINES IN SCHEMA public TO service_role;

ALTER DEFAULT PRIVILEGES IN SCHEMA public
    GRANT ALL PRIVILEGES ON TABLES TO service_role;
ALTER DEFAULT PRIVILEGES IN SCHEMA public
    GRANT ALL PRIVILEGES ON SEQUENCES TO service_role;
ALTER DEFAULT PRIVILEGES IN SCHEMA public
    GRANT ALL PRIVILEGES ON ROUTINES TO service_role;
