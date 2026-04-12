-- ============================================================
-- RPC Functions for atomic multi-step operations
-- Run this in Supabase SQL Editor AFTER 001_schema.sql and 002_seed.sql
-- ============================================================

-- ============================================================
-- create_doctor_with_details
-- Atomically creates a doctor + specialty links + availability.
-- If any step fails, the entire operation is rolled back.
--
-- Parameters are passed as JSON so the function signature stays
-- simple and works well with Supabase's .rpc() client.
-- ============================================================

CREATE OR REPLACE FUNCTION create_doctor_with_details(
    p_full_name     TEXT,
    p_email         TEXT DEFAULT NULL,
    p_phone         TEXT DEFAULT NULL,
    p_image_url     TEXT DEFAULT NULL,
    p_specialty_ids UUID[] DEFAULT '{}',
    p_availability  JSONB DEFAULT '[]'
)
RETURNS JSON
LANGUAGE plpgsql
AS $$
DECLARE
    v_doctor_id UUID;
    v_avail     JSONB;
    v_result    JSON;
BEGIN
    -- Step 1: Insert the doctor
    INSERT INTO doctors (full_name, email, phone, image_url)
    VALUES (p_full_name, p_email, p_phone, p_image_url)
    RETURNING id INTO v_doctor_id;

    -- Step 2: Link specialties (if any provided)
    IF array_length(p_specialty_ids, 1) IS NOT NULL THEN
        INSERT INTO doctor_specialties (doctor_id, specialty_id)
        SELECT v_doctor_id, unnest(p_specialty_ids);
    END IF;

    -- Step 3: Add availability templates (if any provided)
    -- Expected JSONB format: [{"day_of_week": "monday", "start_time": "09:00", "end_time": "14:00", "slot_duration_min": 30}, ...]
    IF jsonb_array_length(p_availability) > 0 THEN
        INSERT INTO doctor_availability (doctor_id, day_of_week, start_time, end_time, slot_duration_min)
        SELECT
            v_doctor_id,
            (elem->>'day_of_week')::day_of_week,
            (elem->>'start_time')::TIME,
            (elem->>'end_time')::TIME,
            COALESCE((elem->>'slot_duration_min')::INTEGER, 30)
        FROM jsonb_array_elements(p_availability) AS elem;
    END IF;

    -- Return the complete doctor record with specialties and availability
    SELECT json_build_object(
        'id', d.id,
        'full_name', d.full_name,
        'email', d.email,
        'phone', d.phone,
        'image_url', d.image_url,
        'is_active', d.is_active,
        'created_at', d.created_at,
        'specialties', COALESCE(
            (SELECT json_agg(json_build_object('id', s.id, 'name', s.name))
             FROM doctor_specialties ds
             JOIN specialties s ON s.id = ds.specialty_id
             WHERE ds.doctor_id = d.id),
            '[]'::json
        ),
        'availability', COALESCE(
            (SELECT json_agg(json_build_object(
                'id', da.id,
                'day_of_week', da.day_of_week,
                'start_time', da.start_time,
                'end_time', da.end_time,
                'slot_duration_min', da.slot_duration_min
             ))
             FROM doctor_availability da
             WHERE da.doctor_id = d.id),
            '[]'::json
        )
    ) INTO v_result
    FROM doctors d
    WHERE d.id = v_doctor_id;

    RETURN v_result;
END;
$$;
