-- ============================================================
-- finalize_reschedule_appointment
-- Run this in Supabase SQL Editor AFTER 001_schema.sql.
--
-- Atomically commits the final reschedule after the patient confirms
-- a new slot. Preview/search still happens in application code; this
-- function is only for the final state-changing update.
-- ============================================================

CREATE OR REPLACE FUNCTION finalize_reschedule_appointment(
    p_appointment_id UUID,
    p_patient_id UUID,
    p_new_doctor_id UUID,
    p_new_specialty_id UUID,
    p_new_start_at TIMESTAMPTZ,
    p_new_end_at TIMESTAMPTZ,
    p_timezone TEXT DEFAULT 'America/Chicago'
)
RETURNS JSON
LANGUAGE plpgsql
AS $$
DECLARE
    v_appointment appointments%ROWTYPE;
    v_local_start TIMESTAMP;
    v_local_end TIMESTAMP;
    v_matches_template BOOLEAN;
BEGIN
    SELECT *
    INTO v_appointment
    FROM appointments
    WHERE id = p_appointment_id
      AND patient_id = p_patient_id
    FOR UPDATE;

    IF NOT FOUND THEN
        RETURN json_build_object('status', 'appointment_not_found');
    END IF;

    IF v_appointment.status = 'cancelled' THEN
        RETURN json_build_object('status', 'appointment_cancelled');
    END IF;

    IF v_appointment.status <> 'scheduled' THEN
        RETURN json_build_object('status', 'appointment_not_reschedulable');
    END IF;

    IF p_new_end_at <= p_new_start_at THEN
        RETURN json_build_object('status', 'invalid_slot');
    END IF;

    IF v_appointment.doctor_id = p_new_doctor_id
       AND v_appointment.specialty_id = p_new_specialty_id
       AND v_appointment.start_at = p_new_start_at
       AND v_appointment.end_at = p_new_end_at THEN
        RETURN json_build_object('status', 'same_slot');
    END IF;

    IF NOT EXISTS (
        SELECT 1
        FROM doctors d
        JOIN doctor_specialties ds
          ON ds.doctor_id = d.id
        WHERE d.id = p_new_doctor_id
          AND d.is_active = TRUE
          AND ds.specialty_id = p_new_specialty_id
    ) THEN
        RETURN json_build_object('status', 'invalid_doctor_specialty');
    END IF;

    v_local_start := p_new_start_at AT TIME ZONE p_timezone;
    v_local_end := p_new_end_at AT TIME ZONE p_timezone;

    IF v_local_end::date <> v_local_start::date THEN
        RETURN json_build_object('status', 'invalid_slot');
    END IF;

    SELECT EXISTS (
        SELECT 1
        FROM doctor_availability da
        WHERE da.doctor_id = p_new_doctor_id
          AND da.day_of_week::text = lower(trim(to_char(v_local_start, 'Day')))
          AND v_local_start::time >= da.start_time
          AND v_local_end::time <= da.end_time
          AND EXTRACT(EPOCH FROM (p_new_end_at - p_new_start_at))::INTEGER = da.slot_duration_min * 60
          AND MOD(
              EXTRACT(EPOCH FROM (v_local_start::time - da.start_time))::INTEGER,
              da.slot_duration_min * 60
          ) = 0
    ) INTO v_matches_template;

    IF NOT v_matches_template THEN
        RETURN json_build_object('status', 'invalid_slot');
    END IF;

    IF EXISTS (
        SELECT 1
        FROM doctor_blocks db
        WHERE db.doctor_id = p_new_doctor_id
          AND db.start_at < p_new_end_at
          AND db.end_at > p_new_start_at
    ) THEN
        RETURN json_build_object('status', 'doctor_blocked');
    END IF;

    BEGIN
        UPDATE appointments
        SET doctor_id = p_new_doctor_id,
            specialty_id = p_new_specialty_id,
            start_at = p_new_start_at,
            end_at = p_new_end_at
        WHERE id = p_appointment_id
          AND patient_id = p_patient_id;
    EXCEPTION
        WHEN exclusion_violation THEN
            RETURN json_build_object('status', 'slot_unavailable');
        WHEN foreign_key_violation THEN
            RETURN json_build_object('status', 'invalid_doctor_specialty');
        WHEN check_violation THEN
            RETURN json_build_object('status', 'invalid_slot');
    END;

    RETURN json_build_object(
        'status', 'ok',
        'appointment_id', p_appointment_id
    );
END;
$$;
