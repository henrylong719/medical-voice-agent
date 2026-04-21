"""
Extended tests for slot_engine — covers slot generation, conflict
subtraction, doctor blocks, booked-slot exclusion, and validation
edge cases.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from pytest import MonkeyPatch

from app.services import slot_engine
from tests.support import FakeQuery, FakeSupabase


# ── Helpers ───────────────────────────────────────────────────


def _fixed_now(monkeypatch: MonkeyPatch, dt: datetime) -> None:
    monkeypatch.setattr(slot_engine, "now_utc", lambda: dt)


def _stub_doctor(monkeypatch: MonkeyPatch) -> None:
    monkeypatch.setattr(
        slot_engine,
        "_get_doctor",
        lambda _: {"id": "doctor-1", "full_name": "Dr. Maya Chen", "is_active": True},
    )


def _stub_specialty(monkeypatch: MonkeyPatch) -> None:
    monkeypatch.setattr(
        slot_engine,
        "_get_specialty",
        lambda _: {"id": "spec-cardio", "name": "Cardiology"},
    )


def _stub_doctor_has_specialty(monkeypatch: MonkeyPatch) -> None:
    monkeypatch.setattr(slot_engine, "_doctor_has_specialty", lambda *_: True)


def _stub_no_conflicts(monkeypatch: MonkeyPatch) -> None:
    monkeypatch.setattr(slot_engine, "_get_booked_slots", lambda *_, **__: set())
    monkeypatch.setattr(slot_engine, "_get_doctor_blocks", lambda *_, **__: [])


def _monday_template(
    start: str = "09:00:00", end: str = "12:00:00", duration: int = 30
):
    return {
        "day_of_week": "monday",
        "start_time": start,
        "end_time": end,
        "slot_duration_min": duration,
    }


def _stub_templates(monkeypatch: MonkeyPatch, templates: list[dict]) -> None:
    monkeypatch.setattr(slot_engine, "_get_availability_templates", lambda _: templates)


# ============================================================
# _get_booked_slots
# ============================================================


def test_get_booked_slots_returns_empty_when_no_appointments(
    monkeypatch: MonkeyPatch,
) -> None:
    fake_supabase = FakeSupabase(tables={"appointments": [FakeQuery([])]})
    monkeypatch.setattr(slot_engine, "supabase", fake_supabase)

    result = slot_engine._get_booked_slots(
        "doctor-1",
        datetime(2026, 4, 13, 14, 0, tzinfo=timezone.utc),
        datetime(2026, 4, 13, 18, 0, tzinfo=timezone.utc),
    )

    assert result == set()


def test_get_booked_slots_excludes_appointment_by_id(
    monkeypatch: MonkeyPatch,
) -> None:
    query = FakeQuery([])
    fake_supabase = FakeSupabase(tables={"appointments": [query]})
    monkeypatch.setattr(slot_engine, "supabase", fake_supabase)

    slot_engine._get_booked_slots(
        "doctor-1",
        datetime(2026, 4, 13, 14, 0, tzinfo=timezone.utc),
        datetime(2026, 4, 13, 18, 0, tzinfo=timezone.utc),
        exclude_appointment_id="appt-1",
    )

    assert ("neq", ("id", "appt-1"), {}) in query.operations


def test_get_booked_slots_does_not_exclude_when_no_id(
    monkeypatch: MonkeyPatch,
) -> None:
    query = FakeQuery([])
    fake_supabase = FakeSupabase(tables={"appointments": [query]})
    monkeypatch.setattr(slot_engine, "supabase", fake_supabase)

    slot_engine._get_booked_slots(
        "doctor-1",
        datetime(2026, 4, 13, 14, 0, tzinfo=timezone.utc),
        datetime(2026, 4, 13, 18, 0, tzinfo=timezone.utc),
    )

    neq_ops = [op for op in query.operations if op[0] == "neq"]
    # Only the status != cancelled filter, not an id filter
    assert all(op[1][0] == "status" for op in neq_ops)


def test_get_booked_slots_parses_multiple_rows(
    monkeypatch: MonkeyPatch,
) -> None:
    query = FakeQuery(
        [
            {
                "start_at": "2026-04-13T14:00:00+00:00",
                "end_at": "2026-04-13T14:30:00+00:00",
            },
            {
                "start_at": "2026-04-13T15:00:00+00:00",
                "end_at": "2026-04-13T15:30:00+00:00",
            },
        ]
    )
    fake_supabase = FakeSupabase(tables={"appointments": [query]})
    monkeypatch.setattr(slot_engine, "supabase", fake_supabase)

    result = slot_engine._get_booked_slots(
        "doctor-1",
        datetime(2026, 4, 13, 14, 0, tzinfo=timezone.utc),
        datetime(2026, 4, 13, 16, 0, tzinfo=timezone.utc),
    )

    assert len(result) == 2


# ============================================================
# _get_doctor_blocks
# ============================================================


def test_get_doctor_blocks_returns_parsed_ranges(
    monkeypatch: MonkeyPatch,
) -> None:
    query = FakeQuery(
        [
            {
                "start_at": "2026-04-13T13:00:00+00:00",
                "end_at": "2026-04-13T18:00:00+00:00",
            }
        ]
    )
    fake_supabase = FakeSupabase(tables={"doctor_blocks": [query]})
    monkeypatch.setattr(slot_engine, "supabase", fake_supabase)

    result = slot_engine._get_doctor_blocks(
        "doctor-1",
        datetime(2026, 4, 13, 14, 0, tzinfo=timezone.utc),
        datetime(2026, 4, 13, 18, 0, tzinfo=timezone.utc),
    )

    assert len(result) == 1
    assert result[0] == (
        datetime(2026, 4, 13, 13, 0, tzinfo=timezone.utc),
        datetime(2026, 4, 13, 18, 0, tzinfo=timezone.utc),
    )


def test_get_doctor_blocks_empty_when_none(
    monkeypatch: MonkeyPatch,
) -> None:
    fake_supabase = FakeSupabase(tables={"doctor_blocks": [FakeQuery([])]})
    monkeypatch.setattr(slot_engine, "supabase", fake_supabase)

    result = slot_engine._get_doctor_blocks(
        "doctor-1",
        datetime(2026, 4, 13, 14, 0, tzinfo=timezone.utc),
        datetime(2026, 4, 13, 18, 0, tzinfo=timezone.utc),
    )

    assert result == []


# ============================================================
# _subtract_conflicts
# ============================================================


def test_subtract_conflicts_removes_exact_booked_match() -> None:
    slot = slot_engine.Slot(
        doctor_id="d1",
        doctor_name="Chen",
        specialty_id="s1",
        specialty_name="Cardiology",
        start_at=datetime(2026, 4, 13, 14, 0, tzinfo=timezone.utc),
        end_at=datetime(2026, 4, 13, 14, 30, tzinfo=timezone.utc),
    )
    booked = {(slot.start_at, slot.end_at)}

    result = slot_engine._subtract_conflicts([slot], booked, [])

    assert result == []


def test_subtract_conflicts_removes_partial_overlap_with_booked() -> None:
    slot = slot_engine.Slot(
        doctor_id="d1",
        doctor_name="Chen",
        specialty_id="s1",
        specialty_name="Cardiology",
        start_at=datetime(2026, 4, 13, 14, 0, tzinfo=timezone.utc),
        end_at=datetime(2026, 4, 13, 14, 30, tzinfo=timezone.utc),
    )
    # A 60-min booked appointment overlapping this 30-min slot
    booked = {
        (
            datetime(2026, 4, 13, 13, 45, tzinfo=timezone.utc),
            datetime(2026, 4, 13, 14, 45, tzinfo=timezone.utc),
        )
    }

    result = slot_engine._subtract_conflicts([slot], booked, [])

    assert result == []


def test_subtract_conflicts_removes_blocked_slots() -> None:
    slot = slot_engine.Slot(
        doctor_id="d1",
        doctor_name="Chen",
        specialty_id="s1",
        specialty_name="Cardiology",
        start_at=datetime(2026, 4, 13, 14, 0, tzinfo=timezone.utc),
        end_at=datetime(2026, 4, 13, 14, 30, tzinfo=timezone.utc),
    )
    blocks = [
        (
            datetime(2026, 4, 13, 13, 0, tzinfo=timezone.utc),
            datetime(2026, 4, 13, 18, 0, tzinfo=timezone.utc),
        )
    ]

    result = slot_engine._subtract_conflicts([slot], set(), blocks)

    assert result == []


def test_subtract_conflicts_keeps_non_conflicting_slots() -> None:
    slot = slot_engine.Slot(
        doctor_id="d1",
        doctor_name="Chen",
        specialty_id="s1",
        specialty_name="Cardiology",
        start_at=datetime(2026, 4, 13, 14, 0, tzinfo=timezone.utc),
        end_at=datetime(2026, 4, 13, 14, 30, tzinfo=timezone.utc),
    )
    # Booked slot is AFTER this slot
    booked = {
        (
            datetime(2026, 4, 13, 15, 0, tzinfo=timezone.utc),
            datetime(2026, 4, 13, 15, 30, tzinfo=timezone.utc),
        )
    }

    result = slot_engine._subtract_conflicts([slot], booked, [])

    assert len(result) == 1
    assert result[0].start_at == slot.start_at


def test_subtract_conflicts_mixed_booked_and_blocks() -> None:
    slot1 = slot_engine.Slot(
        doctor_id="d1",
        doctor_name="Chen",
        specialty_id="s1",
        specialty_name="Cardiology",
        start_at=datetime(2026, 4, 13, 14, 0, tzinfo=timezone.utc),
        end_at=datetime(2026, 4, 13, 14, 30, tzinfo=timezone.utc),
    )
    slot2 = slot_engine.Slot(
        doctor_id="d1",
        doctor_name="Chen",
        specialty_id="s1",
        specialty_name="Cardiology",
        start_at=datetime(2026, 4, 13, 14, 30, tzinfo=timezone.utc),
        end_at=datetime(2026, 4, 13, 15, 0, tzinfo=timezone.utc),
    )
    slot3 = slot_engine.Slot(
        doctor_id="d1",
        doctor_name="Chen",
        specialty_id="s1",
        specialty_name="Cardiology",
        start_at=datetime(2026, 4, 13, 15, 0, tzinfo=timezone.utc),
        end_at=datetime(2026, 4, 13, 15, 30, tzinfo=timezone.utc),
    )

    booked = {(slot1.start_at, slot1.end_at)}
    blocks = [(slot2.start_at, slot2.end_at)]

    result = slot_engine._subtract_conflicts([slot1, slot2, slot3], booked, blocks)

    assert len(result) == 1
    assert result[0].start_at == slot3.start_at


# ============================================================
# Slot.to_dict
# ============================================================


def test_slot_to_dict_includes_all_fields() -> None:
    slot = slot_engine.Slot(
        doctor_id="d1",
        doctor_name="Chen",
        specialty_id="s1",
        specialty_name="Cardiology",
        start_at=datetime(2026, 4, 13, 14, 0, tzinfo=timezone.utc),
        end_at=datetime(2026, 4, 13, 14, 30, tzinfo=timezone.utc),
    )

    d = slot.to_dict()

    assert d["doctor_id"] == "d1"
    assert d["doctor_name"] == "Chen"
    assert d["specialty_id"] == "s1"
    assert d["specialty_name"] == "Cardiology"
    assert d["start_at"] == "2026-04-13T14:00:00+00:00"
    assert d["end_at"] == "2026-04-13T14:30:00+00:00"
    assert "label" in d
    assert "date_label" in d


# ============================================================
# validate_slot_selection — extended edge cases
# ============================================================


def test_validate_slot_rejects_invalid_iso_format(monkeypatch: MonkeyPatch) -> None:
    result = slot_engine.validate_slot_selection(
        doctor_id="doctor-1",
        specialty_id="spec-cardio",
        start_at="not-a-date",
        end_at="also-not-a-date",
    )

    assert result is not None
    assert "invalid" in result.lower()


def test_validate_slot_rejects_naive_datetimes(monkeypatch: MonkeyPatch) -> None:
    result = slot_engine.validate_slot_selection(
        doctor_id="doctor-1",
        specialty_id="spec-cardio",
        start_at="2026-04-13T14:00:00",
        end_at="2026-04-13T14:30:00",
    )

    assert result is not None
    assert "invalid" in result.lower()


def test_validate_slot_rejects_end_before_start(monkeypatch: MonkeyPatch) -> None:
    result = slot_engine.validate_slot_selection(
        doctor_id="doctor-1",
        specialty_id="spec-cardio",
        start_at="2026-04-13T15:00:00+00:00",
        end_at="2026-04-13T14:00:00+00:00",
    )

    assert result is not None
    assert "invalid" in result.lower()


def test_validate_slot_rejects_past_slot(monkeypatch: MonkeyPatch) -> None:
    _fixed_now(monkeypatch, datetime(2026, 4, 14, 12, 0, tzinfo=timezone.utc))

    result = slot_engine.validate_slot_selection(
        doctor_id="doctor-1",
        specialty_id="spec-cardio",
        start_at="2026-04-13T14:00:00+00:00",
        end_at="2026-04-13T14:30:00+00:00",
    )

    assert result is not None
    assert "no longer bookable" in result.lower()


def test_validate_slot_rejects_missing_doctor(monkeypatch: MonkeyPatch) -> None:
    _fixed_now(monkeypatch, datetime(2026, 4, 11, 12, 0, tzinfo=timezone.utc))
    monkeypatch.setattr(slot_engine, "_get_doctor", lambda _: None)

    result = slot_engine.validate_slot_selection(
        doctor_id="doctor-999",
        specialty_id="spec-cardio",
        start_at="2026-04-13T14:00:00+00:00",
        end_at="2026-04-13T14:30:00+00:00",
    )

    assert result is not None
    assert "doctor is not available" in result.lower()


def test_validate_slot_rejects_missing_specialty(monkeypatch: MonkeyPatch) -> None:
    _fixed_now(monkeypatch, datetime(2026, 4, 11, 12, 0, tzinfo=timezone.utc))
    _stub_doctor(monkeypatch)
    monkeypatch.setattr(slot_engine, "_get_specialty", lambda _: None)

    result = slot_engine.validate_slot_selection(
        doctor_id="doctor-1",
        specialty_id="spec-999",
        start_at="2026-04-13T14:00:00+00:00",
        end_at="2026-04-13T14:30:00+00:00",
    )

    assert result is not None
    assert "specialty is not available" in result.lower()


def test_validate_slot_rejects_doctor_without_specialty(
    monkeypatch: MonkeyPatch,
) -> None:
    _fixed_now(monkeypatch, datetime(2026, 4, 11, 12, 0, tzinfo=timezone.utc))
    _stub_doctor(monkeypatch)
    _stub_specialty(monkeypatch)
    monkeypatch.setattr(slot_engine, "_doctor_has_specialty", lambda *_: False)

    result = slot_engine.validate_slot_selection(
        doctor_id="doctor-1",
        specialty_id="spec-derm",
        start_at="2026-04-13T14:00:00+00:00",
        end_at="2026-04-13T14:30:00+00:00",
    )

    assert result is not None
    assert "not available for the selected specialty" in result.lower()


def test_validate_slot_rejects_booked_slot(monkeypatch: MonkeyPatch) -> None:
    _fixed_now(monkeypatch, datetime(2026, 4, 11, 12, 0, tzinfo=timezone.utc))
    _stub_doctor(monkeypatch)
    _stub_specialty(monkeypatch)
    _stub_doctor_has_specialty(monkeypatch)
    _stub_templates(monkeypatch, [_monday_template()])

    # Slot is booked
    monkeypatch.setattr(
        slot_engine,
        "_get_booked_slots",
        lambda *_, **__: {
            (
                datetime(2026, 4, 13, 14, 0, tzinfo=timezone.utc),
                datetime(2026, 4, 13, 14, 30, tzinfo=timezone.utc),
            )
        },
    )
    monkeypatch.setattr(slot_engine, "_get_doctor_blocks", lambda *_, **__: [])

    result = slot_engine.validate_slot_selection(
        doctor_id="doctor-1",
        specialty_id="spec-cardio",
        start_at="2026-04-13T14:00:00+00:00",
        end_at="2026-04-13T14:30:00+00:00",
    )

    assert result is not None
    assert "no longer available" in result.lower()


def test_validate_slot_rejects_blocked_slot(monkeypatch: MonkeyPatch) -> None:
    _fixed_now(monkeypatch, datetime(2026, 4, 11, 12, 0, tzinfo=timezone.utc))
    _stub_doctor(monkeypatch)
    _stub_specialty(monkeypatch)
    _stub_doctor_has_specialty(monkeypatch)
    _stub_templates(monkeypatch, [_monday_template()])
    monkeypatch.setattr(slot_engine, "_get_booked_slots", lambda *_, **__: set())

    # Doctor has a block covering the whole morning
    monkeypatch.setattr(
        slot_engine,
        "_get_doctor_blocks",
        lambda *_, **__: [
            (
                datetime(2026, 4, 13, 13, 0, tzinfo=timezone.utc),
                datetime(2026, 4, 13, 18, 0, tzinfo=timezone.utc),
            )
        ],
    )

    result = slot_engine.validate_slot_selection(
        doctor_id="doctor-1",
        specialty_id="spec-cardio",
        start_at="2026-04-13T14:00:00+00:00",
        end_at="2026-04-13T14:30:00+00:00",
    )

    assert result is not None
    assert "no longer available" in result.lower()


# ============================================================
# find_slots_for_doctor
# ============================================================


def test_find_slots_for_doctor_returns_empty_when_doctor_missing(
    monkeypatch: MonkeyPatch,
) -> None:
    monkeypatch.setattr(slot_engine, "_get_doctor", lambda _: None)

    result = slot_engine.find_slots_for_doctor(
        doctor_id="doctor-999",
        specialty_id="spec-cardio",
    )

    assert result == []


# ============================================================
# find_slots_for_specialty
# ============================================================


def test_find_slots_for_specialty_returns_empty_when_no_doctors(
    monkeypatch: MonkeyPatch,
) -> None:
    monkeypatch.setattr(slot_engine, "_get_doctors_for_specialty", lambda _: [])

    result = slot_engine.find_slots_for_specialty(specialty_id="spec-cardio")

    assert result == []


# ============================================================
# NEXT_AVAILABLE_ALIASES
# ============================================================


def test_next_available_aliases_are_lowercase() -> None:
    for alias in slot_engine.NEXT_AVAILABLE_ALIASES:
        assert alias == alias.lower(), f"Alias '{alias}' should be lowercase"
