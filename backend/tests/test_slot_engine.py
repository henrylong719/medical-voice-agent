from __future__ import annotations

from datetime import datetime, timezone

from pytest import MonkeyPatch

from app.services import slot_engine
from tests.support import FakeQuery, FakeSupabase


def test_get_booked_slots_uses_overlap_query(monkeypatch: MonkeyPatch) -> None:
    query = FakeQuery(
        [
            {
                "start_at": "2026-04-13T13:30:00+00:00",
                "end_at": "2026-04-13T14:30:00+00:00",
            }
        ]
    )
    fake_supabase = FakeSupabase(tables={"appointments": [query]})
    monkeypatch.setattr(slot_engine, "supabase", fake_supabase)

    window_start = datetime(2026, 4, 13, 14, 0, tzinfo=timezone.utc)
    window_end = datetime(2026, 4, 13, 16, 0, tzinfo=timezone.utc)

    result = slot_engine._get_booked_slots("doctor-1", window_start, window_end)

    assert result == {
        (
            datetime(2026, 4, 13, 13, 30, tzinfo=timezone.utc),
            datetime(2026, 4, 13, 14, 30, tzinfo=timezone.utc),
        )
    }
    assert ("lt", ("start_at", window_end.isoformat()), {}) in query.operations
    assert ("gt", ("end_at", window_start.isoformat()), {}) in query.operations
    assert not any(name == "gte" for name, _, _ in query.operations)


def test_validate_slot_selection_accepts_exact_available_slot(
    monkeypatch: MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        slot_engine,
        "now_utc",
        lambda: datetime(2026, 4, 11, 12, 0, tzinfo=timezone.utc),
    )
    monkeypatch.setattr(
        slot_engine,
        "_get_doctor",
        lambda _: {"id": "doctor-1", "full_name": "Dr. Maya Chen", "is_active": True},
    )
    monkeypatch.setattr(
        slot_engine,
        "_get_specialty",
        lambda _: {"id": "spec-cardio", "name": "Cardiology"},
    )
    monkeypatch.setattr(slot_engine, "_doctor_has_specialty", lambda *_: True)
    monkeypatch.setattr(
        slot_engine,
        "_get_availability_templates",
        lambda _: [
            {
                "day_of_week": "monday",
                "start_time": "09:00:00",
                "end_time": "10:00:00",
                "slot_duration_min": 30,
            }
        ],
    )
    monkeypatch.setattr(slot_engine, "_get_booked_slots", lambda *_, **__: set())
    monkeypatch.setattr(slot_engine, "_get_doctor_blocks", lambda *_, **__: [])

    result = slot_engine.validate_slot_selection(
        doctor_id="doctor-1",
        specialty_id="spec-cardio",
        start_at="2026-04-13T14:00:00+00:00",
        end_at="2026-04-13T14:30:00+00:00",
    )

    assert result is None


def test_validate_slot_selection_rejects_non_template_time(
    monkeypatch: MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        slot_engine,
        "now_utc",
        lambda: datetime(2026, 4, 11, 12, 0, tzinfo=timezone.utc),
    )
    monkeypatch.setattr(
        slot_engine,
        "_get_doctor",
        lambda _: {"id": "doctor-1", "full_name": "Dr. Maya Chen", "is_active": True},
    )
    monkeypatch.setattr(
        slot_engine,
        "_get_specialty",
        lambda _: {"id": "spec-cardio", "name": "Cardiology"},
    )
    monkeypatch.setattr(slot_engine, "_doctor_has_specialty", lambda *_: True)
    monkeypatch.setattr(
        slot_engine,
        "_get_availability_templates",
        lambda _: [
            {
                "day_of_week": "monday",
                "start_time": "09:00:00",
                "end_time": "10:00:00",
                "slot_duration_min": 30,
            }
        ],
    )
    monkeypatch.setattr(slot_engine, "_get_booked_slots", lambda *_, **__: set())
    monkeypatch.setattr(slot_engine, "_get_doctor_blocks", lambda *_, **__: [])

    result = slot_engine.validate_slot_selection(
        doctor_id="doctor-1",
        specialty_id="spec-cardio",
        start_at="2026-04-13T14:15:00+00:00",
        end_at="2026-04-13T14:45:00+00:00",
    )

    assert result is not None
    assert "does not match the doctor's current availability" in result
