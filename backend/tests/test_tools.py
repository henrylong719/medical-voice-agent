from __future__ import annotations

import json
from datetime import datetime, timezone

import pytest
from pydantic import ValidationError
from pytest import MonkeyPatch

from app.agent import tools
from tests.support import FakeQuery, FakeSupabase


def test_find_patient_by_identifier_returns_patient_details(
    monkeypatch: MonkeyPatch,
) -> None:
    fake_supabase = FakeSupabase(
        tables={
            "patient_identifiers": [
                FakeQuery(
                    [
                        {
                            "patient_id": "patient-1",
                            "identifier_type": "mrn",
                            "identifier_value": "MRN-1001",
                            "issuing_country": None,
                            "is_primary": True,
                        }
                    ]
                )
            ],
            "patients": [
                FakeQuery(
                    [
                        {
                            "id": "patient-1",
                            "full_name": "Sarah Connor",
                            "date_of_birth": "1985-10-26",
                            "phone": "555-0100",
                            "email": "sarah@example.com",
                            "allergies": ["penicillin", "latex"],
                        }
                    ]
                ),
            ]
        }
    )
    monkeypatch.setattr(tools, "supabase", fake_supabase)

    result = tools.find_patient_by_identifier.invoke(
        {"identifier_type": "mrn", "identifier_value": "MRN-1001"}
    )
    payload = json.loads(result)

    assert payload["status"] == "single_match"
    assert payload["patient"]["patient_id"] == "patient-1"
    assert payload["patient"]["full_name"] == "Sarah Connor"
    assert payload["patient"]["date_of_birth"] == "1985-10-26"
    assert payload["patient"]["phone_last4"] == "0100"
    assert "Ask for explicit confirmation" in payload["message"]
    assert "differs from what the patient said earlier" in payload["message"]
    assert payload["matched_identifier"]["identifier_type"] == "mrn"
    assert payload["matched_identifier"]["identifier_value_masked"].endswith("1001")


def test_find_patient_by_identifier_returns_not_found_message(
    monkeypatch: MonkeyPatch,
) -> None:
    fake_supabase = FakeSupabase(tables={"patient_identifiers": [FakeQuery([])]})
    monkeypatch.setattr(tools, "supabase", fake_supabase)

    result = tools.find_patient_by_identifier.invoke(
        {"identifier_type": "mrn", "identifier_value": "MRN-404"}
    )
    payload = json.loads(result)

    assert payload["status"] == "no_match"
    assert "No patient matched that identifier" in payload["message"]
    assert "name, date of birth, and phone" in payload["message"]


def test_find_patients_by_demographics_returns_multiple_matches(
    monkeypatch: MonkeyPatch,
) -> None:
    fake_supabase = FakeSupabase(
        tables={
            "patients": [
                FakeQuery(
                    [
                        {
                            "id": "patient-1",
                            "full_name": "Alex Kim",
                            "date_of_birth": "1990-01-02",
                            "phone": "555-0100",
                            "email": None,
                            "allergies": None,
                        },
                        {
                            "id": "patient-2",
                            "full_name": "Alex Kim",
                            "date_of_birth": "1990-01-02",
                            "phone": "555-0101",
                            "email": None,
                            "allergies": None,
                        },
                    ]
                )
            ]
        }
    )
    monkeypatch.setattr(tools, "supabase", fake_supabase)

    result = tools.find_patients_by_demographics.invoke(
        {
            "full_name": "Alex Kim",
            "date_of_birth": "1990-01-02",
        }
    )
    payload = json.loads(result)

    assert payload["status"] == "multiple_matches"
    assert payload["match_count"] == 2
    assert "Ask for a phone number first" in payload["message"]


def test_find_patients_by_demographics_normalizes_mm_dd_yyyy(
    monkeypatch: MonkeyPatch,
) -> None:
    lookup_query = FakeQuery(
        [
            {
                "id": "patient-1",
                "full_name": "Sarah Connor",
                "date_of_birth": "1985-10-26",
                "phone": "555-0100",
                "email": None,
                "allergies": None,
            }
        ]
    )
    fake_supabase = FakeSupabase(tables={"patients": [lookup_query]})
    monkeypatch.setattr(tools, "supabase", fake_supabase)

    result = tools.find_patients_by_demographics.invoke(
        {
            "full_name": "Sarah Connor",
            "date_of_birth": "10-26-1985",
        }
    )
    payload = json.loads(result)

    assert ("eq", ("date_of_birth", "1985-10-26"), {}) in lookup_query.operations
    assert payload["status"] == "single_match"
    assert payload["patient"]["date_of_birth"] == "1985-10-26"


def test_find_patients_by_demographics_normalizes_day_month_year(
    monkeypatch: MonkeyPatch,
) -> None:
    lookup_query = FakeQuery(
        [
            {
                "id": "patient-1",
                "full_name": "Sarah Connor",
                "date_of_birth": "1985-10-26",
                "phone": "555-0100",
                "email": None,
                "allergies": None,
            }
        ]
    )
    fake_supabase = FakeSupabase(tables={"patients": [lookup_query]})
    monkeypatch.setattr(tools, "supabase", fake_supabase)

    result = tools.find_patients_by_demographics.invoke(
        {
            "full_name": "Sarah Connor",
            "date_of_birth": "26 October 1985",
        }
    )
    payload = json.loads(result)

    assert ("eq", ("date_of_birth", "1985-10-26"), {}) in lookup_query.operations
    assert payload["status"] == "single_match"


def test_find_patients_by_demographics_with_phone_still_requires_stronger_identifier(
    monkeypatch: MonkeyPatch,
) -> None:
    fake_supabase = FakeSupabase(
        tables={
            "patients": [
                FakeQuery(
                    [
                        {
                            "id": "patient-1",
                            "full_name": "Alex Kim",
                            "date_of_birth": "1990-01-02",
                            "phone": "555-0100",
                            "email": None,
                            "allergies": None,
                        },
                        {
                            "id": "patient-2",
                            "full_name": "Alex Kim",
                            "date_of_birth": "1990-01-02",
                            "phone": "555-0100",
                            "email": None,
                            "allergies": None,
                        },
                    ]
                )
            ]
        }
    )
    monkeypatch.setattr(tools, "supabase", fake_supabase)

    result = tools.find_patients_by_demographics.invoke(
        {
            "full_name": "Alex Kim",
            "date_of_birth": "1990-01-02",
            "phone": "555-0100",
        }
    )
    payload = json.loads(result)

    assert payload["status"] == "multiple_matches"
    assert payload["match_count"] == 2
    assert "Ask for a stronger identifier" in payload["message"]
    assert "Do not guess" in payload["message"]


def test_find_patients_by_demographics_rejects_invalid_birth_date(
    monkeypatch: MonkeyPatch,
) -> None:
    fake_supabase = FakeSupabase(tables={"patients": [FakeQuery([])]})
    monkeypatch.setattr(tools, "supabase", fake_supabase)

    result = tools.find_patients_by_demographics.invoke(
        {
            "full_name": "Sarah Connor",
            "date_of_birth": "26-26-1985",
        }
    )
    payload = json.loads(result)

    assert payload["status"] == "error"
    assert "valid date of birth" in payload["message"]


def test_find_patients_by_demographics_returns_no_match_guidance(
    monkeypatch: MonkeyPatch,
) -> None:
    fake_supabase = FakeSupabase(tables={"patients": [FakeQuery([])]})
    monkeypatch.setattr(tools, "supabase", fake_supabase)

    result = tools.find_patients_by_demographics.invoke(
        {
            "full_name": "Sarah Connor",
            "date_of_birth": "1985-10-26",
        }
    )
    payload = json.loads(result)

    assert payload["status"] == "no_match"
    assert "Ask for a stronger identifier" in payload["message"]
    assert "clinic patient number" in payload["message"]
    assert "offer registration" in payload["message"]


def test_find_patient_by_identifier_returns_multiple_matches_requires_staff_help(
    monkeypatch: MonkeyPatch,
) -> None:
    fake_supabase = FakeSupabase(
        tables={
            "patient_identifiers": [
                FakeQuery(
                    [
                        {
                            "patient_id": "patient-1",
                            "identifier_type": "mrn",
                            "identifier_value": "MRN-1001",
                            "issuing_country": None,
                            "is_primary": True,
                        },
                        {
                            "patient_id": "patient-2",
                            "identifier_type": "mrn",
                            "identifier_value": "MRN-1001",
                            "issuing_country": None,
                            "is_primary": True,
                        },
                    ]
                )
            ],
            "patients": [
                FakeQuery(
                    [
                        {
                            "id": "patient-1",
                            "full_name": "Alex Kim",
                            "date_of_birth": "1990-01-02",
                            "phone": "555-0100",
                            "email": None,
                            "allergies": None,
                        }
                    ]
                ),
                FakeQuery(
                    [
                        {
                            "id": "patient-2",
                            "full_name": "Alex Kim",
                            "date_of_birth": "1990-01-02",
                            "phone": "555-0101",
                            "email": None,
                            "allergies": None,
                        }
                    ]
                ),
            ],
        }
    )
    monkeypatch.setattr(tools, "supabase", fake_supabase)

    result = tools.find_patient_by_identifier.invoke(
        {"identifier_type": "mrn", "identifier_value": "MRN-1001"}
    )
    payload = json.loads(result)

    assert payload["status"] == "multiple_matches"
    assert payload["match_count"] == 2
    assert "Hand off to staff" in payload["message"]


def test_register_patient_requires_phone_argument() -> None:
    with pytest.raises(ValidationError, match="phone"):
        tools.register_patient.invoke(
            {"full_name": "Sarah Connor", "date_of_birth": "1985-10-26"}
        )


def test_register_patient_rejects_blank_phone(monkeypatch: MonkeyPatch) -> None:
    fake_supabase = FakeSupabase(tables={"patients": [FakeQuery([])]})
    monkeypatch.setattr(tools, "supabase", fake_supabase)

    result = tools.register_patient.invoke(
        {
            "full_name": "Sarah Connor",
            "date_of_birth": "1985-10-26",
            "phone": "   ",
        }
    )
    payload = json.loads(result)

    assert payload["status"] == "error"
    assert "Phone number missing" in payload["message"]


def test_register_patient_normalizes_slash_birth_date_before_insert(
    monkeypatch: MonkeyPatch,
) -> None:
    lookup_query = FakeQuery([])
    insert_query = FakeQuery(
        [
            {
                "id": "patient-2",
                "full_name": "Sarah Connor",
                "date_of_birth": "1985-10-26",
                "phone": "555-0100",
                "email": None,
                "allergies": None,
            }
        ]
    )
    fake_supabase = FakeSupabase(tables={"patients": [lookup_query, insert_query]})
    monkeypatch.setattr(tools, "supabase", fake_supabase)

    result = tools.register_patient.invoke(
        {
            "full_name": "Sarah Connor",
            "date_of_birth": "10/26/1985",
            "phone": "555-0100",
        }
    )
    payload = json.loads(result)

    assert ("eq", ("date_of_birth", "1985-10-26"), {}) in lookup_query.operations
    assert insert_query.insert_payloads == [
        {
            "full_name": "Sarah Connor",
            "date_of_birth": "1985-10-26",
            "phone": "555-0100",
        }
    ]
    assert payload["status"] == "registered"


def test_register_patient_normalizes_month_name_birth_date(
    monkeypatch: MonkeyPatch,
) -> None:
    lookup_query = FakeQuery([])
    insert_query = FakeQuery(
        [
            {
                "id": "patient-2",
                "full_name": "Sarah Connor",
                "date_of_birth": "1985-10-26",
                "phone": "555-0100",
                "email": None,
                "allergies": None,
            }
        ]
    )
    fake_supabase = FakeSupabase(tables={"patients": [lookup_query, insert_query]})
    monkeypatch.setattr(tools, "supabase", fake_supabase)

    result = tools.register_patient.invoke(
        {
            "full_name": "Sarah Connor",
            "date_of_birth": "October 26, 1985",
            "phone": "555-0100",
        }
    )
    payload = json.loads(result)

    assert insert_query.insert_payloads == [
        {
            "full_name": "Sarah Connor",
            "date_of_birth": "1985-10-26",
            "phone": "555-0100",
        }
    ]
    assert payload["status"] == "registered"


def test_register_patient_normalizes_day_month_year_birth_date(
    monkeypatch: MonkeyPatch,
) -> None:
    lookup_query = FakeQuery([])
    insert_query = FakeQuery(
        [
            {
                "id": "patient-2",
                "full_name": "Sarah Connor",
                "date_of_birth": "1985-10-26",
                "phone": "555-0100",
                "email": None,
                "allergies": None,
            }
        ]
    )
    fake_supabase = FakeSupabase(tables={"patients": [lookup_query, insert_query]})
    monkeypatch.setattr(tools, "supabase", fake_supabase)

    result = tools.register_patient.invoke(
        {
            "full_name": "Sarah Connor",
            "date_of_birth": "26 Oct 1985",
            "phone": "555-0100",
        }
    )
    payload = json.loads(result)

    assert insert_query.insert_payloads == [
        {
            "full_name": "Sarah Connor",
            "date_of_birth": "1985-10-26",
            "phone": "555-0100",
        }
    ]
    assert payload["status"] == "registered"


def test_register_patient_rejects_existing_exact_match(
    monkeypatch: MonkeyPatch,
) -> None:
    fake_supabase = FakeSupabase(
        tables={
            "patients": [
                FakeQuery(
                    [
                        {
                            "id": "patient-1",
                            "full_name": "Sarah Connor",
                            "date_of_birth": "1985-10-26",
                            "phone": "555-0100",
                            "email": None,
                            "allergies": None,
                        }
                    ]
                )
            ]
        }
    )
    monkeypatch.setattr(tools, "supabase", fake_supabase)

    result = tools.register_patient.invoke(
        {
            "full_name": "Sarah Connor",
            "date_of_birth": "1985-10-26",
            "phone": "555-0100",
        }
    )
    payload = json.loads(result)

    assert payload["status"] == "already_exists"
    assert payload["patient"]["patient_id"] == "patient-1"


def test_register_patient_inserts_phone_when_provided(
    monkeypatch: MonkeyPatch,
) -> None:
    lookup_query = FakeQuery([])
    insert_query = FakeQuery(
        [
            {
                "id": "patient-2",
                "full_name": "Sarah Connor",
                "date_of_birth": "1985-10-26",
                "phone": "555-0100",
                "email": None,
                "allergies": None,
            }
        ]
    )
    fake_supabase = FakeSupabase(
        tables={"patients": [lookup_query, insert_query]}
    )
    monkeypatch.setattr(tools, "supabase", fake_supabase)

    result = tools.register_patient.invoke(
        {
            "full_name": "Sarah Connor",
            "date_of_birth": "1985-10-26",
            "phone": "555-0100",
        }
    )
    payload = json.loads(result)

    assert insert_query.insert_payloads == [
        {
            "full_name": "Sarah Connor",
            "date_of_birth": "1985-10-26",
            "phone": "555-0100",
        }
    ]
    assert payload["status"] == "registered"
    assert payload["patient"]["patient_id"] == "patient-2"


def test_register_patient_accepts_any_non_empty_phone_string(
    monkeypatch: MonkeyPatch,
) -> None:
    lookup_query = FakeQuery([])
    insert_query = FakeQuery(
        [
            {
                "id": "patient-2",
                "full_name": "Sarah Connor",
                "date_of_birth": "1985-10-26",
                "phone": "1234",
                "email": None,
                "allergies": None,
            }
        ]
    )
    fake_supabase = FakeSupabase(
        tables={"patients": [lookup_query, insert_query]}
    )
    monkeypatch.setattr(tools, "supabase", fake_supabase)

    result = tools.register_patient.invoke(
        {
            "full_name": "Sarah Connor",
            "date_of_birth": "1985-10-26",
            "phone": "1234",
        }
    )
    payload = json.loads(result)

    assert insert_query.insert_payloads == [
        {
            "full_name": "Sarah Connor",
            "date_of_birth": "1985-10-26",
            "phone": "1234",
        }
    ]
    assert payload["status"] == "registered"
    assert payload["patient"]["phone_last4"] == "1234"


def test_triage_symptoms_requires_input() -> None:
    result = tools.triage_symptoms.invoke({"symptoms": [], "description": ""})

    assert result == (
        "No symptoms provided. Please ask the patient to describe their symptoms."
    )


def test_triage_symptoms_combines_keyword_and_semantic_matches(
    monkeypatch,
) -> None:
    fake_supabase = FakeSupabase(
        tables={
            "symptom_specialty_map": [
                FakeQuery(
                    [
                        {
                            "symptom": "chest pain",
                            "weight": 4,
                            "follow_up_questions": [
                                "Does it spread to your arm?",
                                "Is it worse with exertion?",
                            ],
                            "specialty_id": "spec-cardio",
                            "specialties": {"name": "Cardiology"},
                        }
                    ]
                ),
                FakeQuery(
                    [
                        {
                            "symptom": "shortness of breath",
                            "weight": 3,
                            "follow_up_questions": [
                                "Is it worse with exertion?",
                                "Do you feel it at rest?",
                            ],
                            "specialty_id": "spec-cardio",
                            "specialties": {"name": "Cardiology"},
                        }
                    ]
                ),
            ]
        }
    )
    monkeypatch.setattr(tools, "supabase", fake_supabase)
    monkeypatch.setattr(
        tools,
        "retrieve_medical_knowledge",
        lambda **_: [
            {
                "id": "chunk-1",
                "content": "Chest pressure with exertion often warrants cardiology follow-up.",
                "metadata": {
                    "specialty_name": "Cardiology",
                    "specialty_id": "spec-cardio",
                    "category": "symptoms",
                },
                "similarity": 0.91,
            }
        ],
    )

    result = tools.triage_symptoms.invoke(
        {
            "symptoms": ["chest pain", "shortness of breath"],
            "description": "It feels like an elephant is sitting on my chest.",
        }
    )

    assert result.startswith(
        "Hybrid triage results — use BOTH keyword and semantic matches"
    )
    assert "=== Keyword Matches (from symptom database) ===" in result
    assert "Cardiology (ID: spec-cardio): score 7.00" in result
    assert "Follow-up questions: Does it spread to your arm?; Is it worse with exertion?" in result
    assert "=== Semantic Matches (from medical knowledge base) ===" in result
    assert "similarity 0.91" in result


def test_triage_symptoms_returns_no_match_guidance(monkeypatch: MonkeyPatch) -> None:
    fake_supabase = FakeSupabase(
        tables={"symptom_specialty_map": [FakeQuery([])]}
    )
    monkeypatch.setattr(tools, "supabase", fake_supabase)
    monkeypatch.setattr(tools, "retrieve_medical_knowledge", lambda **_: [])

    result = tools.triage_symptoms.invoke(
        {"symptoms": ["fatigue"], "description": ""}
    )

    assert "No specialty matches found for symptoms: fatigue." in result
    assert "use list_specialties to show available options" in result


def test_find_slots_formats_available_results(monkeypatch: MonkeyPatch) -> None:
    captured: dict[str, object] = {}

    def fake_find_slots_for_specialty(**kwargs):
        captured.update(kwargs)
        return [
            {
                "doctor_id": "doctor-1",
                "doctor_name": "Chen",
                "specialty_id": "spec-cardio",
                "specialty_name": "Cardiology",
                "start_at": "2026-04-13T14:00:00+00:00",
                "end_at": "2026-04-13T14:30:00+00:00",
                "label": "Monday, April 13th at 9:00 AM",
                "date_label": "Monday, April 13th",
            },
            {
                "doctor_id": "doctor-2",
                "doctor_name": "Patel",
                "specialty_id": "spec-cardio",
                "specialty_name": "Cardiology",
                "start_at": "2026-04-13T15:00:00+00:00",
                "end_at": "2026-04-13T15:30:00+00:00",
                "label": "Monday, April 13th at 10:00 AM",
                "date_label": "Monday, April 13th",
            },
        ]

    monkeypatch.setattr(tools, "find_slots_for_specialty", fake_find_slots_for_specialty)

    result = tools.find_slots.invoke(
        {
            "specialty_id": "spec-cardio",
            "preferred_day": "next monday",
            "preferred_time": "morning",
        }
    )

    assert captured == {
        "specialty_id": "spec-cardio",
        "preferred_day": "next monday",
        "preferred_time": "morning",
        "max_results": 20,
    }
    assert "Found 2 available slot(s):" in result
    assert "1. Chen — Monday, April 13th at 9:00 AM" in result
    assert "doctor_id: doctor-2" in result


def test_find_slots_returns_preference_specific_empty_message(
    monkeypatch: MonkeyPatch,
) -> None:
    monkeypatch.setattr(tools, "find_slots_for_specialty", lambda **_: [])

    result = tools.find_slots.invoke(
        {
            "specialty_id": "spec-cardio",
            "preferred_day": "tomorrow",
            "preferred_time": "morning",
        }
    )

    assert result == (
        "No available slots found on tomorrow in the morning. "
        "Try a different day or time, or ask if the patient is flexible."
    )


def test_book_appointment_inserts_scheduled_visit(monkeypatch: MonkeyPatch) -> None:
    insert_query = FakeQuery([{"id": "appt-1"}])
    fake_supabase = FakeSupabase(tables={"appointments": [insert_query]})
    monkeypatch.setattr(tools, "supabase", fake_supabase)
    monkeypatch.setattr(tools, "validate_slot_selection", lambda **_: None)
    monkeypatch.setattr(
        tools,
        "format_for_voice",
        lambda _: "Monday, April 13th at 9:00 AM",
    )

    result = tools.book_appointment.invoke(
        {
            "patient_id": "patient-1",
            "doctor_id": "doctor-1",
            "specialty_id": "spec-cardio",
            "start_at": "2026-04-13T14:00:00+00:00",
            "end_at": "2026-04-13T14:30:00+00:00",
            "reason": "chest pain",
        }
    )

    assert insert_query.insert_payloads == [
        {
            "patient_id": "patient-1",
            "doctor_id": "doctor-1",
            "specialty_id": "spec-cardio",
            "start_at": "2026-04-13T14:00:00+00:00",
            "end_at": "2026-04-13T14:30:00+00:00",
            "status": "scheduled",
            "reason": "chest pain",
        }
    ]
    assert result == (
        "Appointment booked successfully! "
        "Appointment ID: appt-1. "
        "Scheduled for Monday, April 13th at 9:00 AM."
    )


def test_book_appointment_handles_insert_failure(monkeypatch: MonkeyPatch) -> None:
    fake_supabase = FakeSupabase(
        tables={"appointments": [FakeQuery([])]}
    )
    monkeypatch.setattr(tools, "supabase", fake_supabase)
    monkeypatch.setattr(tools, "validate_slot_selection", lambda **_: None)

    result = tools.book_appointment.invoke(
        {
            "patient_id": "patient-1",
            "doctor_id": "doctor-1",
            "specialty_id": "spec-cardio",
            "start_at": "2026-04-13T14:00:00+00:00",
            "end_at": "2026-04-13T14:30:00+00:00",
        }
    )

    assert result == "Failed to book appointment. Please try again."


def test_book_appointment_rejects_stale_slot_before_insert(
    monkeypatch: MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        tools,
        "validate_slot_selection",
        lambda **_: "That slot is no longer available. Please choose another available time.",
    )

    result = tools.book_appointment.invoke(
        {
            "patient_id": "patient-1",
            "doctor_id": "doctor-1",
            "specialty_id": "spec-cardio",
            "start_at": "2026-04-13T14:00:00+00:00",
            "end_at": "2026-04-13T14:30:00+00:00",
        }
    )

    assert result == (
        "That slot is no longer available. Please choose another available time."
    )


def test_find_appointment_filters_by_doctor_name(monkeypatch: MonkeyPatch) -> None:
    fake_supabase = FakeSupabase(
        tables={
            "appointments": [
                FakeQuery(
                    [
                        {
                            "id": "appt-1",
                            "start_at": "2026-04-13T14:00:00+00:00",
                            "end_at": "2026-04-13T14:30:00+00:00",
                            "status": "scheduled",
                            "reason": "follow-up",
                            "doctors": {"full_name": "Dr. Maya Chen"},
                            "specialties": {"name": "Cardiology"},
                        },
                        {
                            "id": "appt-2",
                            "start_at": "2026-04-14T14:00:00+00:00",
                            "end_at": "2026-04-14T14:30:00+00:00",
                            "status": "scheduled",
                            "reason": None,
                            "doctors": {"full_name": "Dr. Alex Patel"},
                            "specialties": {"name": "Dermatology"},
                        },
                    ]
                )
            ]
        }
    )
    monkeypatch.setattr(tools, "supabase", fake_supabase)
    monkeypatch.setattr(
        tools,
        "format_for_voice",
        lambda value: f"formatted:{value}",
    )

    result = tools.find_appointment.invoke(
        {"patient_id": "patient-1", "doctor_name": "chen"}
    )

    assert "Found 1 upcoming appointment(s):" in result
    assert "Dr. Maya Chen (Cardiology) — formatted:2026-04-13T14:00:00+00:00" in result
    assert "Reason: follow-up" in result
    assert "Dr. Alex Patel" not in result


def test_find_appointment_returns_empty_message(monkeypatch: MonkeyPatch) -> None:
    fake_supabase = FakeSupabase(
        tables={"appointments": [FakeQuery([])]}
    )
    monkeypatch.setattr(tools, "supabase", fake_supabase)

    result = tools.find_appointment.invoke({"patient_id": "patient-1"})

    assert result == "No upcoming appointments found for this patient."


def test_find_appointment_only_queries_upcoming_visits(
    monkeypatch: MonkeyPatch,
) -> None:
    query = FakeQuery([])
    fake_supabase = FakeSupabase(tables={"appointments": [query]})
    monkeypatch.setattr(tools, "supabase", fake_supabase)
    monkeypatch.setattr(
        tools,
        "now_utc",
        lambda: datetime(2026, 4, 11, 12, 0, tzinfo=timezone.utc),
    )

    tools.find_appointment.invoke({"patient_id": "patient-1"})

    assert (
        "gte",
        ("start_at", "2026-04-11T12:00:00+00:00"),
        {},
    ) in query.operations


def test_find_appointment_filters_by_specialty(monkeypatch: MonkeyPatch) -> None:
    fake_supabase = FakeSupabase(
        tables={
            "appointments": [
                FakeQuery(
                    [
                        {
                            "id": "appt-1",
                            "start_at": "2026-04-13T14:00:00+00:00",
                            "end_at": "2026-04-13T14:30:00+00:00",
                            "status": "scheduled",
                            "reason": "follow-up",
                            "doctors": {"full_name": "Dr. Maya Chen"},
                            "specialties": {"name": "Cardiology"},
                        },
                        {
                            "id": "appt-2",
                            "start_at": "2026-04-14T14:00:00+00:00",
                            "end_at": "2026-04-14T14:30:00+00:00",
                            "status": "scheduled",
                            "reason": None,
                            "doctors": {"full_name": "Dr. Alex Patel"},
                            "specialties": {"name": "Dermatology"},
                        },
                    ]
                )
            ]
        }
    )
    monkeypatch.setattr(tools, "supabase", fake_supabase)
    monkeypatch.setattr(
        tools,
        "format_for_voice",
        lambda value: f"formatted:{value}",
    )

    result = tools.find_appointment.invoke(
        {"patient_id": "patient-1", "specialty_name": "cardio"}
    )

    assert "Found 1 upcoming appointment(s):" in result
    assert "Dr. Maya Chen (Cardiology) — formatted:2026-04-13T14:00:00+00:00" in result
    assert "Dr. Alex Patel" not in result


def test_reschedule_appointment_previews_without_cancelling(
    monkeypatch: MonkeyPatch,
) -> None:
    lookup_query = FakeQuery(
        [
            {
                "id": "appt-1",
                "patient_id": "patient-1",
                "doctor_id": "doctor-1",
                "specialty_id": "spec-cardio",
                "start_at": "2026-04-13T14:00:00+00:00",
                "end_at": "2026-04-13T14:30:00+00:00",
                "status": "scheduled",
                "reason": "follow-up",
                "doctors": {"full_name": "Maya Chen"},
                "specialties": {"name": "Cardiology"},
            }
        ]
    )
    fake_supabase = FakeSupabase(
        tables={"appointments": [lookup_query]}
    )
    monkeypatch.setattr(tools, "supabase", fake_supabase)
    monkeypatch.setattr(
        tools,
        "format_for_voice",
        lambda _: "Monday, April 13th at 9:00 AM",
    )
    monkeypatch.setattr(
        tools,
        "find_slots_for_doctor",
        lambda **_: [
            {
                "doctor_id": "doctor-1",
                "doctor_name": "Maya Chen",
                "specialty_id": "spec-cardio",
                "specialty_name": "Cardiology",
                "start_at": "2026-04-20T14:00:00+00:00",
                "end_at": "2026-04-20T14:30:00+00:00",
                "label": "Monday, April 20th at 9:00 AM",
                "date_label": "Monday, April 20th",
            }
        ],
    )

    result = tools.reschedule_appointment.invoke(
        {
            "appointment_id": "appt-1",
            "patient_id": "patient-1",
            "preferred_day": "next week",
        }
    )

    assert "Current appointment: Dr. Maya Chen (Cardiology)" in result
    assert "The current appointment has NOT been cancelled." in result
    assert "Here are 1 alternative slot(s) with the same doctor for next week:" in result
    assert "1. Monday, April 20th at 9:00 AM" in result
    assert "specialty_id: spec-cardio" in result
    assert ("eq", ("patient_id", "patient-1"), {}) in lookup_query.operations


def test_reschedule_appointment_preview_mentions_filters_and_avoids_double_dr(
    monkeypatch: MonkeyPatch,
) -> None:
    lookup_query = FakeQuery(
        [
            {
                "id": "appt-1",
                "patient_id": "patient-1",
                "doctor_id": "doctor-1",
                "specialty_id": "spec-cardio",
                "start_at": "2026-04-13T14:00:00+00:00",
                "end_at": "2026-04-13T14:30:00+00:00",
                "status": "scheduled",
                "reason": "follow-up",
                "doctors": {"full_name": "Dr. Maya Chen"},
                "specialties": {"name": "Cardiology"},
            }
        ]
    )
    fake_supabase = FakeSupabase(
        tables={"appointments": [lookup_query]}
    )
    monkeypatch.setattr(tools, "supabase", fake_supabase)
    monkeypatch.setattr(
        tools,
        "format_for_voice",
        lambda _: "Monday, April 13th at 9:00 AM",
    )
    monkeypatch.setattr(tools, "find_slots_for_doctor", lambda **_: [])

    result = tools.reschedule_appointment.invoke(
        {
            "appointment_id": "appt-1",
            "patient_id": "patient-1",
            "preferred_day": "next week",
            "preferred_time": "afternoon",
        }
    )

    assert "Current appointment: Dr. Maya Chen (Cardiology)" in result
    assert "Dr. Dr. Maya Chen" not in result
    assert "Search criteria: same doctor for next week in the afternoon." in result
    assert "No available slots found with the same doctor for next week in the afternoon." in result


def test_reschedule_appointment_finalizes_confirmed_slot(
    monkeypatch: MonkeyPatch,
) -> None:
    lookup_query = FakeQuery(
        [
            {
                "id": "appt-1",
                "patient_id": "patient-1",
                "doctor_id": "doctor-1",
                "specialty_id": "spec-cardio",
                "start_at": "2026-04-13T14:00:00+00:00",
                "end_at": "2026-04-13T14:30:00+00:00",
                "status": "scheduled",
                "reason": "follow-up",
                "doctors": {"full_name": "Maya Chen"},
                "specialties": {"name": "Cardiology"},
            }
        ]
    )
    rpc_query = FakeQuery({"status": "ok", "appointment_id": "appt-1"})
    fake_supabase = FakeSupabase(
        tables={"appointments": [lookup_query]},
        rpcs={"finalize_reschedule_appointment": [rpc_query]},
    )
    monkeypatch.setattr(tools, "supabase", fake_supabase)
    monkeypatch.setattr(tools, "validate_slot_selection", lambda **_: None)
    monkeypatch.setattr(
        tools,
        "format_for_voice",
        lambda value: (
            "Monday, April 13th at 9:00 AM"
            if value == "2026-04-13T14:00:00+00:00"
            else "Monday, April 20th at 9:00 AM"
        ),
    )

    result = tools.reschedule_appointment.invoke(
        {
            "appointment_id": "appt-1",
            "patient_id": "patient-1",
            "new_doctor_id": "doctor-1",
            "new_specialty_id": "spec-cardio",
            "new_start_at": "2026-04-20T14:00:00+00:00",
            "new_end_at": "2026-04-20T14:30:00+00:00",
        }
    )

    assert ("eq", ("patient_id", "patient-1"), {}) in lookup_query.operations
    assert fake_supabase.rpc_calls == [
        (
            "finalize_reschedule_appointment",
            {
                "p_appointment_id": "appt-1",
                "p_patient_id": "patient-1",
                "p_new_doctor_id": "doctor-1",
                "p_new_specialty_id": "spec-cardio",
                "p_new_start_at": "2026-04-20T14:00:00+00:00",
                "p_new_end_at": "2026-04-20T14:30:00+00:00",
                "p_timezone": tools.settings.timezone,
            },
        )
    ]
    assert "Appointment rescheduled successfully!" in result
    assert "Old appointment on Monday, April 13th at 9:00 AM" in result
    assert "Appointment ID: appt-1." in result


def test_reschedule_appointment_handles_rpc_slot_race(
    monkeypatch: MonkeyPatch,
) -> None:
    lookup_query = FakeQuery(
        [
            {
                "id": "appt-1",
                "patient_id": "patient-1",
                "doctor_id": "doctor-1",
                "specialty_id": "spec-cardio",
                "start_at": "2026-04-13T14:00:00+00:00",
                "end_at": "2026-04-13T14:30:00+00:00",
                "status": "scheduled",
                "reason": "follow-up",
                "doctors": {"full_name": "Maya Chen"},
                "specialties": {"name": "Cardiology"},
            }
        ]
    )
    rpc_query = FakeQuery({"status": "slot_unavailable"})
    fake_supabase = FakeSupabase(
        tables={"appointments": [lookup_query]},
        rpcs={"finalize_reschedule_appointment": [rpc_query]},
    )
    monkeypatch.setattr(tools, "supabase", fake_supabase)
    monkeypatch.setattr(tools, "validate_slot_selection", lambda **_: None)

    result = tools.reschedule_appointment.invoke(
        {
            "appointment_id": "appt-1",
            "patient_id": "patient-1",
            "new_doctor_id": "doctor-1",
            "new_specialty_id": "spec-cardio",
            "new_start_at": "2026-04-20T14:00:00+00:00",
            "new_end_at": "2026-04-20T14:30:00+00:00",
        }
    )

    assert result == (
        "That slot is no longer available. Please choose another available "
        "time. The original appointment was kept."
    )


def test_reschedule_appointment_keeps_original_when_new_slot_is_invalid(
    monkeypatch: MonkeyPatch,
) -> None:
    lookup_query = FakeQuery(
        [
            {
                "id": "appt-1",
                "patient_id": "patient-1",
                "doctor_id": "doctor-1",
                "specialty_id": "spec-cardio",
                "start_at": "2026-04-13T14:00:00+00:00",
                "end_at": "2026-04-13T14:30:00+00:00",
                "status": "scheduled",
                "reason": "follow-up",
                "doctors": {"full_name": "Maya Chen"},
                "specialties": {"name": "Cardiology"},
            }
        ]
    )
    fake_supabase = FakeSupabase(tables={"appointments": [lookup_query]})
    monkeypatch.setattr(tools, "supabase", fake_supabase)
    monkeypatch.setattr(
        tools,
        "validate_slot_selection",
        lambda **_: "That slot is no longer available. Please choose another available time.",
    )

    result = tools.reschedule_appointment.invoke(
        {
            "appointment_id": "appt-1",
            "patient_id": "patient-1",
            "new_doctor_id": "doctor-1",
            "new_specialty_id": "spec-cardio",
            "new_start_at": "2026-04-20T14:00:00+00:00",
            "new_end_at": "2026-04-20T14:30:00+00:00",
        }
    )

    assert result == (
        "That slot is no longer available. Please choose another available time. "
        "The original appointment was kept."
    )


def test_reschedule_appointment_handles_missing_or_cancelled_visits(
    monkeypatch: MonkeyPatch,
) -> None:
    fake_supabase = FakeSupabase(
        tables={"appointments": [FakeQuery([])]}
    )
    monkeypatch.setattr(tools, "supabase", fake_supabase)

    missing = tools.reschedule_appointment.invoke(
        {"appointment_id": "appt-404", "patient_id": "patient-1"}
    )

    assert missing == "Appointment appt-404 not found for this patient."

    fake_supabase = FakeSupabase(
        tables={
            "appointments": [
                FakeQuery(
                    [
                        {
                            "id": "appt-1",
                            "patient_id": "patient-1",
                            "doctor_id": "doctor-1",
                            "specialty_id": "spec-cardio",
                            "start_at": "2026-04-13T14:00:00+00:00",
                            "end_at": "2026-04-13T14:30:00+00:00",
                            "status": "cancelled",
                            "reason": None,
                            "doctors": {"full_name": "Maya Chen"},
                            "specialties": {"name": "Cardiology"},
                        }
                    ]
                )
            ]
        }
    )
    monkeypatch.setattr(tools, "supabase", fake_supabase)

    cancelled = tools.reschedule_appointment.invoke(
        {"appointment_id": "appt-1", "patient_id": "patient-1"}
    )

    assert cancelled == "This appointment is already cancelled."


def test_cancel_appointment_updates_status(monkeypatch: MonkeyPatch) -> None:
    lookup_query = FakeQuery(
        [
            {
                "id": "appt-1",
                "start_at": "2026-04-13T14:00:00+00:00",
                "status": "scheduled",
                "doctors": {"full_name": "Maya Chen"},
                "specialties": {"name": "Cardiology"},
            }
        ]
    )
    update_query = FakeQuery([])
    fake_supabase = FakeSupabase(
        tables={"appointments": [lookup_query, update_query]}
    )
    monkeypatch.setattr(tools, "supabase", fake_supabase)
    monkeypatch.setattr(
        tools,
        "format_for_voice",
        lambda _: "Monday, April 13th at 9:00 AM",
    )

    result = tools.cancel_appointment.invoke(
        {"patient_id": "patient-1", "appointment_id": "appt-1"}
    )

    assert update_query.update_payloads == [{"status": "cancelled"}]
    assert ("eq", ("patient_id", "patient-1"), {}) in lookup_query.operations
    assert ("eq", ("patient_id", "patient-1"), {}) in update_query.operations
    assert result == (
        "Appointment with Dr. Maya Chen (Cardiology) on "
        "Monday, April 13th at 9:00 AM has been cancelled."
    )


def test_cancel_appointment_avoids_double_dr_title(monkeypatch: MonkeyPatch) -> None:
    lookup_query = FakeQuery(
        [
            {
                "id": "appt-1",
                "start_at": "2026-04-13T14:00:00+00:00",
                "status": "scheduled",
                "doctors": {"full_name": "Dr. Maya Chen"},
                "specialties": {"name": "Cardiology"},
            }
        ]
    )
    update_query = FakeQuery([])
    fake_supabase = FakeSupabase(
        tables={"appointments": [lookup_query, update_query]}
    )
    monkeypatch.setattr(tools, "supabase", fake_supabase)
    monkeypatch.setattr(
        tools,
        "format_for_voice",
        lambda _: "Monday, April 13th at 9:00 AM",
    )

    result = tools.cancel_appointment.invoke(
        {"patient_id": "patient-1", "appointment_id": "appt-1"}
    )

    assert "Dr. Dr. Maya Chen" not in result
    assert result == (
        "Appointment with Dr. Maya Chen (Cardiology) on "
        "Monday, April 13th at 9:00 AM has been cancelled."
    )


def test_cancel_appointment_handles_missing_or_cancelled_visits(
    monkeypatch: MonkeyPatch,
) -> None:
    fake_supabase = FakeSupabase(
        tables={"appointments": [FakeQuery([])]}
    )
    monkeypatch.setattr(tools, "supabase", fake_supabase)

    missing = tools.cancel_appointment.invoke(
        {"patient_id": "patient-1", "appointment_id": "appt-404"}
    )

    assert missing == "Appointment appt-404 not found for this patient."

    fake_supabase = FakeSupabase(
        tables={
            "appointments": [
                FakeQuery(
                    [
                        {
                            "id": "appt-1",
                            "start_at": "2026-04-13T14:00:00+00:00",
                            "status": "cancelled",
                            "doctors": {"full_name": "Maya Chen"},
                            "specialties": {"name": "Cardiology"},
                        }
                    ]
                )
            ]
        }
    )
    monkeypatch.setattr(tools, "supabase", fake_supabase)

    cancelled = tools.cancel_appointment.invoke(
        {"patient_id": "patient-1", "appointment_id": "appt-1"}
    )

    assert cancelled == "This appointment is already cancelled."


def test_list_specialties_formats_rows(monkeypatch: MonkeyPatch) -> None:
    fake_supabase = FakeSupabase(
        tables={
            "specialties": [
                FakeQuery(
                    [
                        {
                            "id": "spec-cardio",
                            "name": "Cardiology",
                            "description": "Heart and circulation care",
                        },
                        {
                            "id": "spec-derm",
                            "name": "Dermatology",
                            "description": None,
                        },
                    ]
                )
            ]
        }
    )
    monkeypatch.setattr(tools, "supabase", fake_supabase)

    result = tools.list_specialties.invoke({})

    assert "Available specialties:" in result
    assert "- Cardiology (ID: spec-cardio): Heart and circulation care" in result
    assert "- Dermatology (ID: spec-derm): No description" in result


def test_list_specialties_returns_empty_message(monkeypatch: MonkeyPatch) -> None:
    fake_supabase = FakeSupabase(
        tables={"specialties": [FakeQuery([])]}
    )
    monkeypatch.setattr(tools, "supabase", fake_supabase)

    result = tools.list_specialties.invoke({})

    assert result == "No specialties found."


def test_all_tools_registry_contains_expected_tools() -> None:
    assert [tool.name for tool in tools.ALL_TOOLS] == [
        "find_patient_by_identifier",
        "find_patients_by_demographics",
        "register_patient",
        "triage_symptoms",
        "find_slots",
        "book_appointment",
        "find_appointment",
        "reschedule_appointment",
        "cancel_appointment",
        "list_specialties",
    ]
