"""
Tests for PII redaction in LangSmith traces.

Verifies that the redact_pii anonymizer correctly masks sensitive
patient data while preserving non-PII fields and overall structure.
"""

import pytest

from app.agent.pii_redactor import (
    redact_pii,
    _redact_name,
    _redact_dob,
    _redact_phone,
    _redact_identifier,
    _redact_email,
    _redact_text_pii,
)


# ============================================================
# INDIVIDUAL REDACTION FUNCTIONS
# ============================================================


class TestRedactName:
    def test_normal_name(self):
        assert _redact_name("Sarah Johnson") == "S***h J*****n"

    def test_three_part_name(self):
        assert _redact_name("Mary Jane Watson") == "M**y J**e W****n"

    def test_short_name(self):
        assert _redact_name("Li") == "L*"

    def test_single_char(self):
        assert _redact_name("A") == "A"

    def test_empty(self):
        assert _redact_name("") == ""

    def test_preserves_word_count(self):
        original = "John Michael Smith"
        redacted = _redact_name(original)
        assert len(redacted.split()) == 3


class TestRedactDob:
    def test_iso_format(self):
        assert _redact_dob("1990-01-05") == "****-**-05"

    def test_us_format_slash(self):
        result = _redact_dob("01/05/1990")
        assert "1990" in result  # keeps year in US format
        assert result.startswith("**")

    def test_natural_language(self):
        assert _redact_dob("January 5, 1990") == "[REDACTED DOB]"

    def test_empty(self):
        assert _redact_dob("") == ""


class TestRedactPhone:
    def test_dashed_format(self):
        result = _redact_phone("555-012-3456")
        assert result.endswith("56")
        assert result.count("*") >= 8

    def test_parenthesized_format(self):
        result = _redact_phone("(555) 012-3456")
        assert result.endswith("56")
        assert "(" in result  # preserves formatting

    def test_plain_digits(self):
        result = _redact_phone("5550123456")
        assert result.endswith("56")
        assert len(result) == 10

    def test_short_number(self):
        assert _redact_phone("123") == "***"

    def test_empty(self):
        assert _redact_phone("") == ""


class TestRedactIdentifier:
    def test_mrn(self):
        result = _redact_identifier("MRN123456")
        assert result.endswith("3456")
        assert result.startswith("*")

    def test_passport(self):
        result = _redact_identifier("AB1234567")
        assert result.endswith("4567")

    def test_with_formatting(self):
        result = _redact_identifier("DL-987654")
        assert result.endswith("7654")
        assert "-" in result  # preserves dash

    def test_short_value(self):
        assert _redact_identifier("ABC") == "****"

    def test_empty(self):
        assert _redact_identifier("") == ""


class TestRedactEmail:
    def test_normal_email(self):
        result = _redact_email("sarah.johnson@gmail.com")
        assert result.startswith("s***@")
        assert result.endswith(".com")
        assert "sarah" not in result
        assert "gmail" not in result

    def test_short_local(self):
        result = _redact_email("a@example.org")
        assert result.startswith("a***@")
        assert result.endswith(".org")

    def test_not_an_email(self):
        assert _redact_email("not-an-email") == "not-an-email"

    def test_empty(self):
        assert _redact_email("") == ""


# ============================================================
# TEXT-BASED PII REDACTION
# ============================================================


class TestRedactTextPii:
    def test_phone_in_text(self):
        text = "My phone number is 555-012-3456 please call me"
        result = _redact_text_pii(text)
        assert "555" not in result
        assert "56" in result  # last 2 digits kept

    def test_email_in_text(self):
        text = "Send it to sarah@gmail.com thanks"
        result = _redact_text_pii(text)
        assert "sarah" not in result
        assert "@" in result

    def test_iso_date_in_text(self):
        text = "My birthday is 1990-01-05"
        result = _redact_text_pii(text)
        assert "1990" not in result
        assert "05" in result

    def test_us_date_in_text(self):
        text = "DOB is 01/05/1990"
        result = _redact_text_pii(text)
        assert "01/05" not in result

    def test_no_pii_unchanged(self):
        text = "I have a headache and want to see a doctor"
        assert _redact_text_pii(text) == text

    def test_multiple_pii_in_one_text(self):
        text = "Call me at 555-123-4567 or email me at john@test.com"
        result = _redact_text_pii(text)
        assert "555" not in result
        assert "john" not in result


# ============================================================
# FULL redact_pii FUNCTION (dict-level)
# ============================================================


class TestRedactPii:
    def test_tool_call_args_redacted(self):
        """Simulate a find_patients_by_demographics tool call."""
        data = {
            "name": "find_patients_by_demographics",
            "args": {
                "full_name": "Sarah Johnson",
                "date_of_birth": "1990-01-05",
                "phone": "555-012-3456",
            },
        }
        result = redact_pii(data)
        assert result["args"]["full_name"] == "S***h J*****n"
        assert result["args"]["date_of_birth"] == "****-**-05"
        assert "555" not in result["args"]["phone"]

    def test_register_patient_redacted(self):
        """Simulate a register_patient tool call."""
        data = {
            "name": "register_patient",
            "args": {
                "full_name": "John Smith",
                "date_of_birth": "1985-03-15",
                "phone": "555-987-6543",
                "email": "john.smith@email.com",
            },
        }
        result = redact_pii(data)
        assert "John" not in result["args"]["full_name"]
        assert "1985" not in result["args"]["date_of_birth"]
        assert "987" not in result["args"]["phone"]
        assert "john.smith" not in result["args"]["email"]

    def test_identifier_lookup_redacted(self):
        """Simulate a find_patient_by_identifier tool call."""
        data = {
            "args": {
                "identifier_type": "mrn",
                "identifier_value": "MRN123456789",
            },
        }
        result = redact_pii(data)
        assert result["args"]["identifier_type"] == "mrn"  # type not PII
        assert "MRN123" not in result["args"]["identifier_value"]
        assert result["args"]["identifier_value"].endswith("6789")

    def test_non_pii_fields_preserved(self):
        """Non-PII fields should pass through unchanged."""
        data = {
            "name": "find_slots",
            "args": {
                "specialty_id": "abc-123-def",
                "preferred_day": "next tuesday",
                "preferred_time": "morning",
            },
        }
        result = redact_pii(data)
        assert result == data  # nothing should change

    def test_uuid_not_redacted(self):
        """Patient UUIDs are system-internal, not PII."""
        data = {
            "args": {
                "patient_id": "550e8400-e29b-41d4-a716-446655440000",
                "doctor_id": "660e8400-e29b-41d4-a716-446655440001",
            },
        }
        result = redact_pii(data)
        assert result["args"]["patient_id"] == data["args"]["patient_id"]
        assert result["args"]["doctor_id"] == data["args"]["doctor_id"]

    def test_nested_messages_redacted(self):
        """PII in conversation message content should be redacted."""
        data = {
            "messages": [
                {
                    "role": "human",
                    "content": "My name is Sarah and my DOB is 1990-01-05",
                },
                {
                    "role": "assistant",
                    "content": "Thanks Sarah! Let me look you up.",
                },
            ],
        }
        result = redact_pii(data)
        human_msg = result["messages"][0]["content"]
        assert "1990" not in human_msg
        assert "05" in human_msg  # day kept

    def test_does_not_mutate_original(self):
        """redact_pii should return a copy, not modify the input."""
        data = {
            "args": {
                "full_name": "Sarah Johnson",
                "date_of_birth": "1990-01-05",
            },
        }
        redact_pii(data)
        assert data["args"]["full_name"] == "Sarah Johnson"
        assert data["args"]["date_of_birth"] == "1990-01-05"

    def test_empty_dict(self):
        assert redact_pii({}) == {}

    def test_deeply_nested_structure(self):
        """PII should be redacted at any nesting depth."""
        data = {
            "outer": {
                "inner": {
                    "deep": {
                        "full_name": "Deep Patient",
                        "phone": "555-111-2222",
                    }
                }
            }
        }
        result = redact_pii(data)
        assert "Deep" not in result["outer"]["inner"]["deep"]["full_name"]
        assert "555" not in result["outer"]["inner"]["deep"]["phone"]

    def test_list_of_tool_results(self):
        """Tool results sometimes come as lists."""
        data = {
            "results": [
                {"full_name": "Patient One", "phone": "111-222-3333"},
                {"full_name": "Patient Two", "phone": "444-555-6666"},
            ],
        }
        result = redact_pii(data)
        assert "Patient One" not in str(result)
        assert "Patient Two" not in str(result)
        assert "111" not in str(result)
        assert "444" not in str(result)
