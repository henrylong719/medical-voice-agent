"""
PII redaction for LangSmith traces.

Masks sensitive patient data before it leaves the server. This module
provides an ``anonymizer`` function for the LangSmith Client that
redacts personally identifiable information from trace inputs and
outputs.

What gets redacted:
  - Patient names → "S***h J*****n"
  - Dates of birth → "****-**-05" (keeps day for debugging)
  - Phone numbers → "***-**23" (keeps last 2 digits)
  - Strong identifiers (MRN, passport, driver's license, clinic ID)
    → "****7890" (keeps last 4 characters)
  - Email addresses → "s***@***.com"

What is NOT redacted:
  - UUIDs (patient_id, doctor_id, etc.) — these are system-internal
    and not personally identifiable on their own
  - Doctor names — these are public/professional, not patient PII
  - Specialty names, appointment times, symptoms — not PII

Design:
  The anonymizer walks the entire dict structure recursively,
  looking for known PII field names. This is key-based, not
  value-based — we redact based on the FIELD NAME (e.g.,
  "full_name", "date_of_birth") rather than trying to detect
  PII patterns in arbitrary text. Key-based is more reliable
  and avoids false positives on doctor names or dates that
  appear in scheduling context.

  For conversation messages (which contain PII in free text),
  we apply pattern-based redaction to catch phone numbers,
  DOBs, and email addresses mentioned in natural language.

Usage:
  Pass ``redact_pii`` as the ``anonymizer`` argument when
  creating the LangSmith Client:

    from langsmith import Client
    from app.agent.pii_redactor import redact_pii

    client = Client(anonymizer=redact_pii)
"""

from __future__ import annotations

import re
from copy import deepcopy
from typing import Any

from collections.abc import Callable

# ============================================================
# FIELD-BASED REDACTION RULES
# ============================================================
# Maps known PII field names to their redaction functions.
# Each function takes a string value and returns the masked
# version.
# ============================================================


def _redact_name(value: str) -> str:
    """Mask a person's name, keeping first and last character of each part.

    "Sarah Johnson" → "S***h J*****n"
    "Li" → "L*"  (short names get minimal masking)
    """
    if not value or not value.strip():
        return value

    parts = value.split()
    masked_parts: list[str] = []
    for part in parts:
        if len(part) <= 2:
            masked_parts.append(part[0] + "*" * (len(part) - 1))
        else:
            masked_parts.append(part[0] + "*" * (len(part) - 2) + part[-1])
    return " ".join(masked_parts)


def _redact_dob(value: str) -> str:
    """Mask a date of birth, keeping only the day for debugging.

    "1990-01-05" → "****-**-05"
    "01/05/1990" → "**/**/1990" — keeps last segment
    "January 5, 1990" → "[REDACTED DOB]"
    """
    if not value or not value.strip():
        return value

    # ISO format: YYYY-MM-DD
    if re.match(r"^\d{4}-\d{2}-\d{2}$", value.strip()):
        return "****-**-" + value.strip()[-2:]

    # US format: MM/DD/YYYY or MM-DD-YYYY
    if re.match(r"^\d{2}[/-]\d{2}[/-]\d{4}$", value.strip()):
        sep = "/" if "/" in value else "-"
        parts = value.strip().split(sep)
        return f"**{sep}**{sep}{parts[2]}"

    # Natural language or other formats — full redaction
    return "[REDACTED DOB]"


def _redact_phone(value: str) -> str:
    """Mask a phone number, keeping last 2 digits.

    "555-012-3456" → "***-***-**56"
    "(555) 012-3456" → "(***) ***-**56"
    "5550123456" → "********56"
    """
    if not value or not value.strip():
        return value

    # Extract just the digits
    digits = re.sub(r"\D", "", value)
    if len(digits) < 4:
        return "***"

    # Mask all but last 2 digits, preserving the original formatting
    last_two = digits[-2:]
    masked_digits = "*" * (len(digits) - 2) + last_two

    # Rebuild with original formatting
    result: list[str] = []
    digit_index = 0
    for char in value:
        if char.isdigit():
            result.append(masked_digits[digit_index])
            digit_index += 1
        else:
            result.append(char)
    return "".join(result)


def _redact_identifier(value: str) -> str:
    """Mask a strong identifier (MRN, passport, license), keeping last 4 chars.

    "ABC123456" → "*****3456"
    "DL-98765" → "***-*765"  (preserves formatting)
    """
    if not value or not value.strip():
        return value

    if len(value) <= 4:
        return "****"

    # Keep last 4 characters, mask the rest preserving format
    keep = value[-4:]
    masked_prefix: list[str] = []
    for char in value[:-4]:
        if char.isalnum():
            masked_prefix.append("*")
        else:
            masked_prefix.append(char)
    return "".join(masked_prefix) + keep


def _redact_email(value: str) -> str:
    """Mask an email address.

    "sarah.johnson@gmail.com" → "s***@***.com"
    """
    if not value or not value.strip() or "@" not in value:
        return value

    local, domain = value.rsplit("@", 1)
    domain_parts = domain.split(".")

    masked_local = local[0] + "***" if local else "***"
    masked_domain = "***." + domain_parts[-1] if domain_parts else "***"

    return f"{masked_local}@{masked_domain}"


# ── Field name → redaction function mapping ───────────────────
# We check field names case-insensitively and also handle
# nested structures (e.g., tool call args inside messages).

_FIELD_REDACTORS: dict[str, Any] = {
    # Patient name fields
    # Note: we intentionally exclude the generic "name" key here
    # because it collides with tool metadata fields like
    # {"name": "find_slots"}. Our tools use "full_name" and
    # "patient_name" for patient names.
    "full_name": _redact_name,
    "patient_name": _redact_name,
    # Date of birth
    "date_of_birth": _redact_dob,
    "dob": _redact_dob,
    # Phone
    "phone": _redact_phone,
    "phone_number": _redact_phone,
    # Strong identifiers
    "identifier_value": _redact_identifier,
    "mrn": _redact_identifier,
    "passport_number": _redact_identifier,
    "drivers_license": _redact_identifier,
    "external_patient_id": _redact_identifier,
    # Email
    "email": _redact_email,
}


# ============================================================
# PATTERN-BASED REDACTION FOR FREE TEXT
# ============================================================
# Conversation messages contain PII in natural language.
# We can't rely on field names here — the PII is embedded
# in strings like "My name is Sarah Johnson and my DOB is
# January 5, 1990."
#
# These patterns are intentionally conservative. We only
# redact things that look unambiguously like PII. It's
# better to miss some PII in free text (where it's mixed
# with non-sensitive content) than to over-redact and make
# traces unreadable.
# ============================================================

TextReplacer = Callable[[re.Match[str]], str]

_TEXT_PII_PATTERNS: list[tuple[re.Pattern[str], TextReplacer]] = [
    # Phone numbers: (555) 123-4567, 555-123-4567, 555.123.4567, 5551234567
    (
        re.compile(r"\(?\d{3}\)?[\s.\-]?\d{3}[\s.\-]?\d{4}"),
        lambda m: _redact_phone(m.group(0)),
    ),
    # Email addresses
    (
        re.compile(r"[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}"),
        lambda m: _redact_email(m.group(0)),
    ),
    # ISO dates that look like DOBs (standalone YYYY-MM-DD)
    (
        re.compile(r"\b\d{4}-\d{2}-\d{2}\b"),
        lambda m: _redact_dob(m.group(0)),
    ),
    # US dates: MM/DD/YYYY
    (
        re.compile(r"\b\d{2}/\d{2}/\d{4}\b"),
        lambda m: _redact_dob(m.group(0)),
    ),
]


def _redact_text_pii(text: str) -> str:
    """Apply pattern-based PII redaction to free text."""
    for pattern, replacer in _TEXT_PII_PATTERNS:
        text = pattern.sub(replacer, text)
    return text


# ============================================================
# RECURSIVE DICT WALKER
# ============================================================


def _redact_dict(data: dict[str, Any]) -> dict[str, Any]:
    """Recursively walk a dict and redact PII fields."""
    result: dict[str, Any] = {}
    for key, value in data.items():
        lower_key = key.lower()

        # Check if this key is a known PII field
        if lower_key in _FIELD_REDACTORS and isinstance(value, str):
            result[key] = _FIELD_REDACTORS[lower_key](value)
        elif isinstance(value, dict):
            result[key] = _redact_dict(value)
        elif isinstance(value, list):
            result[key] = _redact_list(value)
        elif isinstance(value, str) and lower_key in (
            "content",
            "text",
            "input",
            "output",
        ):
            # Free-text fields that might contain PII in natural language
            result[key] = _redact_text_pii(value)
        else:
            result[key] = value

    return result


def _redact_list(data: list[Any]) -> list[Any]:
    """Recursively redact PII in a list."""
    result: list[Any] = []
    for item in data:
        if isinstance(item, dict):
            result.append(_redact_dict(item))
        elif isinstance(item, list):
            result.append(_redact_list(item))
        elif isinstance(item, str):
            result.append(_redact_text_pii(item))
        else:
            result.append(item)
    return result


# ============================================================
# PUBLIC API
# ============================================================


def redact_pii(data: dict[str, Any]) -> dict[str, Any]:
    """Anonymizer function for the LangSmith Client.

    Pass this as the ``anonymizer`` argument when creating the
    LangSmith Client:

        from langsmith import Client
        client = Client(anonymizer=redact_pii)

    The function receives the serialized run inputs or outputs
    as a plain dict, and returns a redacted copy. The original
    dict is not modified.
    """
    return _redact_dict(deepcopy(data))
