from __future__ import annotations

from datetime import date, datetime, timezone

from app.services import time_utils
from pytest import MonkeyPatch


def test_parse_preferred_day_handles_relative_and_weekday_inputs(
    monkeypatch: MonkeyPatch,
) -> None:
    fixed_now = datetime(2026, 4, 11, 9, 0, tzinfo=time_utils.CLINIC_TZ)
    monkeypatch.setattr(time_utils, "now_clinic", lambda: fixed_now)

    assert time_utils.parse_preferred_day("today") == time_utils.DayRange(
        date(2026, 4, 11),
        date(2026, 4, 12),
    )
    assert time_utils.parse_preferred_day("next monday") == time_utils.DayRange(
        date(2026, 4, 13),
        date(2026, 4, 14),
    )
    assert time_utils.parse_preferred_day("sat") == time_utils.DayRange(
        date(2026, 4, 11),
        date(2026, 4, 12),
    )


def test_parse_time_bucket_and_bucket_membership() -> None:
    assert time_utils.parse_time_bucket("morning") == "morning"
    assert time_utils.parse_time_bucket("pm") == "afternoon"
    assert time_utils.parse_time_bucket("no preference") == "any"

    morning_utc = datetime(2026, 4, 13, 14, 0, tzinfo=timezone.utc)
    afternoon_utc = datetime(2026, 4, 13, 19, 0, tzinfo=timezone.utc)

    assert time_utils.is_in_bucket(morning_utc, "morning") is True
    assert time_utils.is_in_bucket(afternoon_utc, "morning") is False


def test_day_range_to_utc_and_voice_formatting() -> None:
    dr = time_utils.DayRange(date(2026, 4, 13), date(2026, 4, 14))
    start_utc, end_utc = time_utils.day_range_to_utc(dr)

    assert start_utc == datetime(2026, 4, 13, 5, 0, tzinfo=timezone.utc)
    assert end_utc == datetime(2026, 4, 14, 5, 0, tzinfo=timezone.utc)

    dt_utc = datetime(2026, 4, 13, 14, 30, tzinfo=timezone.utc)
    assert time_utils.format_for_voice(dt_utc) == "Monday, April 13th at 9:30 AM"
    assert time_utils.format_date_for_voice(dt_utc) == "Monday, April 13th"
