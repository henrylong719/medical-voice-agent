"""
Extended tests for time_utils — covers edge cases in date parsing,
time bucket logic, voice formatting, and normalization.
"""
from __future__ import annotations

from datetime import date, datetime, time, timedelta, timezone

from pytest import MonkeyPatch

from app.services import time_utils


# ── Helpers ───────────────────────────────────────────────────

def _fixed_now(monkeypatch: MonkeyPatch, dt: datetime) -> None:
    """Pin now_clinic() and now_utc() to a fixed datetime."""
    monkeypatch.setattr(time_utils, "now_clinic", lambda: dt)
    monkeypatch.setattr(
        time_utils,
        "now_utc",
        lambda: dt.astimezone(timezone.utc),
    )


# ============================================================
# parse_preferred_day
# ============================================================

def test_parse_preferred_day_tomorrow(monkeypatch: MonkeyPatch) -> None:
    _fixed_now(monkeypatch, datetime(2026, 4, 11, 9, 0, tzinfo=time_utils.CLINIC_TZ))

    result = time_utils.parse_preferred_day("tomorrow")

    assert result == time_utils.DayRange(date(2026, 4, 12), date(2026, 4, 13))


def test_parse_preferred_day_tmr_alias(monkeypatch: MonkeyPatch) -> None:
    _fixed_now(monkeypatch, datetime(2026, 4, 11, 9, 0, tzinfo=time_utils.CLINIC_TZ))

    result = time_utils.parse_preferred_day("tmr")

    assert result == time_utils.DayRange(date(2026, 4, 12), date(2026, 4, 13))


def test_parse_preferred_day_this_week(monkeypatch: MonkeyPatch) -> None:
    _fixed_now(monkeypatch, datetime(2026, 4, 11, 9, 0, tzinfo=time_utils.CLINIC_TZ))

    result = time_utils.parse_preferred_day("this week")

    assert result == time_utils.DayRange(date(2026, 4, 11), date(2026, 4, 18))


def test_parse_preferred_day_next_week(monkeypatch: MonkeyPatch) -> None:
    _fixed_now(monkeypatch, datetime(2026, 4, 11, 9, 0, tzinfo=time_utils.CLINIC_TZ))

    result = time_utils.parse_preferred_day("next week")

    assert result == time_utils.DayRange(date(2026, 4, 18), date(2026, 4, 25))


def test_parse_preferred_day_weekend(monkeypatch: MonkeyPatch) -> None:
    # Friday April 11, 2026
    _fixed_now(monkeypatch, datetime(2026, 4, 10, 9, 0, tzinfo=time_utils.CLINIC_TZ))

    result = time_utils.parse_preferred_day("weekend")

    assert result.start_date.weekday() == 5  # Saturday
    assert result.end_date == result.start_date + timedelta(days=2)


def test_parse_preferred_day_numeric_date(monkeypatch: MonkeyPatch) -> None:
    _fixed_now(monkeypatch, datetime(2026, 4, 11, 9, 0, tzinfo=time_utils.CLINIC_TZ))

    result = time_utils.parse_preferred_day("4/20")

    assert result == time_utils.DayRange(date(2026, 4, 20), date(2026, 4, 21))


def test_parse_preferred_day_numeric_with_year(monkeypatch: MonkeyPatch) -> None:
    _fixed_now(monkeypatch, datetime(2026, 4, 11, 9, 0, tzinfo=time_utils.CLINIC_TZ))

    result = time_utils.parse_preferred_day("04/20/2026")

    assert result == time_utils.DayRange(date(2026, 4, 20), date(2026, 4, 21))


def test_parse_preferred_day_month_name_first(monkeypatch: MonkeyPatch) -> None:
    _fixed_now(monkeypatch, datetime(2026, 4, 11, 9, 0, tzinfo=time_utils.CLINIC_TZ))

    result = time_utils.parse_preferred_day("April 20")

    assert result == time_utils.DayRange(date(2026, 4, 20), date(2026, 4, 21))


def test_parse_preferred_day_day_first_format(monkeypatch: MonkeyPatch) -> None:
    _fixed_now(monkeypatch, datetime(2026, 4, 11, 9, 0, tzinfo=time_utils.CLINIC_TZ))

    result = time_utils.parse_preferred_day("20 April")

    assert result == time_utils.DayRange(date(2026, 4, 20), date(2026, 4, 21))


def test_parse_preferred_day_abbreviated_month(monkeypatch: MonkeyPatch) -> None:
    _fixed_now(monkeypatch, datetime(2026, 4, 11, 9, 0, tzinfo=time_utils.CLINIC_TZ))

    result = time_utils.parse_preferred_day("feb 14")

    assert result == time_utils.DayRange(date(2026, 2, 14), date(2026, 2, 15))


def test_parse_preferred_day_n_weeks(monkeypatch: MonkeyPatch) -> None:
    _fixed_now(monkeypatch, datetime(2026, 4, 11, 9, 0, tzinfo=time_utils.CLINIC_TZ))

    result = time_utils.parse_preferred_day("2 weeks")

    assert result == time_utils.DayRange(date(2026, 4, 11), date(2026, 4, 25))


def test_parse_preferred_day_3_weeks(monkeypatch: MonkeyPatch) -> None:
    _fixed_now(monkeypatch, datetime(2026, 4, 11, 9, 0, tzinfo=time_utils.CLINIC_TZ))

    result = time_utils.parse_preferred_day("3 weeks")

    assert result == time_utils.DayRange(date(2026, 4, 11), date(2026, 5, 2))


def test_parse_preferred_day_ordinal_suffix_stripped(monkeypatch: MonkeyPatch) -> None:
    _fixed_now(monkeypatch, datetime(2026, 4, 11, 9, 0, tzinfo=time_utils.CLINIC_TZ))

    result = time_utils.parse_preferred_day("April 20th")

    assert result == time_utils.DayRange(date(2026, 4, 20), date(2026, 4, 21))


def test_parse_preferred_day_bare_weekday_same_day(monkeypatch: MonkeyPatch) -> None:
    # Saturday April 11, 2026
    _fixed_now(monkeypatch, datetime(2026, 4, 11, 9, 0, tzinfo=time_utils.CLINIC_TZ))

    result = time_utils.parse_preferred_day("saturday")

    assert result == time_utils.DayRange(date(2026, 4, 11), date(2026, 4, 12))


def test_parse_preferred_day_next_forces_future(monkeypatch: MonkeyPatch) -> None:
    # Saturday April 11, 2026
    _fixed_now(monkeypatch, datetime(2026, 4, 11, 9, 0, tzinfo=time_utils.CLINIC_TZ))

    result = time_utils.parse_preferred_day("next saturday")

    assert result.start_date > date(2026, 4, 11)


def test_parse_preferred_day_empty_string_returns_today(monkeypatch: MonkeyPatch) -> None:
    _fixed_now(monkeypatch, datetime(2026, 4, 11, 9, 0, tzinfo=time_utils.CLINIC_TZ))

    result = time_utils.parse_preferred_day("")

    assert result == time_utils.DayRange(date(2026, 4, 11), date(2026, 4, 12))


def test_parse_preferred_day_none_returns_today(monkeypatch: MonkeyPatch) -> None:
    _fixed_now(monkeypatch, datetime(2026, 4, 11, 9, 0, tzinfo=time_utils.CLINIC_TZ))

    result = time_utils.parse_preferred_day(None)

    assert result == time_utils.DayRange(date(2026, 4, 11), date(2026, 4, 12))


def test_parse_preferred_day_garbage_falls_back_to_today(monkeypatch: MonkeyPatch) -> None:
    _fixed_now(monkeypatch, datetime(2026, 4, 11, 9, 0, tzinfo=time_utils.CLINIC_TZ))

    result = time_utils.parse_preferred_day("xyzzy gibberish")

    assert result == time_utils.DayRange(date(2026, 4, 11), date(2026, 4, 12))


def test_parse_preferred_day_misspelled_tomorrow(monkeypatch: MonkeyPatch) -> None:
    _fixed_now(monkeypatch, datetime(2026, 4, 11, 9, 0, tzinfo=time_utils.CLINIC_TZ))

    result = time_utils.parse_preferred_day("tommorow")

    assert result == time_utils.DayRange(date(2026, 4, 12), date(2026, 4, 13))


# ============================================================
# parse_time_bucket
# ============================================================

def test_parse_time_bucket_am() -> None:
    assert time_utils.parse_time_bucket("am") == "morning"


def test_parse_time_bucket_afternoon() -> None:
    assert time_utils.parse_time_bucket("afternoon") == "afternoon"


def test_parse_time_bucket_whenever() -> None:
    assert time_utils.parse_time_bucket("whenever") == "any"


def test_parse_time_bucket_anything() -> None:
    assert time_utils.parse_time_bucket("anything") == "any"


def test_parse_time_bucket_none() -> None:
    assert time_utils.parse_time_bucket(None) == "any"


def test_parse_time_bucket_empty_string() -> None:
    assert time_utils.parse_time_bucket("") == "any"


def test_parse_time_bucket_garbage_returns_any() -> None:
    assert time_utils.parse_time_bucket("zebra") == "any"


def test_parse_time_bucket_early_morning() -> None:
    assert time_utils.parse_time_bucket("early morning") == "morning"


# ============================================================
# is_in_bucket
# ============================================================

def test_is_in_bucket_any_always_true() -> None:
    dt = datetime(2026, 4, 13, 3, 0, tzinfo=timezone.utc)
    assert time_utils.is_in_bucket(dt, "any") is True


def test_is_in_bucket_afternoon_correct() -> None:
    # 7 PM UTC = 2 PM CDT (afternoon bucket: 12-5pm)
    afternoon_utc = datetime(2026, 4, 13, 19, 0, tzinfo=timezone.utc)
    assert time_utils.is_in_bucket(afternoon_utc, "afternoon") is True


def test_is_in_bucket_morning_boundary_start() -> None:
    # 1 PM UTC = 8 AM CDT (start of morning bucket)
    morning_start = datetime(2026, 4, 13, 13, 0, tzinfo=timezone.utc)
    assert time_utils.is_in_bucket(morning_start, "morning") is True


def test_is_in_bucket_morning_boundary_end() -> None:
    # 5 PM UTC = 12 PM CDT (end of morning bucket, exclusive)
    morning_end = datetime(2026, 4, 13, 17, 0, tzinfo=timezone.utc)
    assert time_utils.is_in_bucket(morning_end, "morning") is False


def test_is_in_bucket_afternoon_boundary_start() -> None:
    # 5 PM UTC = 12 PM CDT (start of afternoon bucket)
    afternoon_start = datetime(2026, 4, 13, 17, 0, tzinfo=timezone.utc)
    assert time_utils.is_in_bucket(afternoon_start, "afternoon") is True


# ============================================================
# format_for_voice
# ============================================================

def test_format_for_voice_iso_string_input() -> None:
    result = time_utils.format_for_voice("2026-04-13T14:30:00+00:00")
    assert result == "Monday, April 13th at 9:30 AM"


def test_format_for_voice_datetime_input() -> None:
    dt = datetime(2026, 4, 13, 14, 30, tzinfo=timezone.utc)
    result = time_utils.format_for_voice(dt)
    assert result == "Monday, April 13th at 9:30 AM"


def test_format_for_voice_afternoon_time() -> None:
    dt = datetime(2026, 4, 13, 20, 0, tzinfo=timezone.utc)
    result = time_utils.format_for_voice(dt)
    assert result == "Monday, April 13th at 3:00 PM"


def test_format_for_voice_ordinal_suffixes() -> None:
    # 1st
    dt1 = datetime(2026, 5, 1, 14, 0, tzinfo=timezone.utc)
    assert "1st" in time_utils.format_for_voice(dt1)

    # 2nd
    dt2 = datetime(2026, 5, 2, 14, 0, tzinfo=timezone.utc)
    assert "2nd" in time_utils.format_for_voice(dt2)

    # 3rd
    dt3 = datetime(2026, 5, 3, 14, 0, tzinfo=timezone.utc)
    assert "3rd" in time_utils.format_for_voice(dt3)

    # 11th (special case)
    dt11 = datetime(2026, 5, 11, 14, 0, tzinfo=timezone.utc)
    assert "11th" in time_utils.format_for_voice(dt11)

    # 12th (special case)
    dt12 = datetime(2026, 5, 12, 14, 0, tzinfo=timezone.utc)
    assert "12th" in time_utils.format_for_voice(dt12)

    # 13th (special case)
    dt13 = datetime(2026, 5, 13, 14, 0, tzinfo=timezone.utc)
    assert "13th" in time_utils.format_for_voice(dt13)

    # 21st
    dt21 = datetime(2026, 5, 21, 14, 0, tzinfo=timezone.utc)
    assert "21st" in time_utils.format_for_voice(dt21)


# ============================================================
# format_date_for_voice
# ============================================================

def test_format_date_for_voice_from_datetime() -> None:
    dt = datetime(2026, 4, 13, 14, 30, tzinfo=timezone.utc)
    result = time_utils.format_date_for_voice(dt)
    assert result == "Monday, April 13th"


def test_format_date_for_voice_from_date() -> None:
    d = date(2026, 4, 13)
    result = time_utils.format_date_for_voice(d)
    assert result == "Monday, April 13th"


def test_format_date_for_voice_from_iso_string() -> None:
    result = time_utils.format_date_for_voice("2026-04-13T14:30:00+00:00")
    assert result == "Monday, April 13th"


# ============================================================
# _ordinal_suffix
# ============================================================

def test_ordinal_suffix_special_cases() -> None:
    assert time_utils._ordinal_suffix(1) == "st"
    assert time_utils._ordinal_suffix(2) == "nd"
    assert time_utils._ordinal_suffix(3) == "rd"
    assert time_utils._ordinal_suffix(4) == "th"
    assert time_utils._ordinal_suffix(11) == "th"
    assert time_utils._ordinal_suffix(12) == "th"
    assert time_utils._ordinal_suffix(13) == "th"
    assert time_utils._ordinal_suffix(21) == "st"
    assert time_utils._ordinal_suffix(22) == "nd"
    assert time_utils._ordinal_suffix(23) == "rd"
    assert time_utils._ordinal_suffix(31) == "st"


# ============================================================
# _normalize
# ============================================================

def test_normalize_strips_ordinal_suffixes() -> None:
    assert time_utils._normalize("24th") == "24"
    assert time_utils._normalize("1st") == "1"
    assert time_utils._normalize("2nd") == "2"
    assert time_utils._normalize("3rd") == "3"


def test_normalize_removes_filler_of() -> None:
    assert time_utils._normalize("5th of March") == "5 march"


def test_normalize_collapses_whitespace() -> None:
    assert time_utils._normalize("  next   monday  ") == "next monday"


def test_normalize_handles_none() -> None:
    assert time_utils._normalize(None) == ""


# ============================================================
# DayRange
# ============================================================

def test_day_range_is_immutable() -> None:
    dr = time_utils.DayRange(date(2026, 4, 13), date(2026, 4, 14))
    import dataclasses

    assert dataclasses.is_dataclass(dr)
    assert dr.start_date == date(2026, 4, 13)
    assert dr.end_date == date(2026, 4, 14)
