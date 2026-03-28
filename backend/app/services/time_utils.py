"""
Time utilities for natural language date parsing and voice-friendly formatting.
 
Patients don't say "2026-04-07" — they say "next Tuesday morning."
This module bridges that gap with:
 
  parse_preferred_day()   → converts "next thurs", "feb 24", "2/24" to a DayRange
  parse_time_bucket()     → converts "morning" to a Bucket
  format_for_voice()      → converts a datetime to "Monday, April 7th at 9:30 AM"
  format_date_for_voice() → converts a datetime to "Monday, April 7th"
  day_range_to_utc()      → converts a clinic-local DayRange to UTC boundaries
  is_in_bucket()          → checks if a UTC datetime falls in a time bucket
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta, timezone
from typing import Literal
from zoneinfo import ZoneInfo

from app.config import settings

# ============================================================
# TYPES & CONSTANTS
# ============================================================

Bucket = Literal["morning", "afternoon", "any"]

CLINIC_TZ = ZoneInfo(settings.timezone)

BUCKETS: dict[Bucket, tuple[time, time]] = {
    "morning":   (time(8, 0),  time(12, 0)),
    "afternoon": (time(12, 0), time(17, 0)),
    "any":       (time(0, 0),  time(23, 59, 59)),
}

# Supports abbreviations: "mon", "tue", "weds", "thurs", etc.
WEEKDAY_MAP: dict[str, int] = {
    "mon": 0, "monday": 0,
    "tue": 1, "tues": 1, "tuesday": 1,
    "wed": 2, "weds": 2, "wednesday": 2,
    "thu": 3, "thur": 3, "thurs": 3, "thursday": 3,
    "fri": 4, "friday": 4,
    "sat": 5, "saturday": 5,
    "sun": 6, "sunday": 6,
}

MONTH_MAP: dict[str, int] = {
    "jan": 1, "january": 1, "feb": 2, "february": 2,
    "mar": 3, "march": 3, "apr": 4, "april": 4,
    "may": 5, "jun": 6, "june": 6, "jul": 7, "july": 7,
    "aug": 8, "august": 8, "sep": 9, "sept": 9, "september": 9,
    "oct": 10, "october": 10, "nov": 11, "november": 11,
    "dec": 12, "december": 12,
}


@dataclass(frozen=True)
class DayRange:
    """
    An immutable date range with exclusive end.

    'next tuesday' → DayRange(2026-04-07, 2026-04-08)
    'this week'    → DayRange(2026-03-25, 2026-04-01)

    Exclusive end simplifies iteration: for d in range, d < end_date.
    """
    start_date: date  # inclusive
    end_date: date    # exclusive
    
# ============================================================
# HELPERS
# ============================================================

def now_utc() -> datetime:
    """Current time in UTC."""
    return datetime.now(timezone.utc)
  
  
def now_clinic() -> datetime:
    """Current time in the clinic's local timezone."""
    return now_utc().astimezone(CLINIC_TZ)
  
def _normalize(s: str) -> str:
    """
    Clean up raw user input for reliable parsing.

    - Lowercases
    - Strips special characters (keeps alphanumeric, spaces, /, :, -)
    - Collapses whitespace
    - Removes ordinal suffixes: "24th" → "24"
    - Removes filler word "of": "5th of march" → "5 march"
    """
    s = (s or "").strip().lower()
    s = re.sub(r"[^\w\s/:-]", "", s)
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r"\b(\d{1,2})(st|nd|rd|th)\b", r"\1", s)
    s = s.replace("of ", "")
    return s.strip()
  
def _next_weekday(from_date: date, target: int, strictly_after: bool) -> date:
    """
    Find the next occurrence of a weekday.

    Args:
        from_date: Starting date.
        target: Target weekday (0=Monday, 6=Sunday).
        strictly_after: If True, always returns a future date (skips today).
                        If False, returns today if it matches.
    """
    d = from_date + timedelta(days=1) if strictly_after else from_date
    days_ahead = (target - d.weekday()) % 7
    return d + timedelta(days=days_ahead)
  

def _parse_mmdd(s: str, year: int) -> date | None:
    """Parse numeric date formats: '2/24', '02/24', '02/24/2026'."""
    m = re.fullmatch(r"(\d{1,2})/(\d{1,2})(?:/(\d{2,4}))?", s)
    if not m:
        return None
    mm, dd = int(m.group(1)), int(m.group(2))
    yy = m.group(3)
    y = int(yy) + (2000 if int(yy) < 100 else 0) if yy else year
    try:
        return date(y, mm, dd)
    except ValueError:
        return None
      
def _parse_month_day(s: str, year: int) -> date | None:
    """
    Parse month-name date formats in either order:
        'feb 24', 'february 24', '24 feb', '24 february'
        'feb 24 2027', '24 feb 2027'
    """
    for pattern in [
        r"([a-z]+)\s+(\d{1,2})(?:\s+(\d{2,4}))?",   # "feb 24"
        r"(\d{1,2})\s+([a-z]+)(?:\s+(\d{2,4}))?",    # "24 feb"
    ]:
        m = re.fullmatch(pattern, s)
        if not m:
            continue
        g1, g2, g3 = m.group(1), m.group(2), m.group(3)
        if g1.isdigit():
            dd, mon_str = int(g1), g2
        else:
            mon_str, dd = g1, int(g2)
        mm = MONTH_MAP.get(mon_str)
        if not mm:
            continue
        y = int(g3) + (2000 if int(g3) < 100 else 0) if g3 else year
        try:
            return date(y, mm, dd)
        except ValueError:
            continue
    return None


# ============================================================
# DATE RANGE PARSING
# ============================================================

def parse_preferred_day(preferred_day: str | None) -> DayRange:
    """
    Parse natural language day preferences into a concrete date range.

    Supports:
        "today", "tod"                    → today
        "tomorrow", "tmr"                 → tomorrow
        "this week"                       → today through 7 days
        "next week"                       → 7 days from today through 14
        "weekend", "this weekend"         → Saturday–Sunday
        "next monday", "next thurs"       → next occurrence (strictly after today)
        "tuesday", "fri"                  → next occurrence (including today)
        "2/24", "02/24/2026"              → specific date
        "feb 24", "24 feb", "march 5"    → month + day
        "3 weeks"                         → today through N weeks
        None, ""                          → today (single day)

    Returns:
        DayRange with exclusive end_date.
    """
    s = _normalize(preferred_day or "")
    today = now_clinic().date()

    # Direct keywords
    if s in ("", "today", "tod"):
        return DayRange(today, today + timedelta(days=1))

    if s in ("tomorrow", "tmr", "tommorow"):
        d = today + timedelta(days=1)
        return DayRange(d, d + timedelta(days=1))

    if s in ("this week", "thisweek"):
        return DayRange(today, today + timedelta(days=7))

    if s in ("next week", "nextweek"):
        start = today + timedelta(days=7)
        return DayRange(start, start + timedelta(days=7))

    if s in ("weekend", "this weekend"):
        days_until_sat = (5 - today.weekday()) % 7
        sat = today + timedelta(days=days_until_sat)
        return DayRange(sat, sat + timedelta(days=2))

    # "next monday", "next thurs" — strictly after today
    m = re.fullmatch(r"next\s+([a-z]+)", s)
    if m:
        w = WEEKDAY_MAP.get(m.group(1))
        if w is not None:
            d = _next_weekday(today, w, strictly_after=True)
            return DayRange(d, d + timedelta(days=1))

    # Bare weekday name: "tuesday", "fri" — includes today if it matches
    if s in WEEKDAY_MAP:
        d = _next_weekday(today, WEEKDAY_MAP[s], strictly_after=False)
        return DayRange(d, d + timedelta(days=1))

    # Numeric: "2/24", "02/24/2026"
    parsed = _parse_mmdd(s, today.year)
    if parsed:
        return DayRange(parsed, parsed + timedelta(days=1))

    # Month name: "feb 24", "24 feb", "march 5"
    parsed = _parse_month_day(s, today.year)
    if parsed:
        return DayRange(parsed, parsed + timedelta(days=1))

    # N weeks: "2 weeks", "3weeks"
    m = re.fullmatch(r"(\d+)\s*weeks?", s)
    if m:
        weeks = int(m.group(1))
        return DayRange(today, today + timedelta(weeks=weeks))

    # Fallback: today only
    return DayRange(today, today + timedelta(days=1))

 
# ============================================================
# TIME BUCKET PARSING
# ============================================================

def parse_time_bucket(preferred_time: str | None) -> Bucket:
    """
    Parse a time-of-day preference into a bucket.

    Supports: "morning", "afternoon", "am", "pm", "any", "whenever",
              "no preference", None, ""
    """
    s = _normalize(preferred_time or "")
    if s in ("", "any", "anything", "whenever", "no preference"):
        return "any"
    if "morn" in s or s == "am":
        return "morning"
    if "after" in s or s == "pm":
        return "afternoon"
    return "any"
  

# ============================================================
# UTC CONVERSION & BUCKET FILTERING
# ============================================================

def day_range_to_utc(dr: DayRange) -> tuple[datetime, datetime]:
    """
    Convert a clinic-local DayRange to UTC datetime boundaries.

    This is what we pass to Supabase queries — the database stores
    everything in UTC, so we need to convert our local date range.
    """
    start = datetime.combine(dr.start_date, time.min, tzinfo=CLINIC_TZ)
    end = datetime.combine(dr.end_date, time.min, tzinfo=CLINIC_TZ)
    return start.astimezone(timezone.utc), end.astimezone(timezone.utc)
  
  
def is_in_bucket(start_utc: datetime, bucket: Bucket) -> bool:
    """
    Check if a UTC datetime falls within a time bucket in clinic-local time.

    The slot engine generates slots in UTC, but "morning" means
    8 AM–12 PM in the clinic's timezone, not UTC.
    """
    if bucket == "any":
        return True
    local = start_utc.astimezone(CLINIC_TZ)
    b_start, b_end = BUCKETS[bucket]
    return b_start <= local.time() < b_end
  
  
# ============================================================
# VOICE FORMATTING
# ============================================================

def _ordinal_suffix(day: int) -> str:
    """Return the ordinal suffix for a day number (1st, 2nd, 3rd, 4th...)."""
    if 11 <= day <= 13:
        return "th"
    return {1: "st", 2: "nd", 3: "rd"}.get(day % 10, "th")
  
  
  
def format_for_voice(dt: datetime | str) -> str:
    """
    Format a datetime for natural spoken output.

    Input can be a datetime object or an ISO format string (from Supabase).
    Always converts to clinic-local time before formatting.

    Examples:
        2026-04-07T14:30:00+00:00 → "Monday, April 7th at 9:30 AM"  (in America/Chicago)
    """
    if isinstance(dt, str):
        dt = datetime.fromisoformat(dt)

    # Convert to clinic-local time
    local = dt.astimezone(CLINIC_TZ)

    day = local.day
    ordinal = _ordinal_suffix(day)

    # Format time — strip leading zero from hour
    time_str = local.strftime("%I:%M %p").lstrip("0")

    return f"{local.strftime('%A')}, {local.strftime('%B')} {day}{ordinal} at {time_str}"
  
  
def format_date_for_voice(dt: datetime | date | str) -> str:
    """
    Format a date (without time) for voice output.

    Examples:
        2026-04-07 → "Monday, April 7th"
    """
    if isinstance(dt, str):
        dt = datetime.fromisoformat(dt)

    if isinstance(dt, datetime):
        dt = dt.astimezone(CLINIC_TZ).date()

    ordinal = _ordinal_suffix(dt.day)
    return f"{dt.strftime('%A')}, {dt.strftime('%B')} {dt.day}{ordinal}"
