---
  Critical Issues

  2. SQL injection via ilike in triage — tools.py:636

  .ilike("symptom", f"%{symptom}%")
  The symptom string comes from the LLM's tool call arguments, which are derived from user input. While Supabase's PostgREST
  client parameterizes most queries, the % wildcard wrapping means a user who says their symptom is % will match everything.
  More importantly, if PostgREST doesn't properly escape pattern metacharacters (_, %), it could return unexpected results.
  Sanitize the symptom string before passing it to ilike.

  3. CORS allows all origins with credentials — main.py:66-72

  allow_origins=["*"],
  allow_credentials=True,
  allow_origins=["*"] with allow_credentials=True is explicitly forbidden by the CORS spec and browsers will reject it. In
  practice, most browsers block this combination. If you need credentials, you must specify explicit origins. Even for dev,
  this is a misconfiguration.

  ---
  Race Conditions & Concurrency

  4. TOCTOU race in book_appointment — tools.py:789-825

  validate_slot_selection checks availability, then insert books the slot. Between those two calls, another request could book
  the same slot. There's no database-level uniqueness constraint or row-level locking visible here. The cancel_appointment tool
   has the same pattern: fetch → check status → update, with no optimistic locking.

  5. Checkpointer singleton is not thread-safe — graph.py:78-117

  _get_checkpointer() uses global variables with no lock. If two concurrent requests hit it simultaneously during cold start,
  you could initialize two connections and leak one. Same issue with _get_or_build_graph().

  6. Agent cache is not concurrency-safe — agents.py:605-648

  The _get_intake_agent() / _get_triage_agent() / _get_scheduling_agent() functions mutate global state without locks. Two
  concurrent requests could both see None and build duplicate agents.

  ---
  Logic & Edge Cases

  7. invoke_agent finds the wrong "last human message" — graph.py:354-357

  The code finds the last HumanMessage in the full history to slice new responses. But if the agent internally creates
  HumanMessage objects (e.g., in sub-agent tool flows), new_start could point to the wrong message, causing responses to be
  lost or duplicated.

  8. Supabase client created at import time with empty credentials — supabase_client.py:39

  settings defaults all keys to "". The Supabase client is created at module import time. If the .env file is missing or
  misconfigured, this will either crash at import (hard to diagnose) or create a broken client that fails on every call with a
  confusing error.

  9. _normalize strips "of" everywhere — time_utils.py:101

  s = s.replace("of ", "")
  This replaces ALL occurrences of "of " — so "Office visit" becomes "fice visit", and month names aren't immune either. This
  should be more targeted (e.g., only between a number and a month name).

  10. Weekend logic breaks on Saturday/Sunday — time_utils.py:203-205

  days_until_sat = (5 - today.weekday()) % 7
  If today is Saturday (weekday()=5), days_until_sat = 0, so it returns this Saturday–Sunday. If today is Sunday (weekday()=6),
   days_until_sat = 6, returning next Saturday. A user saying "this weekend" on a Sunday would get dates 6-7 days away. This is
   likely not the intended behavior.

  11. _classify_patient_status has false positives — supervisor.py:358

  The marker "i have" will match "I have a headache" as returning, when the patient is actually describing symptoms. Similarly,
   "yes" anywhere will match returning even if the patient is saying "yes" to something else. These markers are too broad.

  12. Negative check beats affirmative in ambiguous replies — agents.py:427-466

  In _classify_identity_reply, negative tokens are checked before affirmative ones. "No, yeah that's me" would be classified as
   negative. The ordering creates a bias — consider checking for the most recent sentiment or falling through to the LLM
  classifier for ambiguous cases.

  13. Date parsing silently falls back to "today" — time_utils.py:238

  If parse_preferred_day can't parse the input, it returns today. A typo like "Wensday" will silently search only today instead
   of showing an error or searching the full horizon. The user would get "no slots found" with no indication their date was
  misunderstood.

  14. No pagination or rate limiting on chat endpoints — routes.py

  There's no rate limiting, no max message length beyond min_length=1, and no limit on concurrent streams per thread_id. A
  malicious client could open many concurrent SSE connections or send extremely long messages to drive up LLM costs.

  15. SSE stream has no error handling — routes.py:70-77

  If stream_agent_response raises mid-stream, the SSE connection will break without sending an error event. The client will
  just see a dropped connection with no indication of what went wrong.

  16. _normalize_date_of_birth doesn't handle 2-digit years well — tools.py:286-302

  Format %m-%d-%y maps "01-15-99" to 1999, but "01-15-30" to 2030. For a medical system where patients can be born in the
  1900s, this ambiguity could register patients with the wrong birth year. There's no validation that the date is in the past.

  17. find_patients_by_demographics does exact name matching — tools.py:476

  .eq("full_name", normalized_name)
  An exact match means "John Smith" won't find "john smith" or "John  Smith" (double space). For a voice agent where names come
   from speech-to-text, this will be fragile. Consider ilike or a fuzzy match function.

  ---
  Architectural Concerns

  18. All Supabase tools are synchronous in an async context — tools.py

  The @tool functions use the synchronous Supabase client, which means they block the event loop during database calls. In a
  FastAPI async endpoint, this can starve other requests. Consider using an async Supabase client or running these in a thread
  pool.

  19. No input validation on UUIDs — tools.py (multiple tools)

  patient_id, doctor_id, specialty_id, and appointment_id are accepted as plain strings with no UUID format validation. If the
  LLM hallucinates a malformed ID, the error message from Supabase will be cryptic rather than helpful.

Each conversation in Claude Code is tied to your terminal session. To resume later:

  - Same directory: Just run claude again from the same project directory. Claude Code will have access to the same files and
  can re-read the issues.md you saved.
  - Continue this exact conversation: Press Esc to exit, then run claude --continue (or claude -c) to pick up where you left
  off.
  - Resume a specific past conversation: Run claude --resume to get a list of recent conversations to choose from.

claude --resume ac4e71c3-a611-4949-99e0-13acef0e0935  
