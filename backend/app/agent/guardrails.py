"""
Guardrails for the medical scheduling agent.

This module provides safety layers that run BEFORE and AFTER the
LLM agents process a message. Unlike prompt-based safety rules
(which the LLM can ignore under adversarial pressure or long
conversations), these are deterministic checks that cannot be
bypassed by clever prompting.

Layers:
  1. Input guardrails — classify incoming messages and intercept
     emergencies, medical advice requests, prompt injections, and
     off-topic queries before they reach the agent graph.
  2. Output guardrails — scan agent responses for medical advice
     patterns before they reach the patient. (Phase 5b)

Design:
  The supervisor calls ``screen_input()`` as its very first step.
  If screening returns a GuardrailResult, the supervisor short-
  circuits and returns that response directly. Otherwise, normal
  routing continues.

  This keeps guardrail logic isolated and testable while ensuring
  it runs on every patient turn (since the supervisor is the only
  node that executes on every message).
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass

logger = logging.getLogger(__name__)


# ============================================================
# RESULT TYPE
# ============================================================


@dataclass(frozen=True, slots=True)
class GuardrailResult:
    """Outcome of a guardrail check that should short-circuit the agent.

    Attributes:
        category: What the guardrail detected (for logging/tracing).
        response: The message to send back to the patient.
    """

    category: str
    response: str


# ============================================================
# EMERGENCY RESPONSE MESSAGE
# ============================================================
# Calming, clear, and actionable. Does NOT diagnose — just tells
# the patient to seek immediate help. Kept short for voice output.
# ============================================================

_EMERGENCY_RESPONSE = (
    "What you're describing sounds like it could be a medical emergency. "
    "Please call 911 or go to your nearest emergency room right away. "
    "Do not wait to make an appointment — get help now. "
    "If someone is with you, ask them to help you get to the ER."
)


# ============================================================
# RED-FLAG SYMPTOM DEFINITIONS
# ============================================================
# Emergency detection uses TWO approaches:
#
# 1. STANDALONE patterns — phrases that are emergencies on their
#    own, regardless of context. Examples: "I can't breathe",
#    "I'm having a stroke", "I'm going to kill myself".
#
# 2. COMBINATION patterns — symptom clusters where no single
#    symptom is an emergency alone, but the combination is.
#    Example: "chest pain" alone could be muscular, but
#    "chest pain + left arm numb + sweating" = heart attack.
#
# The combination approach uses a "primary + corroborating"
# model: a primary symptom (like chest pain) only triggers an
# emergency when accompanied by at least one corroborating
# symptom from its cluster.
#
# Why rule-based instead of LLM-based?
#   - Speed: no API call latency for life-threatening situations
#   - Reliability: deterministic, never misses a pattern
#   - Testability: every pattern can be unit-tested
#   - No failure mode: works even if the LLM API is down
# ============================================================


# ── Standalone emergencies ────────────────────────────────────
# These phrases are emergencies regardless of context. Each is a
# compiled regex so we can handle variations like "can't" / "cannot"
# / "cant" without maintaining a huge keyword list.

_STANDALONE_PATTERNS: list[re.Pattern[str]] = [
    # Breathing emergencies
    re.compile(r"can\s*'?n?o?t\s+breathe?", re.IGNORECASE),
    re.compile(r"(having|have)\s+(a\s+)?hard\s+time\s+breathing", re.IGNORECASE),
    re.compile(r"(struggling|gasping)\s+(to|for)\s+(breathe?|air|breath)", re.IGNORECASE),
    re.compile(r"choking\b", re.IGNORECASE),
    re.compile(r"airway.{0,15}(blocked|closing|swelling)", re.IGNORECASE),
    # Stroke indicators
    re.compile(r"(having|have|think)\b.{0,20}\bstroke\b", re.IGNORECASE),
    re.compile(r"face\s+.{0,10}(droop|numb|drooping)", re.IGNORECASE),
    re.compile(r"(slurring|slurred)\s+(my\s+)?(words|speech)", re.IGNORECASE),
    # Cardiac arrest
    re.compile(r"heart\s+(stopped|attack|is\s+stopping)", re.IGNORECASE),
    re.compile(r"(having|have|think)\b.{0,20}\bheart\s+attack\b", re.IGNORECASE),
    # Loss of consciousness
    re.compile(r"(passed|passing)\s+out\b", re.IGNORECASE),
    re.compile(r"(lost|losing)\s+consciousness", re.IGNORECASE),
    re.compile(r"(unresponsive|not\s+responding)", re.IGNORECASE),
    # Seizure
    re.compile(r"(having|have|had)\s+(a\s+)?seizure", re.IGNORECASE),
    # Severe bleeding
    re.compile(r"(bleeding|blood)\b.{0,15}(won\s*'?t\s+stop|everywhere|heavily|profuse)",
               re.IGNORECASE),
    re.compile(r"(severe|uncontrolled|heavy)\s+bleeding", re.IGNORECASE),
    # Anaphylaxis
    re.compile(r"(throat|tongue).{0,15}(swelling|closing|swollen)", re.IGNORECASE),
    re.compile(r"anaphyla", re.IGNORECASE),  # anaphylaxis, anaphylactic
    # Suicidal / self-harm
    re.compile(r"(want|going|plan)\s+to\s+(kill|hurt|harm)\s+(myself|me)", re.IGNORECASE),
    re.compile(r"(suicidal|end\s+my\s+life|end\s+it\s+all)", re.IGNORECASE),
    # Overdose / poisoning
    re.compile(r"(took|swallowed|ingested)\s+.{0,20}(pills|poison|bleach|chemicals)",
               re.IGNORECASE),
    re.compile(r"overdos", re.IGNORECASE),  # overdose, overdosed, overdosing
]


# ── Combination emergencies ───────────────────────────────────
# Each cluster has:
#   - primary_keywords: the anchor symptom (must be present)
#   - corroborating_keywords: supporting symptoms (≥1 must match)
#   - min_corroborating: how many supporting symptoms are needed
#     (default 1, but some clusters need 2 to avoid false positives)
#
# All matching is case-insensitive on the normalized message text.

@dataclass(frozen=True, slots=True)
class SymptomCluster:
    """A combination of symptoms that together indicate an emergency.

    Attributes:
        name: Human-readable label for logging.
        primary_keywords: At least one must be present in the message.
        corroborating_keywords: Supporting symptoms that confirm the
            emergency when combined with a primary keyword.
        min_corroborating: Minimum number of corroborating keywords
            that must match (default 1).
    """

    name: str
    primary_keywords: tuple[str, ...]
    corroborating_keywords: tuple[str, ...]
    min_corroborating: int = 1


_SYMPTOM_CLUSTERS: tuple[SymptomCluster, ...] = (
    # ── Cardiac emergency ──────────────────────────────────
    # Chest pain alone is common and usually not an emergency.
    # But chest pain + radiating pain + autonomic symptoms =
    # classic heart attack presentation.
    SymptomCluster(
        name="cardiac_emergency",
        primary_keywords=(
            "chest pain",
            "chest tight",
            "chest pressure",
            "crushing chest",
            "chest feels heavy",
            "chest is heavy",
            "chest feels tight",
            "chest is tight",
            "pressure in my chest",
            "pain in my chest",
            "tightness in my chest",
            "squeezing in my chest",
        ),
        corroborating_keywords=(
            "left arm",
            "arm numb",
            "arm pain",
            "arm tingling",
            "jaw pain",
            "jaw ache",
            "jaw hurts",
            "sweating",
            "cold sweat",
            "nausea",
            "nauseous",
            "dizzy",
            "dizziness",
            "lightheaded",
            "short of breath",
            "shortness of breath",
            "can't breathe",
            "radiating",
            "spreading to",
        ),
    ),
    # ── Stroke (combination form) ──────────────────────────
    # Standalone patterns catch explicit "I'm having a stroke"
    # but this catches the symptom-described version:
    # sudden numbness + confusion + vision changes.
    SymptomCluster(
        name="stroke_symptoms",
        primary_keywords=(
            "sudden numbness",
            "sudden weakness",
            "one side of my body",
            "one side of my face",
            "half my face",
            "half my body",
            "face drooping",
            "face is drooping",
        ),
        corroborating_keywords=(
            "confused",
            "confusion",
            "can't speak",
            "can't talk",
            "trouble speaking",
            "trouble talking",
            "slurring",
            "vision",
            "can't see",
            "blurry",
            "severe headache",
            "worst headache",
            "trouble walking",
            "loss of balance",
            "dizzy",
        ),
    ),
    # ── Severe allergic reaction ───────────────────────────
    # Hives alone aren't an emergency. Hives + breathing
    # difficulty + swelling = anaphylaxis.
    SymptomCluster(
        name="anaphylaxis",
        primary_keywords=(
            "hives",
            "allergic reaction",
            "allergy",
            "swelling",
            "swollen",
        ),
        corroborating_keywords=(
            "can't breathe",
            "hard to breathe",
            "difficulty breathing",
            "throat closing",
            "throat swelling",
            "tongue swelling",
            "tongue swollen",
            "dizzy",
            "lightheaded",
            "passing out",
        ),
    ),
    # ── Meningitis / brain bleed ───────────────────────────
    # "Worst headache of my life" + neck stiffness + fever
    # is a classic meningitis or subarachnoid hemorrhage triad.
    SymptomCluster(
        name="meningitis_or_hemorrhage",
        primary_keywords=(
            "worst headache",
            "thunderclap headache",
            "sudden severe headache",
            "worst headache of my life",
            "explosive headache",
        ),
        corroborating_keywords=(
            "stiff neck",
            "neck stiff",
            "neck is stiff",
            "neck feels stiff",
            "neck pain",
            "fever",
            "vomiting",
            "light sensitivity",
            "can't move my neck",
            "confusion",
            "confused",
            "seizure",
            "blacking out",
        ),
    ),
)


# ============================================================
# SELF-HARM RESPONSE (different from medical emergency)
# ============================================================
# Self-harm / suicide requires a different tone and resources
# compared to a medical emergency. We still urge immediate help
# but include the 988 Suicide & Crisis Lifeline.
# ============================================================

_SELF_HARM_RESPONSE = (
    "I'm really concerned about what you're sharing. You're not alone, "
    "and help is available right now. Please call or text 988 to reach "
    "the Suicide and Crisis Lifeline — they're available 24/7. "
    "If you're in immediate danger, please call 911."
)

_SELF_HARM_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"(want|going|plan)\s+to\s+(kill|hurt|harm)\s+(myself|me)", re.IGNORECASE),
    re.compile(r"(suicidal|end\s+my\s+life|end\s+it\s+all)", re.IGNORECASE),
    re.compile(r"don\s*'?t\s+want\s+to\s+(live|be\s+alive|be\s+here)", re.IGNORECASE),
    re.compile(r"(thinking|thought)\s+(about|of)\s+(killing|hurting)\s+(myself|me)",
               re.IGNORECASE),
]


# ============================================================
# EMERGENCY DETECTION
# ============================================================


def _normalize_text(text: str) -> str:
    """Lowercase, collapse whitespace, strip for consistent matching."""
    return " ".join(text.lower().split())


def check_emergency(message_text: str) -> GuardrailResult | None:
    """Check whether a patient message describes a medical emergency.

    Runs two checks in order:
      1. Self-harm patterns → returns crisis resources (988 Lifeline)
      2. Standalone emergency patterns → immediate 911 message
      3. Combination symptom clusters → immediate 911 message

    Returns a GuardrailResult if an emergency is detected, or None
    if the message is safe to continue processing normally.

    This function is deterministic (no LLM calls) and fast — it
    should add negligible latency to every turn.
    """
    if not message_text or not message_text.strip():
        return None

    normalized = _normalize_text(message_text)

    # ── 1. Self-harm check (highest priority) ─────────────
    for pattern in _SELF_HARM_PATTERNS:
        if pattern.search(normalized):
            logger.warning("GUARDRAIL: Self-harm language detected")
            return GuardrailResult(
                category="self_harm",
                response=_SELF_HARM_RESPONSE,
            )

    # ── 2. Standalone emergency patterns ──────────────────
    for pattern in _STANDALONE_PATTERNS:
        if pattern.search(normalized):
            logger.warning(
                "GUARDRAIL: Standalone emergency pattern matched: %s",
                pattern.pattern,
            )
            return GuardrailResult(
                category="emergency_standalone",
                response=_EMERGENCY_RESPONSE,
            )

    # ── 3. Combination symptom clusters ───────────────────
    for cluster in _SYMPTOM_CLUSTERS:
        has_primary = any(kw in normalized for kw in cluster.primary_keywords)
        if not has_primary:
            continue

        corroborating_count = sum(
            1 for kw in cluster.corroborating_keywords if kw in normalized
        )
        if corroborating_count >= cluster.min_corroborating:
            logger.warning(
                "GUARDRAIL: Emergency cluster '%s' triggered "
                "(primary + %d corroborating symptoms)",
                cluster.name,
                corroborating_count,
            )
            return GuardrailResult(
                category=f"emergency_cluster_{cluster.name}",
                response=_EMERGENCY_RESPONSE,
            )

    return None


# ============================================================
# INPUT CLASSIFICATION — MEDICAL ADVICE REQUESTS
# ============================================================
# Catches messages asking for diagnosis, treatment, medication,
# or prognosis. These are out of scope for a scheduling agent.
#
# Important: questions about WHICH SPECIALIST to see are IN scope
# (that's what the triage agent does). We only deflect questions
# about what to do/take/diagnose — not about who to see.
# ============================================================

_MEDICAL_ADVICE_RESPONSE = (
    "I'm not able to provide medical advice, but I can help you "
    "get an appointment with a specialist who can. Would you like "
    "me to help you find the right doctor?"
)

# Patterns that indicate a request for medical advice.
# These are checked ONLY if the message does NOT also contain
# scheduling-related language (to avoid false positives on
# messages like "should I see a dermatologist for my eczema?").
_MEDICAL_ADVICE_PATTERNS: list[re.Pattern[str]] = [
    # Medication / dosage questions
    re.compile(
        r"(what|which)\s+(medication|medicine|drug|pill)s?\s+should\s+i\s+(take|use)",
        re.IGNORECASE,
    ),
    re.compile(r"(can|should)\s+i\s+take\s+\w+.{0,30}(with|and|while)", re.IGNORECASE),
    re.compile(r"(is|are)\s+\w+\s+(safe|dangerous|okay|ok)\s+(to\s+take|with)", re.IGNORECASE),
    re.compile(r"(how\s+much|what\s+dose|what\s+dosage)\s+.{0,20}should\s+i", re.IGNORECASE),
    re.compile(r"(prescribe|prescription)\s+(me|for)", re.IGNORECASE),
    # Diagnosis questions
    re.compile(r"(what\s+is|what's)\s+(wrong|causing|the\s+cause)", re.IGNORECASE),
    re.compile(r"(do\s+i|could\s+i|might\s+i)\s+have\s+\w+", re.IGNORECASE),
    re.compile(
        r"(is\s+it|could\s+(it|this)\s+be)\s+(cancer|serious|dangerous|fatal)",
        re.IGNORECASE,
    ),
    re.compile(r"(diagnose|diagnosis)\s+(me|my|this)", re.IGNORECASE),
    # Treatment / home remedy questions
    re.compile(
        r"how\s+(do\s+i|can\s+i|should\s+i|to)\s+(treat|cure|fix|heal|get\s+rid\s+of)",
        re.IGNORECASE,
    ),
    re.compile(r"(home\s+remed|natural\s+remed|over\s+the\s+counter)", re.IGNORECASE),
    # Prognosis
    re.compile(
        r"(how\s+long|when)\s+will\s+(it|this|i)\s+.{0,15}"
        r"(heal|go\s+away|get\s+better|recover)",
        re.IGNORECASE,
    ),
    re.compile(
        r"(is\s+(it|this)|am\s+i)\s+(going\s+to\s+)?(get\s+worse|die|be\s+okay)",
        re.IGNORECASE,
    ),
]

# Scheduling-related terms. If a message contains these alongside
# medical language, it's probably an in-scope question about which
# specialist to see — not a request for medical advice.
_SCHEDULING_CONTEXT_TERMS = (
    "appointment",
    "book",
    "schedule",
    "see a doctor",
    "see a specialist",
    "which doctor",
    "which specialist",
    "should i see",
    "do you have",
    "do you offer",
    "dermatologist",
    "cardiologist",
    "neurologist",
    "specialist",
    "reschedule",
    "cancel",
)


def check_medical_advice_request(message_text: str) -> GuardrailResult | None:
    """Detect requests for medical advice that are outside scheduling scope.

    Returns None if the message is in scope (scheduling-related) or
    doesn't match any medical advice patterns.
    """
    normalized = _normalize_text(message_text)
    if not normalized:
        return None

    # If the message contains scheduling context, it's likely an
    # in-scope question like "should I see a dermatologist for my
    # eczema?" — let it through to the triage agent.
    has_scheduling_context = any(
        term in normalized for term in _SCHEDULING_CONTEXT_TERMS
    )
    if has_scheduling_context:
        return None

    for pattern in _MEDICAL_ADVICE_PATTERNS:
        if pattern.search(normalized):
            logger.info(
                "GUARDRAIL: Medical advice request detected: %s",
                pattern.pattern,
            )
            return GuardrailResult(
                category="medical_advice_request",
                response=_MEDICAL_ADVICE_RESPONSE,
            )

    return None


# ============================================================
# INPUT CLASSIFICATION — PROMPT INJECTION
# ============================================================
# Detects attempts to override the agent's instructions. These
# are higher-stakes in a medical context because a jailbroken
# agent could give harmful medical advice.
#
# We use a layered approach:
#   1. Strong patterns — high-confidence injection attempts
#      (e.g., "ignore your instructions"). Block immediately.
#   2. No weak/ambiguous tier — we avoid flagging normal
#      conversation that happens to contain words like "ignore"
#      or "pretend". Only block when the structure clearly
#      targets the agent's instructions.
# ============================================================

_PROMPT_INJECTION_RESPONSE = (
    "I'm the clinic's scheduling assistant. I can help you book, "
    "reschedule, or cancel appointments. How can I help?"
)

_PROMPT_INJECTION_PATTERNS: list[re.Pattern[str]] = [
    # Direct instruction override
    re.compile(
        r"(ignore|disregard|forget|override|bypass)\s+.{0,20}"
        r"(instructions|rules|guidelines|prompt|programming|system\s+prompt|constraints)",
        re.IGNORECASE,
    ),
    # Role reassignment
    re.compile(
        r"you\s+are\s+now\s+(a|an|my)\s+",
        re.IGNORECASE,
    ),
    re.compile(
        r"(act|behave|pretend|respond)\s+(as|like)\s+(a|an|if\s+you\s+were)\s+",
        re.IGNORECASE,
    ),
    # System prompt extraction
    re.compile(
        r"(show|reveal|display|print|output|repeat|tell)\s+.{0,10}"
        r"(your|the)\s+"
        r"(system\s+prompt|instructions|rules|prompt|programming|initial\s+prompt)",
        re.IGNORECASE,
    ),
    re.compile(
        r"what\s+(are|is)\s+your\s+(system\s+prompt|instructions|rules|programming)",
        re.IGNORECASE,
    ),
    # Jailbreak framing
    re.compile(r"(DAN|do\s+anything\s+now|jailbreak)", re.IGNORECASE),
    re.compile(
        r"(entering|enter|switch\s+to|enable)\s+"
        r"(developer|admin|debug|unrestricted|god)\s+mode",
        re.IGNORECASE,
    ),
]


def check_prompt_injection(message_text: str) -> GuardrailResult | None:
    """Detect prompt injection / jailbreak attempts.

    Returns a GuardrailResult if an injection attempt is detected.
    The response is deliberately bland — we don't acknowledge the
    attempt or explain what we caught, to avoid giving the attacker
    useful feedback.
    """
    normalized = _normalize_text(message_text)
    if not normalized:
        return None

    for pattern in _PROMPT_INJECTION_PATTERNS:
        if pattern.search(normalized):
            logger.warning(
                "GUARDRAIL: Prompt injection attempt detected: %s",
                pattern.pattern,
            )
            return GuardrailResult(
                category="prompt_injection",
                response=_PROMPT_INJECTION_RESPONSE,
            )

    return None


# ============================================================
# INPUT CLASSIFICATION — OFF-TOPIC
# ============================================================
# Catches requests that are completely unrelated to medical
# scheduling. We keep this lightweight — only flag things that
# are obviously non-medical. Ambiguous messages should flow
# through to the agents, which can handle them gracefully.
# ============================================================

_OFF_TOPIC_RESPONSE = (
    "I'm the clinic's scheduling assistant, so I'm best suited to "
    "help with booking, rescheduling, or cancelling appointments. "
    "Is there anything like that I can help you with?"
)

_OFF_TOPIC_PATTERNS: list[re.Pattern[str]] = [
    # Explicit non-medical requests
    re.compile(
        r"(write|compose|draft)\s+(me\s+)?(a|an|my)\s+(\w+\s+)?"
        r"(essay|email|letter|poem|story|code|script|resume)",
        re.IGNORECASE,
    ),
    re.compile(
        r"(help\s+me\s+with|assist\s+with)\s+(my\s+)?"
        r"(homework|assignment|project|taxes|code|coding|programming)",
        re.IGNORECASE,
    ),
    re.compile(
        r"(what's|what\s+is|tell\s+me)\s+the\s+"
        r"(weather|capital|population|score)",
        re.IGNORECASE,
    ),
    re.compile(
        r"(translate|convert)\s+.{0,30}(to|into)\s+"
        r"(spanish|french|german|chinese|japanese)",
        re.IGNORECASE,
    ),
    re.compile(
        r"(recipe|cook|bake)\s+(for|me)",
        re.IGNORECASE,
    ),
]


def check_off_topic(message_text: str) -> GuardrailResult | None:
    """Detect clearly off-topic requests unrelated to medical scheduling.

    Only flags obvious non-medical requests. Ambiguous messages are
    let through — it's better to let the agent handle a borderline
    case than to incorrectly block a patient who needs help.
    """
    normalized = _normalize_text(message_text)
    if not normalized:
        return None

    for pattern in _OFF_TOPIC_PATTERNS:
        if pattern.search(normalized):
            logger.info(
                "GUARDRAIL: Off-topic request detected: %s",
                pattern.pattern,
            )
            return GuardrailResult(
                category="off_topic",
                response=_OFF_TOPIC_RESPONSE,
            )

    return None


# ============================================================
# UNIFIED INPUT SCREENING
# ============================================================
# Single entry point that runs all input guardrails in priority
# order. The supervisor calls this once per turn.
#
# Priority order:
#   1. Emergency (life-threatening → 911 / crisis line)
#   2. Prompt injection (block manipulation attempts)
#   3. Medical advice (deflect to scheduling scope)
#   4. Off-topic (redirect clearly unrelated requests)
#
# Emergency is first because it's the highest stakes.
# Prompt injection is second because a successful injection
# could cause the agent to bypass all other guardrails.
# Medical advice and off-topic are lower priority because
# the worst case (agent answers a borderline question) is
# much less harmful.
# ============================================================


def screen_input(message_text: str) -> GuardrailResult | None:
    """Run all input guardrails on a patient message.

    Returns a GuardrailResult if the message should be intercepted,
    or None if it's safe to process normally.

    Called by the supervisor as its very first step on every turn.
    All checks are deterministic (no LLM calls) for speed and
    reliability.
    """
    # 1. Emergency — highest priority, life-threatening
    result = check_emergency(message_text)
    if result is not None:
        return result

    # 2. Prompt injection — block before it reaches any agent
    result = check_prompt_injection(message_text)
    if result is not None:
        return result

    # 3. Medical advice — deflect to scheduling scope
    result = check_medical_advice_request(message_text)
    if result is not None:
        return result

    # 4. Off-topic — redirect clearly unrelated requests
    result = check_off_topic(message_text)
    if result is not None:
        return result

    return None

# ============================================================
# OUTPUT GUARDRAILS
# ============================================================
# Scans agent responses BEFORE they reach the patient. Catches
# cases where the LLM slipped past its prompt instructions and
# generated medical advice, diagnoses, or treatment suggestions.
#
# Unlike input guardrails (which block the message entirely),
# output guardrails REWRITE the problematic response. The agent
# already did useful work (identified the patient, found slots,
# etc.) — we just need to strip the medical advice part and
# append a redirect.
#
# Design:
#   - For invoke (non-streaming): scan full text, rewrite if needed
#   - For streaming: collect text as it streams, scan after completion,
#     log a warning if violations are found. We can't un-stream tokens,
#     but we get visibility for prompt improvement.
#     In Phase 6 (voice), we'll have a natural buffering point
#     between agent and TTS where we can intercept in real-time.
# ============================================================


@dataclass(frozen=True, slots=True)
class OutputViolation:
    """A detected output guardrail violation.

    Attributes:
        category: Type of violation (for logging).
        matched_text: The specific text that triggered the violation.
    """

    category: str
    matched_text: str


# Patterns that indicate the agent is giving medical advice.
# These scan the AGENT's response, not the patient's message,
# so they target the language an LLM would use when advising.
_OUTPUT_ADVICE_PATTERNS: list[re.Pattern[str]] = [
    # Direct treatment suggestions
    re.compile(
        r"(you\s+should|i\s+recommend|i\s+suggest|try\s+taking|consider\s+taking)\s+"
        r".{0,30}(medication|medicine|ibuprofen|aspirin|tylenol|acetaminophen|"
        r"advil|motrin|naproxen|antibiotic|antidepressant|prescription)",
        re.IGNORECASE,
    ),
    # Dosage advice
    re.compile(
        r"(take|dose|dosage)\s+.{0,20}(mg|milligram|tablet|pill|capsule|twice|"
        r"three\s+times|daily|every\s+\d+\s+hours)",
        re.IGNORECASE,
    ),
    # Diagnosis language
    re.compile(
        r"(you\s+(have|likely\s+have|probably\s+have|may\s+have|might\s+have|"
        r"could\s+have|seem\s+to\s+have|appear\s+to\s+have))\s+",
        re.IGNORECASE,
    ),
    re.compile(
        r"(this\s+(sounds\s+like|looks\s+like|could\s+be|is\s+likely|is\s+probably|"
        r"appears\s+to\s+be|seems\s+like))\s+.{0,20}"
        r"(disease|disorder|condition|syndrome|infection|diagnosis|cancer|diabetes|"
        r"migraine|arthritis|fracture|sprain|pneumonia|bronchitis)",
        re.IGNORECASE,
    ),
    # Treatment plans
    re.compile(
        r"(treatment\s+plan|course\s+of\s+treatment|here'?s?\s+what\s+you\s+should\s+do)",
        re.IGNORECASE,
    ),
    # Home care instructions (beyond scope)
    re.compile(
        r"(apply\s+(ice|heat|cold\s+compress|warm\s+compress)|"
        r"rest\s+and\s+(elevate|ice)|RICE\s+method)",
        re.IGNORECASE,
    ),
]

# Patterns to EXCLUDE from violation detection. These are
# legitimate phrases the agent uses in its scheduling role
# that happen to overlap with medical language.
_OUTPUT_SAFE_PATTERNS: list[re.Pattern[str]] = [
    # Triage agent recommending a specialist (in scope)
    re.compile(
        r"(i'?d?\s+recommend|i\s+suggest)\s+(seeing|visiting|booking\s+with|"
        r"an?\s+appointment\s+with|a\s+consultation\s+with)\s+",
        re.IGNORECASE,
    ),
    # Explaining what a specialist can help with (in scope)
    re.compile(
        r"(specialist|doctor|dermatologist|cardiologist|neurologist|"
        r"orthopedist|gastroenterologist|psychiatrist|allergist)\s+"
        r"(can|will|would)\s+(help|assist|evaluate|assess|examine)",
        re.IGNORECASE,
    ),
]


def check_output(agent_response: str) -> list[OutputViolation]:
    """Scan an agent response for medical advice violations.

    Returns a list of violations found. An empty list means the
    response is clean.

    This does NOT modify the response — the caller decides whether
    to rewrite, log, or both.
    """
    if not agent_response or not agent_response.strip():
        return []

    normalized = _normalize_text(agent_response)
    violations: list[OutputViolation] = []

    # Check safe patterns first — if a safe pattern matches, skip
    # that region to avoid false positives.
    is_safe = any(p.search(normalized) for p in _OUTPUT_SAFE_PATTERNS)

    for pattern in _OUTPUT_ADVICE_PATTERNS:
        match = pattern.search(normalized)
        if match and not is_safe:
            violations.append(
                OutputViolation(
                    category="medical_advice_in_output",
                    matched_text=match.group(0),
                )
            )

    return violations


def sanitize_output(agent_response: str) -> str:
    """Scan an agent response and rewrite if medical advice is found.

    If no violations are detected, returns the original response
    unchanged. If violations are found, returns a safe replacement
    that redirects to scheduling.

    Used by invoke_agent (non-streaming) where we have the full
    response before sending it to the patient.
    """
    violations = check_output(agent_response)

    if not violations:
        return agent_response

    logger.warning(
        "OUTPUT GUARDRAIL: %d violation(s) detected in agent response: %s",
        len(violations),
        ", ".join(v.matched_text for v in violations),
    )

    # Replace the entire response rather than trying to surgically
    # remove the bad parts. Partial rewriting is fragile — a sentence
    # like "You probably have a migraine, let me find a neurologist"
    # can't be cleanly split. Safer to replace entirely.
    return (
        "I'm not able to provide medical advice or a diagnosis, but I can "
        "help you see a specialist who can. Would you like me to help "
        "you find the right doctor and book an appointment?"
    )
