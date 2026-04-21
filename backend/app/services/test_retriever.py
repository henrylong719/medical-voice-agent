"""
RAG retriever test suite.

Tests the semantic search against a variety of patient queries to
verify retrieval quality across specialties, natural language styles,
and edge cases.

Run from the backend/ directory:
    python -m app.services.test_retriever

Each test case defines:
  - query: what the patient says
  - expected: the specialty that SHOULD be the top result
  - note: why this case is interesting

This isn't a formal eval framework (that's Phase 7) — it's a quick
smoke test you can run after changing chunks, thresholds, or the
embedding model to make sure nothing regressed.
"""

from app.services.rag_retriever import retrieve_medical_knowledge


# ============================================================
# TEST CASES
# ============================================================
# Organized by what we're testing:
#   - Direct matches: clinical terms the retriever should nail
#   - Colloquial language: how real patients actually talk
#   - Ambiguous symptoms: could match multiple specialties
#   - Edge cases: tricky or unusual descriptions
# ============================================================

TEST_CASES: list[dict[str, str]] = [
    # ── Direct matches (clinical terms) ──────────────────────
    {
        "query": "I have chest pain and shortness of breath",
        "expected": "Cardiology",
        "note": "Classic cardiology keywords",
    },
    {
        "query": "I've been having acid reflux and stomach pain after eating",
        "expected": "Gastroenterology",
        "note": "Direct GI symptoms",
    },
    {
        "query": "I have a skin rash that's red and itchy",
        "expected": "Dermatology",
        "note": "Clear dermatology presentation",
    },
    {
        "query": "I've been feeling very anxious and having panic attacks",
        "expected": "Psychiatry",
        "note": "Mental health keywords",
    },
    {
        "query": "My knee is swollen and I can't bend it after a soccer game",
        "expected": "Orthopedics",
        "note": "Sports injury presentation",
    },
    # ── Colloquial language (how patients actually talk) ─────
    {
        "query": "It feels like an elephant is sitting on my chest when I climb stairs",
        "expected": "Cardiology",
        "note": "Metaphorical description, no clinical keywords",
    },
    {
        "query": "I've been getting sharp pains behind my eyes with flashing lights",
        "expected": "Neurology",
        "note": "Migraine with aura — could be confused with ophthalmology",
    },
    {
        "query": "My stomach feels like it's on fire after I eat spicy food",
        "expected": "Gastroenterology",
        "note": "Colloquial acid reflux description",
    },
    {
        "query": "I can't stop coughing and I feel like I'm breathing through a straw",
        "expected": "Pulmonology",
        "note": "Asthma/breathing difficulty in patient language",
    },
    {
        "query": "I threw out my back lifting boxes and now I can't stand up straight",
        "expected": "Orthopedics",
        "note": "Colloquial back injury description",
    },
    {
        "query": "I've been feeling really down and I don't enjoy anything anymore",
        "expected": "Psychiatry",
        "note": "Depression described without clinical terms",
    },
    {
        "query": "My ear is killing me and everything sounds muffled",
        "expected": "ENT",
        "note": "Ear infection in patient language",
    },
    # ── Ambiguous symptoms (multi-specialty overlap) ─────────
    {
        "query": "I feel dizzy and my heart races when I stand up",
        "expected": "Cardiology",
        "note": "Could be cardiology or neurology — palpitations lean cardiology",
    },
    {
        "query": "I'm always tired and I've been gaining weight for no reason",
        "expected": "Endocrinology",
        "note": "Could be thyroid, depression, or other — metabolic signals lean endo",
    },
    {
        "query": "I have a sore throat and my nose is completely blocked",
        "expected": "ENT",
        "note": "Common cold symptoms but persistent → ENT",
    },
    # ── Edge cases ───────────────────────────────────────────
    {
        "query": "I see little squiggly lines floating across my vision and sometimes bright flashes",
        "expected": "Ophthalmology",
        "note": "Floaters + flashes — retinal concern, not migraine (no pain)",
    },
    {
        "query": "I'm having crushing chest pain and my left arm feels numb",
        "expected": "Cardiology",
        "note": "Emergency symptoms — should match severity guide too",
    },
    {
        "query": "My hands are shaking and I can't hold my coffee cup steady",
        "expected": "Neurology",
        "note": "Tremor description without clinical terms",
    },
    {
        "query": "I keep getting these bumps under my skin on my face that really hurt",
        "expected": "Dermatology",
        "note": "Cystic acne described colloquially",
    },
    {
        "query": "I'm peeing a lot more than usual and I'm always thirsty",
        "expected": "Endocrinology",
        "note": "Classic diabetes symptoms in patient language",
    },
]


# ============================================================
# TEST RUNNER
# ============================================================


def main() -> None:
    passed = 0
    failed = 0
    errors: list[str] = []

    print("=" * 68)
    print("RAG Retriever Test Suite")
    print(f"Running {len(TEST_CASES)} test cases")
    print("=" * 68)

    for i, case in enumerate(TEST_CASES, 1):
        query = case["query"]
        expected = case["expected"]
        note = case["note"]

        chunks = retrieve_medical_knowledge(query, match_count=3)

        if not chunks:
            status = "FAIL"
            top_specialty = "No matches"
            top_score = 0.0
            failed += 1
            errors.append(f"  #{i}: Expected {expected}, got no matches — {note}")
        else:
            top = chunks[0]
            top_specialty = top["metadata"].get("specialty_name", "Unknown")
            top_score = top["similarity"]

            if top_specialty == expected:
                status = "PASS"
                passed += 1
            else:
                # Check if expected is in top 3 (partial credit)
                all_specialties = [c["metadata"].get("specialty_name") for c in chunks]
                if expected in all_specialties:
                    status = "SOFT"
                    passed += 1  # Count as pass — LLM will see it
                    errors.append(
                        f"  #{i}: Expected {expected} as #1, got {top_specialty} "
                        f"({top_score:.3f}) — but {expected} is in top 3 — {note}"
                    )
                else:
                    status = "FAIL"
                    failed += 1
                    errors.append(
                        f"  #{i}: Expected {expected}, got {top_specialty} "
                        f"({top_score:.3f}) — {note}"
                    )

        icon = {"PASS": "+", "SOFT": "~", "FAIL": "x"}[status]
        print(
            f"  [{icon}] #{i:2d} | {status:4s} | "
            f"{top_specialty:20s} ({top_score:.3f}) | "
            f"{query[:50]}..."
        )

    # ── Summary ──────────────────────────────────────────────
    print()
    print("=" * 68)
    total = passed + failed
    print(f"Results: {passed}/{total} passed, {failed}/{total} failed")
    print(f"Pass rate: {passed / total * 100:.0f}%")

    if errors:
        print(f"\nIssues ({len(errors)}):")
        for err in errors:
            print(err)

    print("=" * 68)


if __name__ == "__main__":
    main()
