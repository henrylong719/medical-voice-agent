"""
Medical knowledge chunks for RAG-powered triage.

Each chunk is a rich, natural-language passage describing a symptom cluster
for a specific specialty. These are the passages that get embedded and stored
in the medical_knowledge table.

Design principles:
  - Self-contained: each chunk is readable without context from other chunks
  - Variation-rich: includes multiple ways patients might describe the same
    symptoms ("chest tightness", "pressure in my chest", "elephant on my chest")
  - Clinically grounded: mentions related symptoms, severity signals, and
    when to refer
  - Right-sized: ~200–500 tokens per chunk (enough context for good embeddings,
    focused enough for precise retrieval)

These chunks reference specialties from 002_seed.sql using the same UUIDs.
"""

# Specialty UUIDs from 002_seed.sql — keep in sync
SPECIALTY_IDS = {
    "cardiology": "a1000000-0000-0000-0000-000000000001",
    "neurology": "a1000000-0000-0000-0000-000000000002",
    "orthopedics": "a1000000-0000-0000-0000-000000000003",
    "dermatology": "a1000000-0000-0000-0000-000000000004",
    "gastroenterology": "a1000000-0000-0000-0000-000000000005",
    "ophthalmology": "a1000000-0000-0000-0000-000000000006",
    "psychiatry": "a1000000-0000-0000-0000-000000000007",
    "pulmonology": "a1000000-0000-0000-0000-000000000008",
    "endocrinology": "a1000000-0000-0000-0000-000000000009",
    "ent": "a1000000-0000-0000-0000-000000000010",
}


# ============================================================
# CHUNK DEFINITIONS
# ============================================================
# Each dict has:
#   content:  the text passage that gets embedded
#   metadata: structured info for filtering and display
#
# metadata fields:
#   specialty_id:      UUID → specialties table
#   specialty_name:    denormalized for convenience
#   category:          "symptom_cluster" | "severity_guide"
#   severity_keywords: words that hint at urgency
#   source:            where this knowledge came from
# ============================================================

KNOWLEDGE_CHUNKS: list[dict] = [
    # ────────────────────────────────────────────
    # CARDIOLOGY
    # ────────────────────────────────────────────
    {
        "content": (
            "Patients experiencing chest tightness, chest pressure, or a squeezing "
            "sensation in the chest may need cardiology evaluation. This can feel like "
            "something heavy sitting on the chest, a band tightening around the chest, "
            "or a dull ache behind the breastbone. The discomfort often worsens with "
            "physical activity such as climbing stairs, exercising, or walking uphill, "
            "and may improve with rest. Associated symptoms include pain radiating to "
            "the left arm, jaw, neck, or back, shortness of breath, sweating, and "
            "nausea. These symptoms warrant cardiology referral, especially in patients "
            "with risk factors such as high blood pressure, high cholesterol, diabetes, "
            "smoking, or family history of heart disease."
        ),
        "metadata": {
            "specialty_id": SPECIALTY_IDS["cardiology"],
            "specialty_name": "Cardiology",
            "category": "symptom_cluster",
            "severity_keywords": ["chest pain", "radiating pain", "sweating"],
            "source": "clinical_triage_guidelines",
        },
    },
    {
        "content": (
            "Heart palpitations — a sensation of the heart racing, fluttering, "
            "pounding, or skipping beats — may indicate cardiac arrhythmia and should "
            "be evaluated by cardiology. Patients may describe feeling their heartbeat "
            "in their chest, throat, or neck, or say their heart feels like it's doing "
            "flip-flops. Palpitations can occur at rest or during activity. Associated "
            "symptoms include dizziness, lightheadedness, fainting or near-fainting, "
            "shortness of breath, and chest discomfort. Palpitations that are frequent, "
            "prolonged, or accompanied by fainting require prompt evaluation."
        ),
        "metadata": {
            "specialty_id": SPECIALTY_IDS["cardiology"],
            "specialty_name": "Cardiology",
            "category": "symptom_cluster",
            "severity_keywords": ["palpitations", "fainting", "arrhythmia"],
            "source": "clinical_triage_guidelines",
        },
    },
    {
        "content": (
            "Swelling in the ankles, feet, or legs — especially when accompanied by "
            "shortness of breath, fatigue, or rapid weight gain — can be a sign of "
            "heart failure or other cardiovascular conditions. Patients may notice "
            "their shoes feeling tight, socks leaving deep marks, or puffy ankles at "
            "the end of the day. The swelling may be worse after standing for long "
            "periods and may improve with elevation. When combined with difficulty "
            "breathing while lying flat, waking up at night gasping for air, or "
            "persistent coughing, these symptoms suggest fluid retention related to "
            "cardiac issues and should be assessed by cardiology."
        ),
        "metadata": {
            "specialty_id": SPECIALTY_IDS["cardiology"],
            "specialty_name": "Cardiology",
            "category": "symptom_cluster",
            "severity_keywords": ["swelling", "shortness of breath", "heart failure"],
            "source": "clinical_triage_guidelines",
        },
    },
    # ────────────────────────────────────────────
    # NEUROLOGY
    # ────────────────────────────────────────────
    {
        "content": (
            "Severe or recurring headaches, including migraines, may need neurology "
            "evaluation. Migraines typically present as intense, throbbing pain on one "
            "side of the head, often accompanied by nausea, vomiting, and sensitivity "
            "to light and sound. Some patients experience aura before the headache — "
            "visual disturbances like flashing lights, zigzag lines, blind spots, or "
            "shimmering spots. Patients may describe sharp pains behind the eyes, a "
            "headache that feels like pressure building inside the skull, or a "
            "throbbing that pulses with the heartbeat. Headaches that are sudden and "
            "severe (thunderclap headache), progressively worsening over weeks, or "
            "accompanied by fever, stiff neck, or confusion require urgent evaluation."
        ),
        "metadata": {
            "specialty_id": SPECIALTY_IDS["neurology"],
            "specialty_name": "Neurology",
            "category": "symptom_cluster",
            "severity_keywords": ["thunderclap", "worst headache", "sudden severe"],
            "source": "clinical_triage_guidelines",
        },
    },
    {
        "content": (
            "Numbness, tingling, or loss of sensation in the hands, feet, arms, or "
            "legs may indicate neurological conditions and should be evaluated by "
            "neurology. Patients often describe a pins-and-needles feeling, a burning "
            "or prickling sensation, or say their limbs feel like they have fallen "
            "asleep and won't wake up. The numbness may be constant or come and go, "
            "and may affect one side of the body or both. When accompanied by muscle "
            "weakness, difficulty walking, loss of coordination, or changes in bladder "
            "or bowel control, these symptoms may suggest conditions such as "
            "neuropathy, multiple sclerosis, or spinal cord issues. Sudden numbness "
            "on one side of the body, especially with facial drooping or difficulty "
            "speaking, requires emergency evaluation for possible stroke."
        ),
        "metadata": {
            "specialty_id": SPECIALTY_IDS["neurology"],
            "specialty_name": "Neurology",
            "category": "symptom_cluster",
            "severity_keywords": ["stroke", "facial drooping", "sudden numbness"],
            "source": "clinical_triage_guidelines",
        },
    },
    {
        "content": (
            "Seizures, tremors, or involuntary movements should be evaluated by "
            "neurology. Seizures may present as uncontrollable shaking or convulsions, "
            "brief episodes of staring or unresponsiveness, or sudden confusion. "
            "Tremors — rhythmic shaking of the hands, arms, head, or legs — may occur "
            "at rest or during movement and can interfere with daily activities like "
            "writing, eating, or holding objects. Patients may also experience memory "
            "problems, difficulty concentrating, confusion, or cognitive changes that "
            "are getting progressively worse. These symptoms can indicate conditions "
            "such as epilepsy, Parkinson's disease, or other neurological disorders."
        ),
        "metadata": {
            "specialty_id": SPECIALTY_IDS["neurology"],
            "specialty_name": "Neurology",
            "category": "symptom_cluster",
            "severity_keywords": ["seizures", "convulsions", "memory loss"],
            "source": "clinical_triage_guidelines",
        },
    },
    {
        "content": (
            "Migraine with aura often involves sharp or stabbing pain behind one "
            "or both eyes, pressure behind the eye sockets, or a deep ache in the "
            "eye area that precedes or accompanies a severe headache. Before the "
            "headache begins, patients may experience visual disturbances known as "
            "aura: flashing lights, shimmering or sparkling spots, zigzag or "
            "jagged lines across the visual field, temporary blind spots, or tunnel "
            "vision. Some patients describe seeing halos around lights or having "
            "their vision go blurry in one eye. These visual symptoms typically "
            "last 20 to 60 minutes and are followed by an intense headache with "
            "nausea and light sensitivity. Pain behind the eyes with visual "
            "disturbances should be evaluated by neurology, as these are classic "
            "migraine-with-aura symptoms rather than primary eye conditions."
        ),
        "metadata": {
            "specialty_id": SPECIALTY_IDS["neurology"],
            "specialty_name": "Neurology",
            "category": "symptom_cluster",
            "severity_keywords": [
                "aura",
                "visual disturbance",
                "eye pain with headache",
            ],
            "source": "clinical_triage_guidelines",
        },
    },
    # ────────────────────────────────────────────
    # ORTHOPEDICS
    # ────────────────────────────────────────────
    {
        "content": (
            "Joint pain, stiffness, or swelling in the knees, shoulders, hips, or "
            "elbows may need orthopedic evaluation. Patients may describe aching "
            "joints that are worse in the morning or after sitting for long periods, "
            "a grinding or popping sensation when moving the joint, or joints that "
            "feel stiff and hard to bend. The pain may worsen with activity and "
            "improve with rest, or it may be constant. Swelling, redness, or warmth "
            "around a joint can indicate inflammation or injury. Sports injuries "
            "involving joints — such as a twisted knee, dislocated shoulder, or "
            "sprained ankle — also fall under orthopedic care."
        ),
        "metadata": {
            "specialty_id": SPECIALTY_IDS["orthopedics"],
            "specialty_name": "Orthopedics",
            "category": "symptom_cluster",
            "severity_keywords": ["dislocation", "unable to move", "sports injury"],
            "source": "clinical_triage_guidelines",
        },
    },
    {
        "content": (
            "Back pain, neck pain, or spinal issues should be evaluated by "
            "orthopedics. Patients may describe lower back pain that radiates down "
            "one or both legs (sciatica), upper back pain between the shoulder blades, "
            "or neck pain with stiffness. The pain may have started after lifting "
            "something heavy, a fall, or a car accident, or it may have come on "
            "gradually over time. Patients might say they threw out their back, "
            "feel like their back locked up, or have pain that shoots down their "
            "leg like an electric shock. When back pain is accompanied by numbness "
            "in the legs, difficulty walking, or loss of bladder or bowel control, "
            "urgent evaluation is needed."
        ),
        "metadata": {
            "specialty_id": SPECIALTY_IDS["orthopedics"],
            "specialty_name": "Orthopedics",
            "category": "symptom_cluster",
            "severity_keywords": ["can't walk", "bladder control", "fracture"],
            "source": "clinical_triage_guidelines",
        },
    },
    {
        "content": (
            "Suspected fractures or broken bones require orthopedic evaluation. "
            "Patients may describe intense pain after a fall, collision, or impact, "
            "with visible swelling, bruising, or deformity at the injury site. The "
            "affected area may be impossible to move or bear weight on. Patients "
            "might say they heard a crack or snap, or that the bone feels out of "
            "place. Common fracture locations include the wrist (from falling on an "
            "outstretched hand), ankle (from twisting), and collarbone (from direct "
            "impact). Any suspected broken bone should be immobilized and assessed "
            "as soon as possible."
        ),
        "metadata": {
            "specialty_id": SPECIALTY_IDS["orthopedics"],
            "specialty_name": "Orthopedics",
            "category": "symptom_cluster",
            "severity_keywords": ["fracture", "broken bone", "can't bear weight"],
            "source": "clinical_triage_guidelines",
        },
    },
    # ────────────────────────────────────────────
    # DERMATOLOGY
    # ────────────────────────────────────────────
    {
        "content": (
            "Skin rashes, irritation, or allergic reactions on the skin should be "
            "evaluated by dermatology. Patients may describe red, itchy patches, "
            "hives or welts that come and go, dry flaky skin, or blistering. The "
            "rash may be localized to one area or spread across the body. Eczema "
            "typically presents as itchy, inflamed patches in the creases of the "
            "elbows, behind the knees, or on the hands and face. Psoriasis appears "
            "as thick, scaly, silvery patches often on the elbows, knees, and scalp. "
            "Contact dermatitis occurs when the skin reacts to something it touched, "
            "like a new soap, jewelry, or plant. Rashes that are rapidly spreading, "
            "painful, or accompanied by fever may need more urgent attention."
        ),
        "metadata": {
            "specialty_id": SPECIALTY_IDS["dermatology"],
            "specialty_name": "Dermatology",
            "category": "symptom_cluster",
            "severity_keywords": ["spreading rash", "blistering", "fever with rash"],
            "source": "clinical_triage_guidelines",
        },
    },
    {
        "content": (
            "Acne, skin breakouts, or persistent skin blemishes may need "
            "dermatology evaluation, especially when over-the-counter treatments have "
            "not helped. Patients may describe painful cystic bumps under the skin, "
            "whiteheads and blackheads that won't go away, or breakouts on the face, "
            "chest, or back. Concerns about moles should also be directed to "
            "dermatology — especially moles that have changed in size, shape, color, "
            "or texture, moles that are asymmetric or have irregular borders, or new "
            "moles that appeared after age 30. Hair loss — whether patchy (alopecia "
            "areata), overall thinning, or receding — can also be assessed by "
            "dermatology to determine the cause and treatment options."
        ),
        "metadata": {
            "specialty_id": SPECIALTY_IDS["dermatology"],
            "specialty_name": "Dermatology",
            "category": "symptom_cluster",
            "severity_keywords": ["changing mole", "rapid growth", "asymmetric"],
            "source": "clinical_triage_guidelines",
        },
    },
    # ────────────────────────────────────────────
    # GASTROENTEROLOGY
    # ────────────────────────────────────────────
    {
        "content": (
            "Stomach pain, abdominal discomfort, or digestive issues may need "
            "gastroenterology evaluation. Patients may describe a burning feeling "
            "in the upper abdomen, cramping or bloating after eating, sharp or "
            "stabbing pains in the belly, or a constant dull ache. Acid reflux or "
            "heartburn — a burning sensation rising from the stomach into the chest "
            "or throat — is a common complaint, especially when it occurs frequently "
            "or disrupts sleep. Nausea, vomiting, loss of appetite, or feeling full "
            "quickly can also indicate digestive system problems. Symptoms that are "
            "persistent, worsening, or associated with unexplained weight loss "
            "warrant evaluation."
        ),
        "metadata": {
            "specialty_id": SPECIALTY_IDS["gastroenterology"],
            "specialty_name": "Gastroenterology",
            "category": "symptom_cluster",
            "severity_keywords": [
                "vomiting blood",
                "severe abdominal pain",
                "weight loss",
            ],
            "source": "clinical_triage_guidelines",
        },
    },
    {
        "content": (
            "Changes in bowel habits, blood in the stool, or difficulty swallowing "
            "should be evaluated by gastroenterology. Patients may report chronic "
            "diarrhea, constipation, or alternating between the two. Irritable bowel "
            "syndrome (IBS) symptoms include cramping, bloating, gas, and changes in "
            "stool consistency. Blood in the stool — whether bright red (rectal "
            "bleeding) or dark and tarry (suggesting upper GI bleeding) — should "
            "always be evaluated. Difficulty swallowing (dysphagia), where food feels "
            "stuck in the throat or chest, or pain when swallowing, may indicate "
            "esophageal issues. These symptoms require gastroenterology referral, "
            "especially in patients over 50 or with a family history of colorectal "
            "conditions."
        ),
        "metadata": {
            "specialty_id": SPECIALTY_IDS["gastroenterology"],
            "specialty_name": "Gastroenterology",
            "category": "symptom_cluster",
            "severity_keywords": ["blood in stool", "tarry stool", "can't swallow"],
            "source": "clinical_triage_guidelines",
        },
    },
    # ────────────────────────────────────────────
    # OPHTHALMOLOGY
    # ────────────────────────────────────────────
    {
        "content": (
            "Vision problems, eye pain, or changes in eyesight should be evaluated "
            "by ophthalmology. Patients may describe blurry or cloudy vision, "
            "difficulty seeing at night, double vision, or a gradual loss of "
            "peripheral (side) vision. Sudden vision loss — partial or complete, in "
            "one or both eyes — requires urgent evaluation. Eye pain can be a sharp, "
            "stabbing sensation, a dull ache behind the eye, or a feeling of pressure "
            "in the eye. Light sensitivity (photophobia), excessive tearing, or a "
            "gritty sensation like something is in the eye are also reasons to see "
            "ophthalmology."
        ),
        "metadata": {
            "specialty_id": SPECIALTY_IDS["ophthalmology"],
            "specialty_name": "Ophthalmology",
            "category": "symptom_cluster",
            "severity_keywords": ["sudden vision loss", "eye injury", "double vision"],
            "source": "clinical_triage_guidelines",
        },
    },
    {
        "content": (
            "Eye floaters, flashing lights, or redness and discharge should be "
            "assessed by ophthalmology. Floaters appear as small spots, squiggly "
            "lines, or cobweb-like shapes drifting across the field of vision. While "
            "often harmless, a sudden increase in floaters — especially when "
            "accompanied by flashing lights or a shadow or curtain across part of the "
            "vision — can indicate a retinal detachment and requires urgent "
            "evaluation. Red eyes with discharge may suggest conjunctivitis (pink "
            "eye), which can be viral, bacterial, or allergic. Eye infections that "
            "involve pain, vision changes, or swelling around the eye need prompt "
            "attention."
        ),
        "metadata": {
            "specialty_id": SPECIALTY_IDS["ophthalmology"],
            "specialty_name": "Ophthalmology",
            "category": "symptom_cluster",
            "severity_keywords": [
                "retinal detachment",
                "sudden floaters",
                "vision curtain",
            ],
            "source": "clinical_triage_guidelines",
        },
    },
    # ────────────────────────────────────────────
    # PSYCHIATRY
    # ────────────────────────────────────────────
    {
        "content": (
            "Anxiety, persistent worry, or panic attacks may benefit from psychiatry "
            "evaluation. Patients may describe feeling constantly on edge, a sense of "
            "dread or impending doom, racing thoughts that are hard to control, or "
            "physical symptoms like a racing heart, sweating, trembling, and "
            "shortness of breath during anxiety episodes. Panic attacks involve "
            "sudden, intense fear with chest tightness, numbness or tingling, feeling "
            "detached from reality, or a fear of losing control. Social anxiety may "
            "present as intense nervousness about social situations, avoidance of "
            "gatherings, or fear of being judged. When anxiety interferes with work, "
            "relationships, or daily functioning, professional support can help."
        ),
        "metadata": {
            "specialty_id": SPECIALTY_IDS["psychiatry"],
            "specialty_name": "Psychiatry",
            "category": "symptom_cluster",
            "severity_keywords": ["panic attack", "can't function", "crisis"],
            "source": "clinical_triage_guidelines",
        },
    },
    {
        "content": (
            "Depression, persistent sadness, or mood changes may need psychiatry "
            "evaluation. Patients may describe feeling sad, empty, or hopeless for "
            "weeks or months, losing interest in activities they used to enjoy, "
            "withdrawing from friends and family, or feeling worthless or guilty. "
            "Physical symptoms of depression include changes in sleep (sleeping too "
            "much or insomnia), changes in appetite or weight, fatigue and low energy, "
            "and difficulty concentrating or making decisions. Mood swings — "
            "alternating between periods of high energy, euphoria, and impulsive "
            "behavior and periods of deep depression — may suggest bipolar disorder. "
            "Insomnia that is persistent and not related to other medical conditions "
            "can also be addressed through psychiatric care."
        ),
        "metadata": {
            "specialty_id": SPECIALTY_IDS["psychiatry"],
            "specialty_name": "Psychiatry",
            "category": "symptom_cluster",
            "severity_keywords": ["hopelessness", "can't sleep", "mood swings"],
            "source": "clinical_triage_guidelines",
        },
    },
    # ────────────────────────────────────────────
    # PULMONOLOGY
    # ────────────────────────────────────────────
    {
        "content": (
            "Chronic cough, wheezing, or breathing difficulties may need pulmonology "
            "evaluation. Patients may describe a cough that has lasted more than "
            "three weeks, wheezing or whistling sounds when breathing, feeling like "
            "they can't get enough air, or shortness of breath that gets worse with "
            "activity. Asthma symptoms include episodes of wheezing, coughing "
            "(especially at night or early morning), chest tightness, and difficulty "
            "breathing triggered by exercise, cold air, allergens, or stress. Patients "
            "who say they get winded easily, can't catch their breath going up stairs, "
            "or feel like they're breathing through a straw should be evaluated by "
            "pulmonology. A chronic cough that produces mucus or blood requires "
            "prompt evaluation."
        ),
        "metadata": {
            "specialty_id": SPECIALTY_IDS["pulmonology"],
            "specialty_name": "Pulmonology",
            "category": "symptom_cluster",
            "severity_keywords": ["coughing blood", "can't breathe", "severe asthma"],
            "source": "clinical_triage_guidelines",
        },
    },
    # ────────────────────────────────────────────
    # ENDOCRINOLOGY
    # ────────────────────────────────────────────
    {
        "content": (
            "Diabetes management, blood sugar issues, or thyroid problems should be "
            "evaluated by endocrinology. Patients with diabetes may report difficulty "
            "controlling blood sugar levels, frequent high or low blood sugar episodes, "
            "increased thirst and urination, blurry vision, slow-healing wounds, or "
            "tingling in the hands and feet. Thyroid issues can present as unexplained "
            "weight gain or loss, fatigue, feeling cold or hot all the time, hair "
            "thinning, dry skin, or a visible swelling in the neck (goiter). Patients "
            "may also describe unexplained changes in weight, energy levels, or "
            "metabolism that don't have an obvious cause. Hormonal imbalances "
            "affecting mood, energy, or reproductive health can also fall under "
            "endocrinology."
        ),
        "metadata": {
            "specialty_id": SPECIALTY_IDS["endocrinology"],
            "specialty_name": "Endocrinology",
            "category": "symptom_cluster",
            "severity_keywords": [
                "uncontrolled diabetes",
                "thyroid crisis",
                "severe fatigue",
            ],
            "source": "clinical_triage_guidelines",
        },
    },
    {
        "content": (
            "Patients who are urinating much more frequently than usual, feeling "
            "constantly thirsty no matter how much they drink, or experiencing "
            "unexplained weight loss despite eating normally may have undiagnosed "
            "diabetes or uncontrolled blood sugar and should be evaluated by "
            "endocrinology. Other signs include feeling extremely hungry all the "
            "time, blurry vision that comes and goes, cuts or sores that are slow "
            "to heal, frequent infections, and tingling or numbness in the hands "
            "or feet. Patients may describe needing to get up multiple times at "
            "night to pee, drinking excessive amounts of water, or feeling "
            "exhausted even after sleeping well. A family history of diabetes "
            "increases the likelihood. These symptoms warrant blood sugar testing "
            "and endocrinology evaluation."
        ),
        "metadata": {
            "specialty_id": SPECIALTY_IDS["endocrinology"],
            "specialty_name": "Endocrinology",
            "category": "symptom_cluster",
            "severity_keywords": [
                "frequent urination",
                "excessive thirst",
                "undiagnosed diabetes",
            ],
            "source": "clinical_triage_guidelines",
        },
    },
    # ────────────────────────────────────────────
    # ENT (Ear, Nose & Throat)
    # ────────────────────────────────────────────
    {
        "content": (
            "Ear pain, hearing problems, or ringing in the ears may need ENT "
            "evaluation. Patients may describe a sharp or dull ache inside the ear, "
            "a feeling of fullness or pressure in the ear, muffled hearing, or "
            "ringing, buzzing, or humming sounds (tinnitus). Ear infections can cause "
            "pain, drainage from the ear, and temporary hearing loss. Sudden hearing "
            "loss — especially in one ear — requires urgent evaluation. Recurrent ear "
            "infections, persistent fluid behind the eardrum, or dizziness and "
            "balance problems associated with ear issues are also reasons to see ENT."
        ),
        "metadata": {
            "specialty_id": SPECIALTY_IDS["ent"],
            "specialty_name": "ENT",
            "category": "symptom_cluster",
            "severity_keywords": ["sudden hearing loss", "ear drainage", "vertigo"],
            "source": "clinical_triage_guidelines",
        },
    },
    {
        "content": (
            "Sore throat, sinus problems, or nasal issues should be evaluated by "
            "ENT. Patients may describe a persistent sore throat that doesn't improve "
            "with over-the-counter remedies, difficulty or pain when swallowing, a "
            "hoarse or raspy voice that has lasted more than two weeks, or frequent "
            "throat infections (such as tonsillitis or strep throat). Sinus symptoms "
            "include facial pain or pressure around the cheeks, forehead, or between "
            "the eyes, nasal congestion, thick nasal discharge, reduced sense of "
            "smell, and post-nasal drip. Chronic sinusitis — sinus symptoms lasting "
            "more than 12 weeks — and frequent nosebleeds should also be assessed "
            "by ENT."
        ),
        "metadata": {
            "specialty_id": SPECIALTY_IDS["ent"],
            "specialty_name": "ENT",
            "category": "symptom_cluster",
            "severity_keywords": [
                "can't swallow",
                "voice changes",
                "chronic sinusitis",
            ],
            "source": "clinical_triage_guidelines",
        },
    },
    # ────────────────────────────────────────────
    # CROSS-SPECIALTY: SEVERITY GUIDES
    # ────────────────────────────────────────────
    # These chunks help the LLM reason about urgency across
    # specialties. They won't map to a single specialty but
    # provide context for severity assessment.
    # ────────────────────────────────────────────
    {
        "content": (
            "Certain symptom combinations are red flags that require emergency care, "
            "not a scheduled appointment. These include: chest pain with shortness of "
            "breath, jaw or arm pain, and sweating (possible heart attack); sudden "
            "severe headache described as the worst headache of their life (possible "
            "aneurysm); sudden numbness or weakness on one side of the body, facial "
            "drooping, or difficulty speaking (possible stroke); difficulty breathing "
            "that is severe or rapidly worsening; high fever with stiff neck and "
            "confusion (possible meningitis); severe allergic reaction with throat "
            "swelling and difficulty breathing (anaphylaxis). In any of these "
            "situations, the patient should be directed to call 911 or go to the "
            "nearest emergency room immediately."
        ),
        "metadata": {
            "specialty_id": None,
            "specialty_name": None,
            "category": "severity_guide",
            "severity_keywords": [
                "heart attack",
                "stroke",
                "aneurysm",
                "meningitis",
                "anaphylaxis",
                "emergency",
            ],
            "source": "emergency_triage_protocol",
        },
    },
]
