1. symptom_specialty_map

```sql

INSERT INTO symptom_specialty_map (specialty_id, symptom, weight, follow_up_questions) VALUES
    ('a1000000-0000-0000-0000-000000000001', 'chest pain', 1.0,
     ARRAY['Is the pain sharp or dull?', 'Does it get worse with physical activity?', 'Does the pain radiate to your arm, jaw, or back?']),
    ('a1000000-0000-0000-0000-000000000001', 'heart palpitations', 0.95,
     ARRAY['How often do you feel them?', 'Do they happen at rest or during activity?', 'Do you feel dizzy when they occur?']),
    ('a1000000-0000-0000-0000-000000000001', 'high blood pressure', 0.9,
     ARRAY['Have you had your blood pressure measured recently?', 'Are you currently on any medication for it?']),
    ('a1000000-0000-0000-0000-000000000001', 'shortness of breath', 0.7,
     ARRAY['Does it happen at rest or only during activity?', 'Do you also have chest pain?', 'How long has this been going on?']),
    ('a1000000-0000-0000-0000-000000000001', 'swollen ankles', 0.6,
     ARRAY['Is the swelling in both ankles or just one?', 'Does it get worse during the day?', 'Do you also have shortness of breath?']),
    ('a1000000-0000-0000-0000-000000000001', 'dizziness', 0.4,
     ARRAY['Do you feel lightheaded or like the room is spinning?', 'Does it happen when you stand up?', 'Have you fainted?']);

-- Neurology
INSERT INTO symptom_specialty_map (specialty_id, symptom, weight, follow_up_questions) VALUES
    ('a1000000-0000-0000-0000-000000000002', 'severe headache', 0.9,
     ARRAY['Where is the pain located?', 'Is it throbbing or constant?', 'Do you see any visual changes like flashing lights?']),
    ('a1000000-0000-0000-0000-000000000002', 'migraine', 1.0,
     ARRAY['How often do you get them?', 'Do you experience aura before the headache?', 'What triggers them?']),
    ('a1000000-0000-0000-0000-000000000002', 'numbness or tingling', 0.85,
     ARRAY['Where do you feel the numbness?', 'Is it constant or does it come and go?', 'Did it start suddenly?']),
    ('a1000000-0000-0000-0000-000000000002', 'seizures', 1.0,
     ARRAY['When was your last seizure?', 'Are you on any seizure medication?', 'How long do they typically last?']),
    ('a1000000-0000-0000-0000-000000000002', 'memory problems', 0.75,
     ARRAY['How long have you noticed memory issues?', 'Is it short-term or long-term memory?', 'Has it been getting worse?']),
    ('a1000000-0000-0000-0000-000000000002', 'tremors', 0.85,
     ARRAY['Which body part trembles?', 'Does it happen at rest or during movement?', 'When did you first notice it?']);

```

Weight tells the triage system how strongly this symptom points to this specialty. When a patient mentions multiple symptoms, the agent sums up (or averages) the weights to rank which specialty is the best match.
Follow-up questions are what the agent asks after the patient mentions that symptom, regardless of weight. If a patient says "I'm dizzy," the agent should ask "Do you feel lightheaded or like the room is spinning?" — whether the weight is 0.4 or 1.0.
They're on the same row because they both belong to the same symptom-specialty pair. Think of each row as answering: "When a patient mentions this symptom, how likely is it to be this specialty (weight), and what should we ask to learn more (follow-up questions)?"

Here's a concrete example of how they'll work together in Phase 2. A patient says "I'm dizzy and I have chest pain":

1. The agent matches "dizziness" → Cardiology (0.4) and "chest pain" → Cardiology (1.0)

2. Combined score for Cardiology: 1.4 — highest match, so recommend Cardiology

3. The agent then asks follow-up questions from both matched rows to gather more detail before confirming the specialty

The weight drives the ranking decision. The follow-up questions drive the conversation. They just happen to live together because they're both properties of a specific symptom-specialty relationship.

---

Q: what if the patient says chest pain and severe headache ?

"Chest pain" matches Cardiology with weight 1.0. "Severe headache" matches Neurology with weight 0.9. So the scores are close — Cardiology 1.0 vs Neurology 0.9. Which specialty should the agent recommend?
This is where the follow-up questions become critical. Instead of just picking the highest score, the agent should ask follow-up questions from both matches to narrow things down:
From the chest pain row: "Is the pain sharp or dull?", "Does it radiate to your arm?"
From the severe headache row: "Do you see flashing lights?", "Is it throbbing or constant?"
The patient's answers help the agent decide. If they say "the chest pain radiates to my left arm and I feel sweaty" — that's a strong cardiac signal, route to Cardiology. If they say "the chest pain is mild but the headache is blinding with flashing lights" — that sounds more like a migraine, route to Neurology.
There's also a third possibility: the symptoms might be unrelated, and the patient genuinely needs to see both specialists. The agent could say "Based on your symptoms, I'd recommend booking with Cardiology for the chest pain and Neurology for the headaches. Which would you like to schedule first?"

Right now our schema supports all of this — the data is there. The logic for how to handle it will be built in Phase 2 (simple keyword matching with weight comparison) and then made much smarter in Phase 3 with RAG, where the agent can reason semantically about symptom combinations rather than just adding up weights.

---

Q: we ask follow up questions only when the weights are close or we always ask ?

Option A: Always ask follow-up questions. Every time a symptom is matched, ask the follow-ups. This is the safest approach for a medical context — you gather maximum information. But it makes conversations longer and can feel annoying if the answer is obvious. "I broke my arm playing football" → "What sport were you playing?" feels redundant.

Option B: Only ask when scores are close. If Cardiology scores 1.0 and everything else is below 0.3, just route to Cardiology. But if Cardiology is 1.0 and Neurology is 0.9, ask follow-ups to differentiate. This makes conversations shorter and more natural. The risk is you might miss something — what if the patient said "chest pain" but actually meant heartburn (Gastroenterology)?

Option C: Hybrid — always ask, but adjust how many. If the top specialty is a clear winner (big gap), ask just 1 follow-up to confirm. If scores are close, ask 2–3 follow-ups to differentiate. This balances thoroughness with conversational flow.
For a medical scheduling agent, I'd recommend Option C. Here's why: in a medical context, even a quick confirmation question adds safety. But you don't want to interrogate someone who clearly knows what they need. A patient saying "I need to see my cardiologist for my annual check-up" doesn't need five follow-up questions.
The good news is this is a logic decision, not a schema decision. Our data already supports all three approaches — the follow-up questions are stored per symptom, and the agent code in Phase 2 will decide how many to ask based on the score gap.

Q: how to deal with the situation when the scores are close eventually?

What we have now: The triage tool returns all matching specialties ranked by score. The LLM sees something like "Cardiology: 1.7, Pulmonology: 1.5" and it can reason about it. Because we wrote the system prompt to say "present the specialty recommendation," Claude will typically say something like "Based on your symptoms, I'd recommend either Cardiology or Pulmonology. Cardiology seems most relevant because of the chest pain — does that sound right?" The LLM is naturally good at this kind of ambiguity handling.

So the agent handles it through conversation, not code. It presents the top options, explains why, and lets the patient choose. That's actually the ideal behavior for a scheduling assistant — it shouldn't be making medical decisions silently.
If you wanted to make this more explicit, you could add a threshold in the triage tool — say, if the top two specialties are within 0.3 of each other, include a flag like "NOTE: Multiple specialties matched closely. Present options to the patient." But honestly, the LLM handles this well without extra logic. And in Phase 3 when we switch to RAG, the confidence signals will be richer anyway.

Q: Why would cosine similarity be preferred over Euclidean (L2) distance for comparing text embeddings?

Think about what each metric actually measures. Euclidean distance measures how far apart two points are in space — it cares about both direction and magnitude. Cosine similarity measures the angle between two vectors — it only cares about direction.
Why does that matter for embeddings? When an embedding model converts "chest tightness" into a 1536-dimensional vector, the direction that vector points encodes the semantic meaning. But the length (magnitude) can vary based on factors like text length, punctuation, or model quirks that have nothing to do with meaning.
So if "chest tightness" produces a vector pointing in roughly the same direction as "pressure in my chest" but with different magnitudes, cosine similarity says "these mean the same thing" (angle ≈ 0, similarity ≈ 1.0). Euclidean distance might say "these are somewhat far apart" because the points are at different distances from the origin, even though they're semantically identical.
That said — in practice, most modern embedding models normalize their output vectors to unit length (magnitude = 1), which means L2 distance and cosine similarity become mathematically equivalent. pgvector supports both, and we'll use cosine distance (<=> operator) because it's the conventional choice and works correctly even if you ever swap to a model that doesn't normalize.

Q: Should we chunk by individual symptom (one chunk per symptom), by specialty (one big chunk per specialty covering all its symptoms), or by symptom cluster (groups of related symptoms)?

Right now, your symptom_specialty_map has ~50 rows of short keyword entries. For RAG, we need to replace those with richer text passages that an embedding model can work with. The question is: how do we organize those passages?

Option A: One chunk per symptom (very small chunks)
Each symptom gets its own rich text passage:

Chunk 1: "Patients experiencing chest tightness, pressure, or squeezing in the chest area, especially during physical activity or emotional stress, may need cardiology evaluation."
Chunk 2: "Heart palpitations — a racing, fluttering, or pounding heartbeat — can indicate cardiac arrhythmia. May occur at rest or during activity."
Chunk 3: "Swollen ankles, particularly when accompanied by shortness of breath, may indicate heart failure or other cardiovascular conditions."

You'd end up with ~50 chunks (one for each symptom mapping you already have).

Option B: One chunk per specialty (very large chunks)
All of Cardiology goes into a single big passage:

Chunk 1 (Cardiology): "The Cardiology department treats conditions of the heart and cardiovascular system. Patients may present with chest tightness, pressure, or squeezing sensations, heart palpitations, high blood pressure, shortness of breath during exertion, swollen ankles, dizziness upon standing... [continues for 500+ words covering every cardiology symptom, when to refer, severity indicators, etc.]"

You'd end up with just ~10 chunks (one per specialty).

Option C: Symptom clusters (medium chunks)
Group related symptoms that tend to appear together:

Chunk 1 (Cardiac chest symptoms): "Patients experiencing chest tightness, pressure, squeezing, or pain — especially during physical activity or emotional stress — may need cardiology evaluation. Associated symptoms include pain radiating to the left arm, jaw, or back, shortness of breath, and sweating."
Chunk 2 (Cardiac rhythm symptoms): "Heart palpitations, racing heartbeat, irregular pulse, or a fluttering sensation in the chest may indicate cardiac arrhythmia. These symptoms may occur at rest or during activity and can be accompanied by dizziness or fainting."

You'd end up with maybe 20–30 chunks.

Now here's the tradeoff to think about. When a patient says "I get winded going up stairs," we embed that sentence and find the closest chunks. Consider what happens in each scenario:

Chunks too small (Option A): The retrieval might work well (precise matches), but each chunk has very little context. The LLM gets back "shortness of breath during exertion may need cardiology evaluation" — but it doesn't know about related symptoms it should ask about, like chest pain or swollen ankles.
Chunks too big (Option B): The embedding for a 500-word passage is an average of everything in it. That Cardiology mega-chunk covers chest pain, palpitations, swollen ankles, dizziness — the embedding gets pulled in many directions. It becomes a "jack of all trades" vector that's moderately close to everything but strongly close to nothing. A patient saying "I get winded going up stairs" might match Pulmonology's mega-chunk just as well as Cardiology's.
Symptom clusters (Option C): You get the best of both worlds. The "cardiac chest symptoms" chunk is focused enough to embed well, but rich enough to give the LLM context about related symptoms and follow-up questions.

Q: why are we using vector(1536) specifically? What would need to change if we picked a different embedding model?

For our use case — embedding medical symptom descriptions and patient queries — here's what we care about:
What we need:

Good semantic understanding of medical/health language
Fast enough for real-time triage (patient is waiting on the phone in later phases)
Cheap enough that we're not blowing our cost budget (remember, we're targeting $0.03–0.06 per call)
Reasonable dimensionality (more dimensions = better quality but more storage and slower search)

The main contenders:

OpenAI text-embedding-3-small — 1536 dimensions, very cheap ($0.02 per million tokens), fast, solid general-purpose quality. The "workhorse" option.
OpenAI text-embedding-3-large — 3072 dimensions, better quality on benchmarks, costs 6.5x more. Overkill for our ~30 chunks.
Voyage voyage-3 — strong general-purpose model, competitive quality, 1024 dimensions. Good alternative.
Voyage voyage-clinical-1 — specifically trained on clinical/medical text. Would likely give the best results for our medical symptom matching. But it's a specialized model with more limited availability.

For a learning project with ~30 chunks where we're doing medical symptom matching, text-embedding-3-small is the pragmatic choice: it's cheap, fast, well-documented, and widely used so you'll find plenty of examples and debugging help. The quality difference between it and more expensive models is marginal at our scale.
That said, if this were a production medical system, I'd seriously evaluate voyage-clinical-1 — domain-specific embedding models can meaningfully outperform general-purpose ones on specialized text. That's a good concept to remember: embedding model selection is a tradeoff between cost, speed, quality, and domain fit.
One more thing worth noting: text-embedding-3-small actually supports dimension reduction. You can request 512 or 768 dimensions instead of the full 1536, trading some quality for faster search and less storage. We'll stick with 1536 since our dataset is tiny, but it's a lever you'd pull at scale.

Q: Should we completely replace the keyword search, or run both and combine the results (hybrid search)? (triage_symptoms)

Think about it this way — imagine a patient says "I have diabetes." Your keyword-based symptom_specialty_map has an exact entry for "diabetes" → Endocrinology with weight 1.0. That's a perfect, high-confidence match. The semantic search would also match it, but it would return a broader chunk about diabetes, thyroid, and hormonal issues with a similarity score of maybe 0.75.
For exact keyword hits, the old system is actually more precise. For fuzzy, natural language descriptions like "it feels like an elephant sitting on my chest," semantic search wins by a mile.
Hybrid search combines both — run the keyword match AND the semantic search, then let the LLM see results from both. This gives you the precision of keywords when they hit, plus the flexibility of semantics when they don't.
This is a common pattern in production RAG systems. Pure semantic search sounds elegant, but hybrid almost always performs better in practice.

Q: Imagine a patient calls in and says "I want to reschedule my cardiology appointment." With your current single-agent setup, the agent has to figure out the right sequence on its own from one big prompt. With the multi-agent supervisor approach, what do you think the sequence of agent handoffs would look like? Which agents would be involved and in what order?

Your steps 1–2 (identify the patient, look up their record) → that's the Intake Agent's job. Steps 3–5 (find the existing appointment, find new slots, book) → that's all the Scheduling Agent.
But here's the interesting part — notice how the Triage Agent gets skipped entirely. The patient said "reschedule my cardiology appointment" — they already have a specialty, they already have an appointment. No symptom collection needed.
That's exactly the kind of decision the Supervisor makes. It looks at the state and asks: "Is the patient identified? Do we know what they want? Do we need triage, or can we go straight to scheduling?" In this case:

Supervisor sees patient_id is None → routes to Intake Agent
Intake Agent collects demographic details first, identifies the patient, and only falls back to stronger identifiers if needed, then hands back to Supervisor
Supervisor sees patient is identified, intent is "reschedule" (no triage needed) → routes to Scheduling Agent
Scheduling Agent finds the existing appointment, cancels it, finds new slots, books the new one

For a new booking flow, it would be: Supervisor → Intake → Supervisor → Triage → Supervisor → Scheduling. Triage only enters the picture when there are symptoms to analyze.
This is why the supervisor earns its keep — it handles the non-linear routing that would be messy to encode in a single prompt.

Q: why the messages needs to read by all agents?

You're right that each agent only cares about the part relevant to its job. The Scheduling Agent doesn't need to deeply understand the symptom discussion that happened during triage. That's what the other state fields are for — specialty_id, patient_id, etc. carry the conclusions forward so agents don't have to re-read the whole history.
But the agents still need the full message list for two practical reasons:
Reason 1: Conversational context. The patient is having one continuous conversation. If the Scheduling Agent can't see what was said before, it might re-ask things the patient already answered. Imagine this:

Patient: "I have headaches and my name is Sarah Connor and my birthday is October 26, 1985"
Intake Agent handles identification, hands off to Triage
Triage Agent: "Hi! What symptoms are you experiencing?"

The patient just told us about headaches! If Triage can't see the earlier messages, it asks again and the patient feels like they're talking to a broken system. Remember — from the patient's perspective, they're talking to one agent, not three.
Reason 2: The LLM needs conversation structure. When we invoke a sub-agent, we're calling an LLM with a system prompt + messages. The LLM needs to see the message history to generate a coherent next response. If you only pass it the last message, it has no context for what's going on.

But here's the nuance you're sensing: each agent's system prompt acts as a filter. The Scheduling Agent sees all messages but its prompt says "you handle scheduling" — so it naturally focuses on the scheduling-relevant parts and ignores the symptom discussion. The state fields (specialty_id, patient_id) are the structured handoff data. The messages are the conversational continuity.
One current limitation in the codebase: if identity stays ambiguous after demographics, phone, and stronger identifiers, the assistant can explain that staff help is needed, but there is not yet a dedicated human-handoff implementation.
Think of it like a relay race — the baton (state fields) carries the key data forward, but each runner (agent) can also look back and see the whole track (messages) if they need to understand what happened before them.

## Phase 3

Q: Look at the intent field. The Supervisor sets it by analyzing what the patient said. Can you give me an example of a patient message where the Supervisor would have trouble deciding the intent — where it's ambiguous?

Yes, exactly! "I'd like to chat with you" doesn't signal any of the three intents — they haven't said anything about booking, rescheduling, or cancelling. So intent stays None and the Supervisor would need to ask: "Sure! Are you looking to book a new appointment, reschedule an existing one, or something else?"
Other ambiguous examples: "I'm not feeling well" (sounds like it could be a booking, but maybe they just want to talk), or "I called last week about my appointment" (reschedule? cancel? just checking details?).
The None state is essentially the Supervisor saying "I don't have enough information to route yet — I need to ask a follow-up question."
