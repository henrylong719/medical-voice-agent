-- ============================================================
-- Medical Voice Agent — Seed Data
-- Phase 1: Realistic test data for development
-- Run this AFTER 001_schema.sql
-- ============================================================

-- ============================================================
-- SPECIALTIES (10)
-- These represent the departments our clinic offers.
-- ============================================================

INSERT INTO specialties (id, name, description) VALUES
    ('a1000000-0000-0000-0000-000000000001', 'Cardiology',
     'Heart and cardiovascular system — chest pain, palpitations, blood pressure, heart disease'),
    ('a1000000-0000-0000-0000-000000000002', 'Neurology',
     'Brain and nervous system — headaches, migraines, seizures, dizziness, numbness'),
    ('a1000000-0000-0000-0000-000000000003', 'Orthopedics',
     'Bones, joints, and muscles — fractures, joint pain, back pain, sports injuries'),
    ('a1000000-0000-0000-0000-000000000004', 'Dermatology',
     'Skin, hair, and nails — rashes, acne, eczema, moles, skin infections'),
    ('a1000000-0000-0000-0000-000000000005', 'Gastroenterology',
     'Digestive system — stomach pain, acid reflux, IBS, nausea, bowel issues'),
    ('a1000000-0000-0000-0000-000000000006', 'Ophthalmology',
     'Eyes and vision — blurry vision, eye pain, vision loss, eye infections'),
    ('a1000000-0000-0000-0000-000000000007', 'Psychiatry',
     'Mental health — anxiety, depression, insomnia, stress, mood disorders'),
    ('a1000000-0000-0000-0000-000000000008', 'Pulmonology',
     'Lungs and respiratory system — coughing, shortness of breath, asthma, wheezing'),
    ('a1000000-0000-0000-0000-000000000009', 'Endocrinology',
     'Hormones and metabolism — diabetes, thyroid issues, fatigue, weight changes'),
    ('a1000000-0000-0000-0000-000000000010', 'ENT',
     'Ear, nose, and throat — ear pain, sore throat, sinus issues, hearing loss, tonsillitis');


-- ============================================================
-- SYMPTOM-SPECIALTY MAPPINGS (50+)
-- These power the keyword-based triage in Phase 2.
-- 
-- Weight: how strongly this symptom indicates this specialty
--   1.0  = very strong indicator (chest pain → Cardiology)
--   0.7  = moderate indicator (fatigue → Endocrinology)
--   0.4  = weak indicator (could be many things)
--
-- Follow-up questions: what the agent should ask to narrow down
-- ============================================================

-- Cardiology
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

-- Orthopedics
INSERT INTO symptom_specialty_map (specialty_id, symptom, weight, follow_up_questions) VALUES
    ('a1000000-0000-0000-0000-000000000003', 'joint pain', 0.9,
     ARRAY['Which joint is affected?', 'Did it start after an injury?', 'Is there any swelling?']),
    ('a1000000-0000-0000-0000-000000000003', 'back pain', 0.85,
     ARRAY['Is it upper or lower back?', 'Does it radiate down your legs?', 'Did it start after lifting something heavy?']),
    ('a1000000-0000-0000-0000-000000000003', 'knee pain', 0.95,
     ARRAY['Can you bend and straighten it fully?', 'Does it swell?', 'Did it start after an injury or gradually?']),
    ('a1000000-0000-0000-0000-000000000003', 'shoulder pain', 0.9,
     ARRAY['Can you raise your arm above your head?', 'Does it hurt more at night?', 'Did you injure it?']),
    ('a1000000-0000-0000-0000-000000000003', 'sports injury', 1.0,
     ARRAY['What sport were you playing?', 'Which body part was injured?', 'Can you put weight on it?']),
    ('a1000000-0000-0000-0000-000000000003', 'fracture', 1.0,
     ARRAY['Which bone do you think is broken?', 'Can you move the affected area?', 'Is there visible swelling or bruising?']);

-- Dermatology
INSERT INTO symptom_specialty_map (specialty_id, symptom, weight, follow_up_questions) VALUES
    ('a1000000-0000-0000-0000-000000000004', 'skin rash', 1.0,
     ARRAY['Where on your body is the rash?', 'Is it itchy?', 'When did it first appear?']),
    ('a1000000-0000-0000-0000-000000000004', 'acne', 1.0,
     ARRAY['Where is the acne located?', 'How long have you had it?', 'Have you tried any treatments?']),
    ('a1000000-0000-0000-0000-000000000004', 'eczema', 0.95,
     ARRAY['Where are the affected areas?', 'Does it flare up seasonally?', 'Do you have any known allergies?']),
    ('a1000000-0000-0000-0000-000000000004', 'mole changes', 0.9,
     ARRAY['Has the mole changed in size, shape, or color?', 'Is it painful or itchy?', 'How quickly has it changed?']),
    ('a1000000-0000-0000-0000-000000000004', 'hair loss', 0.8,
     ARRAY['Is it patchy or overall thinning?', 'When did you first notice it?', 'Have you been under a lot of stress?']);

-- Gastroenterology
INSERT INTO symptom_specialty_map (specialty_id, symptom, weight, follow_up_questions) VALUES
    ('a1000000-0000-0000-0000-000000000005', 'stomach pain', 0.85,
     ARRAY['Where exactly is the pain?', 'Does it get worse after eating?', 'How long has it been going on?']),
    ('a1000000-0000-0000-0000-000000000005', 'acid reflux', 1.0,
     ARRAY['How often does it happen?', 'Is it worse at night?', 'Have you tried antacids?']),
    ('a1000000-0000-0000-0000-000000000005', 'nausea', 0.6,
     ARRAY['Have you been vomiting?', 'Is it constant or does it come and go?', 'Could you be pregnant?']),
    ('a1000000-0000-0000-0000-000000000005', 'bloating', 0.7,
     ARRAY['Is it related to eating specific foods?', 'How often does it happen?', 'Do you also have changes in bowel habits?']),
    ('a1000000-0000-0000-0000-000000000005', 'blood in stool', 0.95,
     ARRAY['What color is the blood — bright red or dark?', 'How often has this happened?', 'Do you have any abdominal pain?']),
    ('a1000000-0000-0000-0000-000000000005', 'difficulty swallowing', 0.8,
     ARRAY['Is it with solids, liquids, or both?', 'Does food feel stuck?', 'How long has this been happening?']);

-- Ophthalmology
INSERT INTO symptom_specialty_map (specialty_id, symptom, weight, follow_up_questions) VALUES
    ('a1000000-0000-0000-0000-000000000006', 'blurry vision', 0.9,
     ARRAY['Is it in one eye or both?', 'Did it come on suddenly or gradually?', 'Do you wear glasses or contacts?']),
    ('a1000000-0000-0000-0000-000000000006', 'eye pain', 0.9,
     ARRAY['Is it a sharp pain or dull ache?', 'Is the eye red?', 'Are you sensitive to light?']),
    ('a1000000-0000-0000-0000-000000000006', 'vision loss', 1.0,
     ARRAY['Is it partial or complete?', 'Did it happen suddenly?', 'Is it in one eye or both?']),
    ('a1000000-0000-0000-0000-000000000006', 'eye redness', 0.75,
     ARRAY['Is there any discharge?', 'Is it itchy?', 'Have you been around anyone with pink eye?']),
    ('a1000000-0000-0000-0000-000000000006', 'floaters', 0.7,
     ARRAY['When did you first notice them?', 'Have they increased recently?', 'Do you also see flashing lights?']);

-- Psychiatry
INSERT INTO symptom_specialty_map (specialty_id, symptom, weight, follow_up_questions) VALUES
    ('a1000000-0000-0000-0000-000000000007', 'anxiety', 0.95,
     ARRAY['How long have you been feeling anxious?', 'Does it interfere with your daily activities?', 'Have you had any panic attacks?']),
    ('a1000000-0000-0000-0000-000000000007', 'depression', 0.95,
     ARRAY['How long have you been feeling this way?', 'Have you lost interest in activities you used to enjoy?', 'How is your sleep?']),
    ('a1000000-0000-0000-0000-000000000007', 'insomnia', 0.8,
     ARRAY['Do you have trouble falling asleep or staying asleep?', 'How long has this been going on?', 'Have you tried any sleep aids?']),
    ('a1000000-0000-0000-0000-000000000007', 'mood swings', 0.75,
     ARRAY['How extreme are the mood changes?', 'How quickly do they shift?', 'Has anyone in your family had bipolar disorder?']),
    ('a1000000-0000-0000-0000-000000000007', 'panic attacks', 1.0,
     ARRAY['How often do they happen?', 'What do they feel like?', 'Do you know what triggers them?']);

-- Pulmonology
INSERT INTO symptom_specialty_map (specialty_id, symptom, weight, follow_up_questions) VALUES
    ('a1000000-0000-0000-0000-000000000008', 'chronic cough', 0.9,
     ARRAY['How long have you had the cough?', 'Is it dry or productive?', 'Do you cough up blood?']),
    ('a1000000-0000-0000-0000-000000000008', 'wheezing', 0.9,
     ARRAY['When does the wheezing happen?', 'Do you have asthma?', 'Are you exposed to any irritants?']),
    ('a1000000-0000-0000-0000-000000000008', 'asthma', 1.0,
     ARRAY['Do you use an inhaler?', 'How often do you have attacks?', 'What triggers your asthma?']),
    ('a1000000-0000-0000-0000-000000000008', 'difficulty breathing', 0.85,
     ARRAY['Is it constant or does it come and go?', 'Does it get worse lying down?', 'Do you smoke?']);

-- Endocrinology
INSERT INTO symptom_specialty_map (specialty_id, symptom, weight, follow_up_questions) VALUES
    ('a1000000-0000-0000-0000-000000000009', 'diabetes', 1.0,
     ARRAY['Have you been diagnosed with diabetes?', 'Are you on any medication?', 'Do you monitor your blood sugar?']),
    ('a1000000-0000-0000-0000-000000000009', 'thyroid problems', 1.0,
     ARRAY['Do you have hypothyroid or hyperthyroid?', 'Are you on thyroid medication?', 'Have you had your levels checked recently?']),
    ('a1000000-0000-0000-0000-000000000009', 'unexplained weight changes', 0.7,
     ARRAY['Is it weight gain or loss?', 'Over what time period?', 'Have your eating habits changed?']),
    ('a1000000-0000-0000-0000-000000000009', 'excessive thirst', 0.75,
     ARRAY['How much water are you drinking daily?', 'Are you also urinating more frequently?', 'Do you have a family history of diabetes?']),
    ('a1000000-0000-0000-0000-000000000009', 'fatigue', 0.4,
     ARRAY['How long have you been feeling tired?', 'Does sleep help?', 'Have you had any blood work done recently?']);

-- ENT
INSERT INTO symptom_specialty_map (specialty_id, symptom, weight, follow_up_questions) VALUES
    ('a1000000-0000-0000-0000-000000000010', 'ear pain', 0.9,
     ARRAY['Is it one ear or both?', 'Do you have any discharge?', 'Has your hearing changed?']),
    ('a1000000-0000-0000-0000-000000000010', 'sore throat', 0.75,
     ARRAY['How long have you had it?', 'Do you have a fever?', 'Is it painful to swallow?']),
    ('a1000000-0000-0000-0000-000000000010', 'sinus problems', 0.9,
     ARRAY['Do you have facial pain or pressure?', 'Is there nasal discharge?', 'How long has this been going on?']),
    ('a1000000-0000-0000-0000-000000000010', 'hearing loss', 1.0,
     ARRAY['Is it in one ear or both?', 'Did it come on suddenly or gradually?', 'Do you have ringing in your ears?']),
    ('a1000000-0000-0000-0000-000000000010', 'tonsillitis', 0.95,
     ARRAY['Do you have a fever?', 'Are your tonsils visibly swollen?', 'How many times have you had tonsillitis this year?']),
    ('a1000000-0000-0000-0000-000000000010', 'nosebleeds', 0.7,
     ARRAY['How often do they happen?', 'Are they from one nostril or both?', 'Do they last a long time?']);


-- ============================================================
-- DOCTORS (8)
-- Each has specialties and weekly availability schedules.
-- ============================================================

-- Doctor IDs (fixed UUIDs for easy reference in seed data)
-- d1 through d8

INSERT INTO doctors (id, full_name, email, phone, image_url) VALUES
    ('dd000000-0000-0000-0000-000000000001', 'Dr. Sarah Chen',      'sarah.chen@clinic.com',      '555-0101', 'https://placeholder.co/200x200?text=Dr+Chen'),
    ('dd000000-0000-0000-0000-000000000002', 'Dr. Michael Rodriguez','michael.rodriguez@clinic.com','555-0102', 'https://placeholder.co/200x200?text=Dr+Rodriguez'),
    ('dd000000-0000-0000-0000-000000000003', 'Dr. Emily Watson',    'emily.watson@clinic.com',     '555-0103', 'https://placeholder.co/200x200?text=Dr+Watson'),
    ('dd000000-0000-0000-0000-000000000004', 'Dr. James Kim',       'james.kim@clinic.com',        '555-0104', 'https://placeholder.co/200x200?text=Dr+Kim'),
    ('dd000000-0000-0000-0000-000000000005', 'Dr. Lisa Patel',      'lisa.patel@clinic.com',       '555-0105', 'https://placeholder.co/200x200?text=Dr+Patel'),
    ('dd000000-0000-0000-0000-000000000006', 'Dr. David Thompson',  'david.thompson@clinic.com',   '555-0106', 'https://placeholder.co/200x200?text=Dr+Thompson'),
    ('dd000000-0000-0000-0000-000000000007', 'Dr. Maria Garcia',    'maria.garcia@clinic.com',     '555-0107', 'https://placeholder.co/200x200?text=Dr+Garcia'),
    ('dd000000-0000-0000-0000-000000000008', 'Dr. Robert Lee',      'robert.lee@clinic.com',       '555-0108', 'https://placeholder.co/200x200?text=Dr+Lee');

-- ============================================================
-- DOCTOR SPECIALTIES
-- Some doctors have multiple specialties.
-- ============================================================

INSERT INTO doctor_specialties (doctor_id, specialty_id) VALUES
    -- Dr. Chen: Cardiology + Pulmonology (heart-lung overlap is common)
    ('dd000000-0000-0000-0000-000000000001', 'a1000000-0000-0000-0000-000000000001'),
    ('dd000000-0000-0000-0000-000000000001', 'a1000000-0000-0000-0000-000000000008'),
    -- Dr. Rodriguez: Neurology
    ('dd000000-0000-0000-0000-000000000002', 'a1000000-0000-0000-0000-000000000002'),
    -- Dr. Watson: Orthopedics
    ('dd000000-0000-0000-0000-000000000003', 'a1000000-0000-0000-0000-000000000003'),
    -- Dr. Kim: Dermatology
    ('dd000000-0000-0000-0000-000000000004', 'a1000000-0000-0000-0000-000000000004'),
    -- Dr. Patel: Gastroenterology + Endocrinology (GI-metabolic overlap)
    ('dd000000-0000-0000-0000-000000000005', 'a1000000-0000-0000-0000-000000000005'),
    ('dd000000-0000-0000-0000-000000000005', 'a1000000-0000-0000-0000-000000000009'),
    -- Dr. Thompson: Ophthalmology
    ('dd000000-0000-0000-0000-000000000006', 'a1000000-0000-0000-0000-000000000006'),
    -- Dr. Garcia: Psychiatry
    ('dd000000-0000-0000-0000-000000000007', 'a1000000-0000-0000-0000-000000000007'),
    -- Dr. Lee: ENT + Pulmonology (airway overlap)
    ('dd000000-0000-0000-0000-000000000008', 'a1000000-0000-0000-0000-000000000010'),
    ('dd000000-0000-0000-0000-000000000008', 'a1000000-0000-0000-0000-000000000008');


-- ============================================================
-- DOCTOR AVAILABILITY (weekly templates)
-- Varied schedules to make slot computation interesting.
-- ============================================================

-- Dr. Chen (Cardiology/Pulmonology): Mon-Wed-Fri mornings, Tue afternoons
INSERT INTO doctor_availability (doctor_id, day_of_week, start_time, end_time, slot_duration_min) VALUES
    ('dd000000-0000-0000-0000-000000000001', 'monday',    '09:00', '13:00', 30),
    ('dd000000-0000-0000-0000-000000000001', 'tuesday',   '13:00', '17:00', 30),
    ('dd000000-0000-0000-0000-000000000001', 'wednesday', '09:00', '13:00', 30),
    ('dd000000-0000-0000-0000-000000000001', 'friday',    '09:00', '12:00', 30);

-- Dr. Rodriguez (Neurology): Mon-Thu, full mornings, longer 45-min slots
INSERT INTO doctor_availability (doctor_id, day_of_week, start_time, end_time, slot_duration_min) VALUES
    ('dd000000-0000-0000-0000-000000000002', 'monday',    '08:00', '12:00', 45),
    ('dd000000-0000-0000-0000-000000000002', 'tuesday',   '08:00', '12:00', 45),
    ('dd000000-0000-0000-0000-000000000002', 'wednesday', '08:00', '12:00', 45),
    ('dd000000-0000-0000-0000-000000000002', 'thursday',  '08:00', '12:00', 45);

-- Dr. Watson (Orthopedics): Mon-Fri, split schedule (morning + afternoon)
INSERT INTO doctor_availability (doctor_id, day_of_week, start_time, end_time, slot_duration_min) VALUES
    ('dd000000-0000-0000-0000-000000000003', 'monday',    '09:00', '12:00', 30),
    ('dd000000-0000-0000-0000-000000000003', 'monday',    '14:00', '17:00', 30),
    ('dd000000-0000-0000-0000-000000000003', 'wednesday', '09:00', '12:00', 30),
    ('dd000000-0000-0000-0000-000000000003', 'wednesday', '14:00', '17:00', 30),
    ('dd000000-0000-0000-0000-000000000003', 'friday',    '09:00', '14:00', 30);

-- Dr. Kim (Dermatology): Tue-Thu-Sat, 20-min slots (dermatology visits are often shorter)
INSERT INTO doctor_availability (doctor_id, day_of_week, start_time, end_time, slot_duration_min) VALUES
    ('dd000000-0000-0000-0000-000000000004', 'tuesday',   '09:00', '13:00', 20),
    ('dd000000-0000-0000-0000-000000000004', 'thursday',  '09:00', '13:00', 20),
    ('dd000000-0000-0000-0000-000000000004', 'saturday',  '10:00', '14:00', 20);

-- Dr. Patel (Gastro/Endocrinology): Mon-Wed-Fri afternoons, longer 60-min initial consults
INSERT INTO doctor_availability (doctor_id, day_of_week, start_time, end_time, slot_duration_min) VALUES
    ('dd000000-0000-0000-0000-000000000005', 'monday',    '13:00', '17:00', 60),
    ('dd000000-0000-0000-0000-000000000005', 'wednesday', '13:00', '17:00', 60),
    ('dd000000-0000-0000-0000-000000000005', 'friday',    '13:00', '17:00', 60);

-- Dr. Thompson (Ophthalmology): Mon-Tue-Thu, standard 30-min
INSERT INTO doctor_availability (doctor_id, day_of_week, start_time, end_time, slot_duration_min) VALUES
    ('dd000000-0000-0000-0000-000000000006', 'monday',    '10:00', '15:00', 30),
    ('dd000000-0000-0000-0000-000000000006', 'tuesday',   '10:00', '15:00', 30),
    ('dd000000-0000-0000-0000-000000000006', 'thursday',  '10:00', '15:00', 30);

-- Dr. Garcia (Psychiatry): Mon-Fri, 60-min sessions (standard for psychiatry)
INSERT INTO doctor_availability (doctor_id, day_of_week, start_time, end_time, slot_duration_min) VALUES
    ('dd000000-0000-0000-0000-000000000007', 'monday',    '09:00', '16:00', 60),
    ('dd000000-0000-0000-0000-000000000007', 'tuesday',   '09:00', '16:00', 60),
    ('dd000000-0000-0000-0000-000000000007', 'wednesday', '09:00', '16:00', 60),
    ('dd000000-0000-0000-0000-000000000007', 'thursday',  '09:00', '16:00', 60),
    ('dd000000-0000-0000-0000-000000000007', 'friday',    '09:00', '13:00', 60);

-- Dr. Lee (ENT/Pulmonology): Tue-Wed-Thu-Fri, mixed schedule
INSERT INTO doctor_availability (doctor_id, day_of_week, start_time, end_time, slot_duration_min) VALUES
    ('dd000000-0000-0000-0000-000000000008', 'tuesday',   '08:00', '12:00', 30),
    ('dd000000-0000-0000-0000-000000000008', 'wednesday', '13:00', '17:00', 30),
    ('dd000000-0000-0000-0000-000000000008', 'thursday',  '08:00', '12:00', 30),
    ('dd000000-0000-0000-0000-000000000008', 'friday',    '08:00', '12:00', 30);


-- ============================================================
-- DOCTOR BLOCKS (sample time-off)
-- A few realistic blocks to test the slot engine.
-- Using dates in early April 2026 so they're in the future.
-- ============================================================

INSERT INTO doctor_blocks (doctor_id, start_at, end_at, reason) VALUES
    -- Dr. Chen: out for a conference on Monday April 6
    ('dd000000-0000-0000-0000-000000000001', '2026-04-06 00:00:00+00', '2026-04-06 23:59:59+00', 'Medical Conference'),
    -- Dr. Watson: vacation week of April 13-17
    ('dd000000-0000-0000-0000-000000000003', '2026-04-13 00:00:00+00', '2026-04-17 23:59:59+00', 'Vacation'),
    -- Dr. Garcia: morning block on April 8 (personal appointment)
    ('dd000000-0000-0000-0000-000000000007', '2026-04-08 09:00:00+00', '2026-04-08 12:00:00+00', 'Personal');


-- ============================================================
-- SAMPLE PATIENTS (5)
-- For testing appointment flows.
-- ============================================================

INSERT INTO patients (id, uin, full_name, phone, email, allergies) VALUES
    ('aa100000-0000-0000-0000-000000000001', '123456789', 'Alice Johnson',  '555-1001', 'alice.j@university.edu',  ARRAY['penicillin']),
    ('aa100000-0000-0000-0000-000000000002', '234567890', 'Bob Martinez',   '555-1002', 'bob.m@university.edu',    ARRAY[]::TEXT[]),
    ('aa100000-0000-0000-0000-000000000003', '345678901', 'Carol Williams', '555-1003', 'carol.w@university.edu',  ARRAY['latex', 'sulfa drugs']),
    ('aa100000-0000-0000-0000-000000000004', '456789012', 'Dan Brown',      '555-1004', 'dan.b@university.edu',    ARRAY['peanuts']),
    ('aa100000-0000-0000-0000-000000000005', '567890123', 'Emma Davis',     '555-1005', 'emma.d@university.edu',   ARRAY[]::TEXT[]);


-- ============================================================
-- SAMPLE APPOINTMENTS (3)
-- Some pre-existing appointments for testing reschedule/cancel.
-- Using dates in early April 2026.
-- ============================================================

INSERT INTO appointments (patient_id, doctor_id, specialty_id, start_at, end_at, status, reason, severity_rating) VALUES
    -- Alice has a cardiology appointment on April 7 at 9:00 AM
    ('aa100000-0000-0000-0000-000000000001', 'dd000000-0000-0000-0000-000000000001',
     'a1000000-0000-0000-0000-000000000001', '2026-04-07 09:00:00+00', '2026-04-07 09:30:00+00',
     'scheduled', 'Follow-up for chest pain', 5),
    -- Bob has a neurology appointment on April 7 at 8:00 AM
    ('aa100000-0000-0000-0000-000000000002', 'dd000000-0000-0000-0000-000000000002',
     'a1000000-0000-0000-0000-000000000002', '2026-04-07 08:00:00+00', '2026-04-07 08:45:00+00',
     'scheduled', 'Recurring migraines', 6),
    -- Carol had a completed dermatology appointment
    ('aa100000-0000-0000-0000-000000000003', 'dd000000-0000-0000-0000-000000000004',
     'a1000000-0000-0000-0000-000000000004', '2026-03-20 09:00:00+00', '2026-03-20 09:20:00+00',
     'completed', 'Skin rash evaluation', 3);
