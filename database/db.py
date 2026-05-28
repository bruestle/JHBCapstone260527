# ═══════════════════════════════════════════════════════════════════════════════
# SEEDING METHODOLOGY
# ═══════════════════════════════════════════════════════════════════════════════
#
# This module seeds the healthcare database with realistic mock data on demand.
# The following methodology governs how each category of data is generated:
#
# PATIENTS (100 total, IDs 1000–1099, all "P" last names)
#   Each patient is assigned a random birthdate yielding an age of 25–85 years.
#   Age distribution is weighted toward middle-aged and elderly patients to
#   reflect a realistic chronic-care outpatient population:
#       30 %  young adults  (25–44 years)
#       50 %  middle-aged   (45–69 years)
#       20 %  seniors       (70–85 years)
#   The birthdate is passed to the LLM profile generator so that chronic
#   conditions, medications, and vitals baselines are age-appropriate.
#
# DOCTORS (6 total, IDs 100–105, multi-specialty)
#   Doctor assignment follows an 80/20 split:
#       80 %  of patients see a single primary doctor for all visits.
#       20 %  of patients are assigned two doctors (one per appointment).
#
# MEDICAL RECORDS (30 per patient — 3 per year × 10 years)
#   Records span the ten calendar years ending on the reset date.
#   Each year's three records fall on randomly chosen business days (weekdays)
#   within that calendar year.  For the reset year itself, only weekdays
#   strictly before the reset date are eligible — no future-dated records are
#   ever created.
#   Notes are generated locally in one of four rotating format styles:
#       SOAP, bullet-list, narrative lines, structured key-value.
#   Patient profiles (conditions, medications, vitals) are created by one
#   OpenAI call per patient and then used to synthesise all 30 notes locally.
#
# APPOINTMENTS (2 per patient)
#   Appointments are scheduled within a 4-week weekday-only window that begins
#   on the database reset date (covering the next ~20 business days).
#   Each patient receives exactly 2 appointments on randomly chosen weekdays
#   within that window.  Doctor assignment follows the 80/20 split above:
#   single-doctor patients see the same doctor for both appointments;
#   two-doctor patients see one doctor each.
#
# VECTOR STORE (ChromaDB — collection "patient_history")
#   After all medical records are seeded they are chunked (480-char window,
#   120-char overlap, word-boundary splits) and embedded into a ChromaDB
#   collection for semantic search by the admin AI agent.
#
# ═══════════════════════════════════════════════════════════════════════════════

import os
import random
import sqlite3
import json
import uuid
from functools import lru_cache
from collections.abc import Callable
from contextlib import contextmanager
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Iterator

DB_DEFAULT = "database/healthcare.db"
CHROMA_DEFAULT = "database/chroma"
CHROMA_COLLECTION = "patient_history"

_SEED_PATIENT_FIRST_NAMES = [
    "James", "Maria", "Elena", "Noah", "Ava", "Liam", "Sophia", "Mason", "Isabella", "Ethan",
    "Mia", "Lucas", "Charlotte", "Henry", "Amelia", "Oliver", "Harper", "Benjamin", "Ella", "Jackson",
    "Scarlett", "Sebastian", "Grace", "Aiden", "Chloe", "Daniel", "Lily", "Matthew", "Aria", "David",
    "Emma", "Logan", "Olivia", "Jacob", "Emily", "Michael", "Abigail", "Alexander", "Sofia", "William",
    "Avery", "Joseph", "Evelyn", "Samuel", "Camila", "Daniel", "Madison", "Carter", "Luna", "Owen",
    "Hannah", "Gabriel", "Nora", "Anthony", "Zoey", "Isaac", "Leah", "Dylan", "Stella", "Nathan",
    "Violet", "Caleb", "Hazel", "Ryan", "Aurora", "Andrew", "Savannah", "Julian", "Audrey", "Eli",
    "Brooklyn", "Thomas", "Bella", "Charles", "Claire", "Christopher", "Skylar", "Josiah", "Lucy", "Isaiah",
    "Anna", "Jeremiah", "Caroline", "Hudson", "Genesis", "Levi", "Aaliyah", "Aaron", "Kennedy", "Adrian",
    "Sarah", "Jonathan", "Allison", "Hunter", "Hailey", "Cameron", "Natalie", "Santiago", "Paisley", "Ezekiel",
]

_SEED_PATIENT_LAST_NAMES = [
    "Pace", "Pacheco", "Padgett", "Padilla", "Pagan", "Page", "Paige", "Painter", "Palacios", "Palma",
    "Palmer", "Paredes", "Park", "Parker", "Parks", "Parra", "Parrish", "Parsons", "Pate", "Patel",
    "Patino", "Patrick", "Patterson", "Patton", "Paul", "Paulson", "Payne", "Payton", "Paz", "Peacock",
    "Pearce", "Pearson", "Peck", "Pedersen", "Pena", "Penn", "Pennington", "Peoples", "Peralta", "Pereira",
    "Perez", "Perkins", "Perry", "Person", "Peters", "Petersen", "Peterson", "Pettit", "Petty", "Pham",
    "Phan", "Phelps", "Phillips", "Phipps", "Pickett", "Pierce", "Pierre", "Pierson", "Pike", "Pimentel",
    "Pina", "Pineda", "Pinto", "Piper", "Pittman", "Pitts", "Platt", "Plummer", "Poe", "Polanco",
    "Polk", "Pollard", "Pollock", "Ponce", "Poole", "Pope", "Porter", "Portillo", "Posey", "Post",
    "Potter", "Potts", "Powell", "Powers", "Prado", "Prater", "Pratt", "Preston", "Price", "Prieto",
    "Prince", "Pritchard", "Pritchett", "Proctor", "Pruitt", "Pryor", "Puckett", "Pugh", "Pulido", "Purcell",
]


def _build_seed_patients() -> list[tuple[int, str, str]]:
    if len(_SEED_PATIENT_FIRST_NAMES) != 100:
        raise ValueError(f"Expected 100 patient first names, got {len(_SEED_PATIENT_FIRST_NAMES)}")
    if len(_SEED_PATIENT_LAST_NAMES) != 100:
        raise ValueError(f"Expected 100 patient last names, got {len(_SEED_PATIENT_LAST_NAMES)}")

    names = list(zip(_SEED_PATIENT_FIRST_NAMES, _SEED_PATIENT_LAST_NAMES))

    if len(names) != 100:
        raise ValueError(f"Expected 100 patient names, got {len(names)}")

    if any(not last_name.startswith("P") for _, last_name in names):
        raise ValueError("All patient last names must start with 'P'")

    if len(set(names)) != len(names):
        raise ValueError("Duplicate patient first+last names detected")

    return [(1000 + i, first_name, last_name) for i, (first_name, last_name) in enumerate(names)]


SEED_PATIENTS = _build_seed_patients()

SEED_DOCTORS = [
    (100, "Nina", "Dawson", "nephrology"),
    (101, "Sam", "Diaz", "cardiology"),
    (102, "Iris", "Dalton", "general practice"),
    (103, "Leo", "Duncan", "pulmonology"),
    (104, "Maya", "Dorsey", "orthopedics"),
    (105, "Owen", "Delgado", "internal medicine"),
]

VISIT_TYPES = ["follow_up", "lab", "consult", "telehealth", "procedure"]

SINGLE_DOCTOR_SHARE = 0.80
TWO_DOCTOR_SHARE = 0.20

INSURANCE_CARRIERS = [
    "Aetna",
    "Blue Cross",
    "Cigna",
    "Humana",
    "Kaiser",
    "UnitedHealthcare",
]

ADDRESS_STREETS = [
    "Maple Ave",
    "Oak St",
    "Pine Rd",
    "Cedar Ln",
    "Elm Dr",
    "Willow Ct",
    "Lakeview Blvd",
    "Sunset Way",
]

ADDRESS_CITIES = [
    "Riverton",
    "Pinehurst",
    "Fairview",
    "Georgetown",
    "Ashland",
    "Milford",
]

# Backward-compatible names expected by app pages.
PATIENT_SEED = SEED_PATIENTS
DOCTOR_SEED = SEED_DOCTORS


def _db_path() -> Path:
    return Path(os.getenv("APP_DB_PATH", DB_DEFAULT))


def _chroma_path() -> Path:
    return Path(os.getenv("APP_CHROMA_PATH", CHROMA_DEFAULT))


@lru_cache(maxsize=1)
def _get_chroma_collection():
    try:
        import chromadb
        from chromadb.utils import embedding_functions
    except ImportError as exc:
        raise RuntimeError(
            "chromaDB is required for vector indexing/search. Install dependencies and retry."
        ) from exc

    chroma_dir = _chroma_path()
    chroma_dir.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(chroma_dir))
    embedding_fn = embedding_functions.DefaultEmbeddingFunction()
    return client.get_or_create_collection(
        name=CHROMA_COLLECTION,
        embedding_function=embedding_fn,
        metadata={"hnsw:space": "cosine"},
    )


def _chunk_note_text(note: str, *, chunk_size: int = 480, overlap: int = 120) -> list[str]:
    compact = " ".join(note.split())
    if not compact:
        return []

    chunks: list[str] = []
    start = 0
    while start < len(compact):
        window_end = min(len(compact), start + chunk_size)
        split_at = compact.rfind(" ", start + int(chunk_size * 0.55), window_end)
        if split_at <= start:
            split_at = window_end

        chunk = compact[start:split_at].strip()
        if chunk:
            chunks.append(chunk)

        if split_at >= len(compact):
            break
        start = max(0, split_at - overlap)

    return chunks


def clear_patient_history_vectors() -> None:
    collection = _get_chroma_collection()
    try:
        existing = collection.get(where={"record_type": "medical_record_chunk"}, limit=1)
        if existing and existing.get("ids"):
            collection.delete(where={"record_type": "medical_record_chunk"})
    except Exception:  # noqa: BLE001
        pass  # collection may be empty; nothing to clear


def index_medical_record_rows(
    medical_record_rows: list[tuple[str, int, str, str, int | None, str]],
    progress_cb: Callable[[str], None] | None = None,
) -> None:
    collection = _get_chroma_collection()
    ids: list[str] = []
    documents: list[str] = []
    metadatas: list[dict] = []
    record_count = len(medical_record_rows)
    chunk_total = 0

    def flush_batch() -> None:
        if not ids:
            return
        collection.upsert(ids=ids[:], documents=documents[:], metadatas=metadatas[:])
        ids.clear()
        documents.clear()
        metadatas.clear()

    for idx, (record_id, patient_id, note, encounter_type, doctor_id, created_at) in enumerate(medical_record_rows, start=1):
        chunks = _chunk_note_text(note)
        for chunk_index, chunk in enumerate(chunks):
            ids.append(f"{record_id}-chunk-{chunk_index:03d}")
            documents.append(chunk)
            metadatas.append(
                {
                    "record_type": "medical_record_chunk",
                    "record_id": record_id,
                    "patient_id": int(patient_id),
                    "encounter_type": encounter_type,
                    "doctor_id": int(doctor_id) if doctor_id is not None else 0,
                    "created_at": created_at,
                    "chunk_index": chunk_index,
                    "chunk_count": len(chunks),
                }
            )
            chunk_total += 1

        if len(ids) >= 240:
            flush_batch()

        if idx % 20 == 0 or idx == record_count:
            _emit(
                progress_cb,
                f"Sub-step: vector indexing {idx}/{record_count} records ({chunk_total} chunks)",
            )

    flush_batch()
    _emit(progress_cb, f"Sub-step: vector indexing complete ({chunk_total} chunks)")


def sync_patient_history_vectors(
    progress_cb: Callable[[str], None] | None = None,
) -> int:
    """Re-index all medical_records from SQLite into the Chroma vector store.

    Clears existing vectors, reads every record from the DB, chunks and
    re-embeds them. Returns the total number of chunks indexed.
    """
    _emit(progress_cb, "Step: clearing existing medical record vectors")
    clear_patient_history_vectors()

    with connection() as conn:
        rows = conn.execute(
            "SELECT record_id, patient_id, note, encounter_type, doctor_id, created_at FROM medical_records ORDER BY patient_id, created_at"
        ).fetchall()

    medical_record_rows: list[tuple[str, int, str, str, int | None, str]] = [
        (row["record_id"], int(row["patient_id"]), row["note"], row["encounter_type"] or "OfficeVisit", row["doctor_id"], row["created_at"])
        for row in rows
    ]

    _emit(
        progress_cb,
        f"Step: indexing {len(medical_record_rows)} medical record(s) into vector store",
    )
    index_medical_record_rows(medical_record_rows, progress_cb=progress_cb)

    total_chunks = sum(len(_chunk_note_text(row[2])) for row in medical_record_rows)
    _emit(progress_cb, f"Step: vector sync complete — {total_chunks} chunk(s) indexed")
    return total_chunks


def search_patient_history_semantic(patient_id: int, query: str, top_k: int = 6) -> list[dict]:
    cleaned_query = query.strip()
    if not cleaned_query:
        return []

    collection = _get_chroma_collection()

    # Chroma raises if n_results exceeds the number of indexed items for this patient.
    try:
        count_result = collection.get(
            where={"patient_id": int(patient_id)},
            include=[],
        )
        available = len(count_result.get("ids") or [])
    except Exception:  # noqa: BLE001
        available = 0

    if available == 0:
        return []

    n = max(1, min(int(top_k), available))

    results = collection.query(
        query_texts=[cleaned_query],
        n_results=n,
        where={"patient_id": int(patient_id)},
    )

    ids = (results.get("ids") or [[]])[0]
    docs = (results.get("documents") or [[]])[0]
    metadatas = (results.get("metadatas") or [[]])[0]
    distances = (results.get("distances") or [[]])[0]

    rows: list[dict] = []
    for idx, doc in enumerate(docs):
        metadata = metadatas[idx] if idx < len(metadatas) else {}
        distance = distances[idx] if idx < len(distances) else None
        similarity = None if distance is None else max(0.0, 1.0 - float(distance))

        rows.append(
            {
                "id": ids[idx] if idx < len(ids) else "",
                "record_id": metadata.get("record_id", ""),
                "patient_id": metadata.get("patient_id", patient_id),
                "created_at": metadata.get("created_at", ""),
                "chunk_index": metadata.get("chunk_index", 0),
                "similarity": similarity,
                "text": doc,
            }
        )

    return rows


def search_all_patients_history_semantic(query: str, top_k: int = 100) -> list[dict]:
    """Search medical record chunks across ALL patients without a patient_id filter.

    Useful for cross-patient queries such as counting patients on a specific
    medication or finding everyone with a given diagnosis.
    Returns up to *top_k* chunks ordered by relevance.
    """
    cleaned_query = query.strip()
    if not cleaned_query:
        return []

    collection = _get_chroma_collection()

    try:
        available = collection.count()
    except Exception:  # noqa: BLE001
        available = 0

    if available == 0:
        return []

    n = max(1, min(int(top_k), available))

    results = collection.query(
        query_texts=[cleaned_query],
        n_results=n,
    )

    ids = (results.get("ids") or [[]])[0]
    docs = (results.get("documents") or [[]])[0]
    metadatas = (results.get("metadatas") or [[]])[0]
    distances = (results.get("distances") or [[]])[0]

    rows: list[dict] = []
    for idx, doc in enumerate(docs):
        metadata = metadatas[idx] if idx < len(metadatas) else {}
        distance = distances[idx] if idx < len(distances) else None
        similarity = None if distance is None else max(0.0, 1.0 - float(distance))
        rows.append(
            {
                "id": ids[idx] if idx < len(ids) else "",
                "record_id": metadata.get("record_id", ""),
                "patient_id": metadata.get("patient_id", ""),
                "created_at": metadata.get("created_at", ""),
                "chunk_index": metadata.get("chunk_index", 0),
                "similarity": similarity,
                "text": doc,
            }
        )

    return rows


@contextmanager
def connection() -> Iterator[sqlite3.Connection]:
    path = _db_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db() -> None:
    with connection() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS patients (
                patient_id INTEGER PRIMARY KEY,
                first_name TEXT NOT NULL,
                last_name TEXT NOT NULL,
                telephone TEXT,
                address TEXT,
                insurance_carrier TEXT,
                birthdate TEXT
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS doctors (
                doctor_id INTEGER PRIMARY KEY,
                first_name TEXT NOT NULL,
                last_name TEXT NOT NULL,
                specialty TEXT NOT NULL
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS appointments (
                appointment_id TEXT PRIMARY KEY,
                patient_id INTEGER NOT NULL,
                doctor_id INTEGER NOT NULL,
                start_at TEXT NOT NULL,
                end_at TEXT NOT NULL,
                duration_min INTEGER NOT NULL,
                visit_type TEXT NOT NULL,
                status TEXT NOT NULL,
                notes TEXT,
                FOREIGN KEY(patient_id) REFERENCES patients(patient_id),
                FOREIGN KEY(doctor_id) REFERENCES doctors(doctor_id)
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS medical_records (
                record_id      TEXT PRIMARY KEY,
                patient_id     INTEGER NOT NULL,
                note           TEXT NOT NULL,
                encounter_type TEXT NOT NULL DEFAULT 'OfficeVisit',
                doctor_id      INTEGER,
                created_at     TEXT NOT NULL,
                FOREIGN KEY(patient_id) REFERENCES patients(patient_id),
                FOREIGN KEY(doctor_id)  REFERENCES doctors(doctor_id)
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS app_meta (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS encounter_types (
                category TEXT NOT NULL,
                name     TEXT PRIMARY KEY
            )
            """
        )
        _ensure_patient_columns(conn)
        _ensure_medical_record_columns(conn)


def _ensure_patient_columns(conn: sqlite3.Connection) -> None:
    existing = {
        row["name"]
        for row in conn.execute("PRAGMA table_info(patients)").fetchall()
    }

    if "telephone" not in existing:
        conn.execute("ALTER TABLE patients ADD COLUMN telephone TEXT")
    if "address" not in existing:
        conn.execute("ALTER TABLE patients ADD COLUMN address TEXT")
    if "insurance_carrier" not in existing:
        conn.execute("ALTER TABLE patients ADD COLUMN insurance_carrier TEXT")
    if "birthdate" not in existing:
        conn.execute("ALTER TABLE patients ADD COLUMN birthdate TEXT")


def _ensure_medical_record_columns(conn: sqlite3.Connection) -> None:
    existing = {
        row["name"]
        for row in conn.execute("PRAGMA table_info(medical_records)").fetchall()
    }
    if "encounter_type" not in existing:
        conn.execute(
            "ALTER TABLE medical_records ADD COLUMN encounter_type TEXT DEFAULT 'OfficeVisit'"
        )
    if "doctor_id" not in existing:
        conn.execute(
            "ALTER TABLE medical_records ADD COLUMN doctor_id INTEGER REFERENCES doctors(doctor_id)"
        )


def _emit(progress_cb: Callable[[str], None] | None, message: str) -> None:
    if progress_cb:
        progress_cb(message)


def _next_business_day_on_or_after(day_value: date) -> date:
    current = day_value
    while current.weekday() >= 5:
        current += timedelta(days=1)
    return current


def _business_days_for_horizon(start_day: date, horizon_days: int = 31) -> list[date]:
    end_day = start_day + timedelta(days=horizon_days - 1)
    days: list[date] = []
    current = start_day
    while current <= end_day:
        if current.weekday() < 5:
            days.append(current)
        current += timedelta(days=1)
    return days


def is_db_seeded() -> bool:
    init_db()
    with connection() as conn:
        row = conn.execute("SELECT value FROM app_meta WHERE key = 'seeded_at'").fetchone()
    return row is not None


def _set_seed_metadata(seed_start: date) -> None:
    with connection() as conn:
        conn.execute(
            "INSERT OR REPLACE INTO app_meta(key, value) VALUES ('seeded_at', ?)",
            (datetime.now().isoformat(timespec="seconds"),),
        )
        conn.execute(
            "INSERT OR REPLACE INTO app_meta(key, value) VALUES ('seed_start_date', ?)",
            (seed_start.isoformat(),),
        )


def _clear_seed_metadata(conn: sqlite3.Connection) -> None:
    conn.execute("DELETE FROM app_meta WHERE key IN ('seeded_at', 'seed_start_date')")


def _has_legacy_string_ids(conn: sqlite3.Connection) -> bool:
    patient_legacy = conn.execute(
        "SELECT 1 FROM patients WHERE CAST(patient_id AS TEXT) LIKE 'patient-%' LIMIT 1"
    ).fetchone()
    doctor_legacy = conn.execute(
        "SELECT 1 FROM doctors WHERE CAST(doctor_id AS TEXT) LIKE 'doctor-%' LIMIT 1"
    ).fetchone()
    return patient_legacy is not None or doctor_legacy is not None


def _expected_doctor_split(total_patients: int) -> tuple[int, int]:
    one_doctor_target = int(round(total_patients * SINGLE_DOCTOR_SHARE))
    two_doctor_target = total_patients - one_doctor_target
    return one_doctor_target, two_doctor_target


def _validate_patient_doctor_distribution(
    rows: list[tuple[str, int, int, str, str, int, str, str, str]],
    patient_ids: list[int],
) -> bool:
    patient_doctors: dict[int, set[int]] = {patient_id: set() for patient_id in patient_ids}
    for row in rows:
        patient_id = row[1]
        doctor_id = row[2]
        patient_doctors[patient_id].add(doctor_id)

    one_doctor = sum(1 for patient_id in patient_ids if len(patient_doctors[patient_id]) == 1)
    two_doctor = sum(1 for patient_id in patient_ids if len(patient_doctors[patient_id]) == 2)
    over_two = sum(1 for patient_id in patient_ids if len(patient_doctors[patient_id]) > 2)
    zero_doctor = sum(1 for patient_id in patient_ids if len(patient_doctors[patient_id]) == 0)

    expected_one, expected_two = _expected_doctor_split(len(patient_ids))

    return (
        one_doctor == expected_one
        and two_doctor == expected_two
        and over_two == 0
        and zero_doctor == 0
    )


def _build_patient_seed_rows(seed_start: date) -> list[tuple[int, str, str, str, str, str, str]]:
    rng = random.Random(seed_start.toordinal() + 101)
    rows: list[tuple[int, str, str, str, str, str, str]] = []

    # Weighted age pool: 30 % young adults (25-44), 50 % middle-aged (45-69), 20 % seniors (70-85)
    age_pool = (
        [rng.randint(25, 44) for _ in range(30)]
        + [rng.randint(45, 69) for _ in range(50)]
        + [rng.randint(70, 85) for _ in range(20)]
    )
    rng.shuffle(age_pool)

    for i, (patient_id, first_name, last_name) in enumerate(SEED_PATIENTS):
        telephone = f"(555) {rng.randint(200, 989)}-{rng.randint(1000, 9999)}"
        address = f"{rng.randint(100, 9999)} {rng.choice(ADDRESS_STREETS)}, {rng.choice(ADDRESS_CITIES)}"
        insurance_carrier = rng.choice(INSURANCE_CARRIERS)
        age = age_pool[i]
        birth_year = seed_start.year - age
        birth_month = rng.randint(1, 12)
        birth_day = rng.randint(1, 28)
        birthdate = date(birth_year, birth_month, birth_day).isoformat()
        rows.append((patient_id, first_name, last_name, telephone, address, insurance_carrier, birthdate))

    return rows


def _generate_seed_appointments(seed_start: date, progress_cb: Callable[[str], None] | None) -> list[tuple[str, int, int, str, str, int, str, str, str]]:
    """Schedule exactly 2 appointments per patient on random weekdays in a 4-week
    window that starts on seed_start.  Doctor assignment follows the 80/20 split:
    single-doctor patients see the same doctor for both appointments; two-doctor
    patients see one doctor each.
    """
    # Collect all weekdays in the 4-week window (28 calendar days from seed_start)
    window_days: list[date] = []
    current = seed_start
    for _ in range(28):
        if current.weekday() < 5:
            window_days.append(current)
        current += timedelta(days=1)

    patient_ids = [p[0] for p in SEED_PATIENTS]
    doctor_ids = [d[0] for d in SEED_DOCTORS]

    time_slots = [
        (8, 0),
        (8, 30),
        (9, 0),
        (9, 30),
        (10, 0),
        (10, 30),
        (11, 0),
        (11, 30),
        (13, 0),
        (13, 30),
        (14, 0),
        (14, 30),
        (15, 0),
        (15, 30),
        (16, 0),
        (16, 30),
    ]

    rng = random.Random(seed_start.toordinal() + 42)

    # Assign allowed doctors per patient (80 % one doctor, 20 % two doctors)
    shuffled = patient_ids[:]
    rng.shuffle(shuffled)
    expected_one, expected_two = _expected_doctor_split(len(patient_ids))

    allowed_by_patient: dict[int, list[int]] = {}
    for pid in shuffled[:expected_one]:
        allowed_by_patient[pid] = [rng.choice(doctor_ids)]
    for pid in shuffled[expected_one : expected_one + expected_two]:
        allowed_by_patient[pid] = rng.sample(doctor_ids, k=2)

    rows: list[tuple[str, int, int, str, str, int, str, str, str]] = []
    counter = 1

    for patient_id in patient_ids:
        doctors = allowed_by_patient[patient_id]
        # Pick 2 distinct weekdays within the window (sorted chronologically)
        appt_days = sorted(rng.sample(window_days, min(2, len(window_days))))

        for i, appt_day in enumerate(appt_days):
            doctor_id = doctors[i % len(doctors)]
            hour, minute = rng.choice(time_slots)
            duration_min = rng.choice([30, 30, 60])
            start_dt = datetime(appt_day.year, appt_day.month, appt_day.day, hour, minute)
            end_dt = start_dt + timedelta(minutes=duration_min)
            rows.append(
                (
                    f"appointment-seed-{counter:05d}",
                    patient_id,
                    doctor_id,
                    start_dt.strftime("%Y-%m-%d %H:%M"),
                    end_dt.strftime("%Y-%m-%d %H:%M"),
                    duration_min,
                    rng.choice(VISIT_TYPES),
                    "booked",
                    "Seeded mock appointment",
                )
            )
            counter += 1

    _emit(
        progress_cb,
        f"Sub-step: generated {len(rows)} appointment(s) for {len(patient_ids)} patients "
        f"over 4-week window ({len(window_days)} weekdays)",
    )
    return rows


def _build_patient_doctor_counts(
    appointment_rows: list[tuple[str, int, int, str, str, int, str, str, str]],
) -> dict[int, dict[int, int]]:
    counts: dict[int, dict[int, int]] = {}
    for _, patient_id, doctor_id, *_ in appointment_rows:
        patient_counts = counts.setdefault(patient_id, {})
        patient_counts[doctor_id] = patient_counts.get(doctor_id, 0) + 1
    return counts


def _random_business_days_in_year(year: int, before_date: date, k: int, rng: random.Random) -> list[date]:
    """Return k randomly sampled weekdays from *year*, all strictly before *before_date*."""
    start = date(year, 1, 1)
    end = min(date(year, 12, 31), before_date - timedelta(days=1))
    if start > end:
        return []
    pool: list[date] = []
    current = start
    while current <= end:
        if current.weekday() < 5:
            pool.append(current)
        current += timedelta(days=1)
    if not pool:
        return []
    return sorted(rng.sample(pool, min(k, len(pool))))


def _build_medical_record_encounters(
    seed_start: date,
    patient_rows: list[tuple],
    appointment_rows: list[tuple],
    years: int = 10,
    records_per_year: int = 3,
) -> dict[int, list[dict]]:
    rng = random.Random(seed_start.toordinal() + 808)
    patient_doctor_counts = _build_patient_doctor_counts(appointment_rows)
    all_doctor_ids = [doctor_id for doctor_id, _, _, _ in SEED_DOCTORS]
    encounters_by_patient: dict[int, list[dict]] = {}

    for patient_id, *_ in patient_rows:
        doctor_counts = patient_doctor_counts.get(patient_id, {})
        allowed_doctors = sorted(doctor_counts.keys()) or [rng.choice(all_doctor_ids)]
        primary_doctor_id = max(allowed_doctors, key=lambda d_id: doctor_counts.get(d_id, 0))
        encounters: list[dict] = []

        for year_index in range(years):
            encounter_year = seed_start.year - (years - 1 - year_index)
            business_days = _random_business_days_in_year(encounter_year, seed_start, records_per_year, rng)

            for day in business_days:
                hour = rng.choice([9, 10, 11, 13, 14, 15, 16])
                minute = rng.choice([0, 15, 30, 45])
                encounters.append(
                    {
                        "created_at": datetime(day.year, day.month, day.day, hour, minute).strftime("%Y-%m-%d %H:%M"),
                        "doctor_id": primary_doctor_id,
                        "visit_type": rng.choice(VISIT_TYPES),
                    }
                )

        encounters.sort(key=lambda row: row["created_at"])
        encounters_by_patient[patient_id] = encounters

    return encounters_by_patient


# ── Note-synthesis constants ─────────────────────────────────────────────────

_ROUTINE_VISIT_LABELS = [
    "Chronic disease management — follow-up",
    "Routine outpatient visit",
    "Scheduled chronic care follow-up",
    "Ongoing disease management",
    "Preventive care follow-up",
]

_ANNUAL_VISIT_LABELS = [
    "Annual wellness exam",
    "Yearly health maintenance visit",
    "Annual physical — established patient",
    "Comprehensive annual review",
]

_ACUTE_SYMPTOMS = [
    "mild ankle edema without pain",
    "increased fatigue over the past 2 weeks",
    "occasional palpitations; no syncope or chest pain",
    "persistent dry cough — possible ACE inhibitor side effect",
    "intermittent dizziness on standing",
    "mild lower back pain; no radicular symptoms",
    "weight gain of 4–6 lbs since last visit",
    "seasonal nasal congestion and mild dyspnea",
    "insomnia and mild anxiety",
    "increased urinary frequency without dysuria",
    "mild nausea; no vomiting",
    "knee discomfort limiting daily activity",
    "frontal headaches, relieved by ibuprofen",
    "difficulty sleeping due to nocturia",
    "mild peripheral tingling in both hands",
]

_LAB_FINDINGS = [
    "Lipid panel: LDL mildly elevated at 118 mg/dL",
    "HbA1c 6.9% — slightly above target",
    "Creatinine 1.1 mg/dL; stable",
    "BMP within normal limits",
    "CBC: mild anemia, Hgb 11.8 g/dL",
    "TSH 3.2 — euthyroid",
    "LFTs normal; patient continues statin therapy",
    "Fasting glucose 108 mg/dL — pre-diabetic range",
    "Potassium 3.6 mEq/L; monitoring given diuretic use",
    "Lipid panel improved: LDL 94 mg/dL",
    "Urine microalbumin borderline; renal function otherwise normal",
    "eGFR 72 mL/min — CKD stage 2, monitoring",
    "A1c improved to 6.4% since last draw",
    "Iron studies: ferritin 18 ng/mL — low-normal",
    "Vitamin D 21 ng/mL — supplementation initiated",
]

_EXAM_FINDINGS = [
    "Alert, oriented, and in no acute distress",
    "Well-appearing; no acute distress",
    "Cooperative; appears comfortable and appropriately dressed",
    "Alert and oriented x3; mildly fatigued appearance",
    "Pleasant and conversational; no distress noted",
]

_PLAN_ACTIONS = [
    "Continue current medications. Return in 3 months.",
    "Dose adjusted. Recheck labs in 4 weeks.",
    "Lifestyle counseling provided. Follow-up in 6 months.",
    "Expedited labs ordered. Results to be reviewed at next visit.",
    "CBC, CMP, and lipid panel ordered. Review at next visit.",
    "Prescription refilled. Patient reminded of warning signs.",
    "New medication initiated. Monitor for adverse effects over next 2 weeks.",
    "Annual preventive labs ordered. Return in 12 months unless symptoms arise.",
    "Patient educated on condition management. Written resources provided.",
    "Symptoms improving. Continue current regimen. RTC in 2 months.",
    "Medication trial discontinued; alternative agent started.",
    "Home BP monitoring recommended. Log readings for next visit.",
    "Referral placed to specialist for further evaluation.",
    "Discussed risk-factor modification. Weight loss goal set.",
    "Sleep hygiene counseling provided. Reassess at next visit.",
]

_STATUS_WORDS = [
    "stable",
    "adequately controlled",
    "suboptimally controlled",
    "improving",
    "requires monitoring",
]

_NOTE_STYLES = ["soap", "bullets", "narrative_lines", "structured"]

# Approved encounter-type reference data (category, name)
ENCOUNTER_TYPES: list[tuple[str, str]] = [
    # core
    ("core", "OfficeVisit"),
    ("core", "SpecialistVisit"),
    ("core", "TelehealthVisit"),
    ("core", "EDVisit"),
    ("core", "UrgentCareVisit"),
    ("core", "HospitalAdmit"),
    ("core", "HospitalDischarge"),
    ("core", "Surgery"),
    ("core", "FollowUpVisit"),
    ("core", "CareTransition"),
    # diagnostic
    ("diagnostic", "LabOrder"),
    ("diagnostic", "LabResult"),
    ("diagnostic", "ImagingStudy"),
    ("diagnostic", "ImagingReport"),
    ("diagnostic", "PathReport"),
    ("diagnostic", "Vitals"),
    ("diagnostic", "PhysExam"),
    ("diagnostic", "DiagAssess"),
    ("diagnostic", "Screening"),
    ("diagnostic", "Biopsy"),
    # treatment
    ("treatment", "MedAdmin"),
    ("treatment", "Infusion"),
    ("treatment", "Prescription"),
    ("treatment", "Vaccination"),
    ("treatment", "WoundCare"),
    ("treatment", "PTSession"),
    ("treatment", "OTSession"),
    ("treatment", "STSession"),
    ("treatment", "Dialysis"),
    ("treatment", "RadiationTx"),
    ("treatment", "Procedure"),
    ("treatment", "TreatPlan"),
]


def _make_vitals(profile: dict, rng: random.Random, encounter_index: int, total: int) -> dict:
    """Generate visit vitals with realistic drift over the encounter timeline."""
    baseline_bp = (profile.get("baseline_vitals") or {}).get("bp", "")
    base_sys, base_dia = 128, 82
    if baseline_bp:
        parts = baseline_bp.replace(" ", "").split("/")
        if len(parts) == 2:
            try:
                base_sys, base_dia = int(parts[0]), int(parts[1])
            except ValueError:
                pass
    # Slight upward drift over 10 years (disease progression)
    late_factor = encounter_index / max(total - 1, 1)
    bp_drift = int(late_factor * 6)
    systolic = rng.randint(base_sys - 8, base_sys + 10) + bp_drift
    diastolic = rng.randint(base_dia - 6, base_dia + 8)
    return {
        "bp": f"{systolic}/{diastolic}",
        "hr": rng.randint(62, 90),
        "rr": rng.randint(14, 20),
        "temp": round(rng.uniform(97.4, 99.2), 1),
        "spo2": rng.randint(96, 99),
    }


def _pick_problems(profile: dict, rng: random.Random, n: int = 2) -> list[str]:
    pool = list(profile.get("problem_list") or ["hypertension", "hyperlipidemia"])
    return rng.sample(pool, min(n, len(pool)))


def _note_soap(
    *,
    patient_name: str,
    doctor_name: str,
    specialty: str,
    visit_label: str,
    vitals: dict,
    active_problems: list[str],
    acute_symptom: str,
    lab_finding: str,
    exam_finding: str,
    plan: str,
    medication: str,
    status: str,
) -> str:
    subjective = acute_symptom.capitalize() if acute_symptom else "No new complaints. Routine follow-up."
    lines = [
        f"PATIENT: {patient_name}  |  PROVIDER: Dr. {doctor_name} ({specialty})",
        f"VISIT: {visit_label}",
        "",
        f"S: {subjective}",
        f"   Problems: {', '.join(active_problems)}.",
        "",
        "O:",
        f"   BP {vitals['bp']} mmHg  |  HR {vitals['hr']} bpm  |  RR {vitals['rr']}/min",
        f"   Temp {vitals['temp']}°F  |  SpO2 {vitals['spo2']}%",
        f"   General: {exam_finding}.",
        "",
        f"A: {', '.join(active_problems)} — {status}.",
    ]
    if lab_finding:
        lines.append(f"   Labs: {lab_finding}.")
    lines += [
        "",
        f"P: {plan}",
        f"   Medication: {medication} — continued/refilled.",
    ]
    return "\n".join(lines)


def _note_bullets(
    *,
    patient_name: str,
    doctor_name: str,
    specialty: str,
    visit_label: str,
    vitals: dict,
    active_problems: list[str],
    acute_symptom: str,
    lab_finding: str,
    exam_finding: str,
    plan: str,
    medication: str,
    status: str,
) -> str:
    cc = acute_symptom.capitalize() if acute_symptom else "Routine chronic care; no new complaints"
    lines = [
        f"Visit: {visit_label}",
        f"Patient: {patient_name}   Provider: Dr. {doctor_name}, {specialty}",
        "",
        f"• Chief Complaint: {cc}",
        f"• Active Problems: {'; '.join(active_problems)} ({status})",
        f"• Vitals: BP {vitals['bp']} | HR {vitals['hr']} | RR {vitals['rr']} | Temp {vitals['temp']}°F | SpO2 {vitals['spo2']}%",
        f"• Exam: {exam_finding}.",
        f"• Current Medication: {medication}.",
    ]
    if lab_finding:
        lines.append(f"• Recent Labs: {lab_finding}.")
    lines.append(f"• Plan: {plan}")
    return "\n".join(lines)


def _note_narrative_lines(
    *,
    patient_name: str,
    doctor_name: str,
    specialty: str,
    visit_label: str,
    vitals: dict,
    active_problems: list[str],
    acute_symptom: str,
    lab_finding: str,
    exam_finding: str,
    plan: str,
    medication: str,
    status: str,
) -> str:
    lines = [
        f"{patient_name} seen by Dr. {doctor_name} ({specialty}) — {visit_label.lower()}.",
    ]
    if acute_symptom:
        lines.append(f"Reports {acute_symptom}.")
    else:
        lines.append("No acute complaints since last visit.")
    lines += [
        f"BP {vitals['bp']}, HR {vitals['hr']}, RR {vitals['rr']}, Temp {vitals['temp']}°F, SpO2 {vitals['spo2']}%.",
        f"{exam_finding}.",
        f"Active concerns: {', '.join(active_problems)} — {status}.",
    ]
    if lab_finding:
        lines.append(f"Lab note: {lab_finding}.")
    lines += [
        f"Medications: {medication}.",
        f"{plan}",
    ]
    return "\n".join(lines)


def _note_structured(
    *,
    patient_name: str,
    doctor_name: str,
    specialty: str,
    visit_label: str,
    vitals: dict,
    active_problems: list[str],
    acute_symptom: str,
    lab_finding: str,
    exam_finding: str,
    plan: str,
    medication: str,
    status: str,
) -> str:
    lines = [
        f"ENCOUNTER — {visit_label.upper()}",
        f"Patient: {patient_name}   Provider: {doctor_name}, {specialty}",
        "",
        "CHIEF COMPLAINT",
        f"  {acute_symptom.capitalize() if acute_symptom else 'Routine management visit; no new symptoms.'}",
        "",
        "VITALS",
        f"  BP {vitals['bp']} mmHg   HR {vitals['hr']} bpm   RR {vitals['rr']}/min   Temp {vitals['temp']}°F   SpO2 {vitals['spo2']}%",
        "",
        f"ACTIVE PROBLEMS  ({status.upper()})",
    ]
    for p in active_problems:
        lines.append(f"  – {p}")
    lines += [
        "",
        f"EXAM: {exam_finding}.",
    ]
    if lab_finding:
        lines += [
            "",
            f"LABS / RESULTS: {lab_finding}",
        ]
    lines += [
        "",
        f"MEDICATIONS: {medication} — continued.",
        "",
        f"PLAN: {plan}",
    ]
    return "\n".join(lines)


def _llm_generate_patient_profile(
    *,
    client,
    model: str,
    patient_id: int,
    patient_name: str,
    age: int,
    allowed_doctors: list[dict],
    primary_doctor_id: int,
    encounter_count: int,
) -> dict:
    system_prompt = (
        "You are a clinical documentation assistant. "
        "Generate a compact, realistic longitudinal patient profile for outpatient chronic-care note synthesis. "
        "Make each patient medically distinct — vary the conditions, medications, symptom patterns, and vitals baselines. "
        "Select conditions, medications, and vitals appropriate for the patient's age. "
        "Output compact JSON only."
    )

    user_payload = {
        "task": "Generate a unique patient profile covering 10 years of outpatient follow-up.",
        "patient": {"patient_id": patient_id, "patient_name": patient_name, "age_years": age},
        "constraints": {
            "allowed_doctors": allowed_doctors,
            "primary_doctor_id": primary_doctor_id,
            "encounter_count": encounter_count,
        },
        "output_json_schema": {
            "profile": {
                "problem_list": ["string — 2 to 4 distinct chronic conditions"],
                "baseline_vitals": {"bp": "NNN/NN", "hr": "NN", "rr": "NN", "temp": "NN.N", "spo2": "NN"},
                "trend": "string — one-sentence health trajectory over 10 years",
                "symptom_focus": ["string — 4 to 6 concise patient-reported symptoms specific to this patient"],
                "acute_concerns": ["string — 3 to 5 episodic complaints this patient might raise at visits"],
                "common_assessments": ["string"],
                "recommendation_style": ["string"],
                "medications": ["string — 3 to 5 medications plausible for this patient's conditions"],
                "doctor_relationship": "string",
            }
        },
    }

    response = client.chat.completions.create(
        model=model,
        temperature=0.85,
        response_format={"type": "json_object"},
        max_tokens=800,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": json.dumps(user_payload)},
        ],
    )

    raw_content = response.choices[0].message.content or "{}"
    parsed = json.loads(raw_content)
    profile = parsed.get("profile")
    if not isinstance(profile, dict):
        raise ValueError("LLM response missing profile object")

    return profile


def _choose_encounter_type(
    is_annual: bool,
    is_acute: bool,
    has_lab: bool,
    rng: random.Random,
) -> str:
    """Pick an encounter type semantically consistent with the note context.

    Weighted heavily toward the three most common types:
    OfficeVisit, FollowUpVisit, and LabResult.
    """
    if is_annual:
        return rng.choices(
            ["OfficeVisit", "PhysExam", "Screening", "LabOrder", "Vitals", "LabResult"],
            weights=[40, 20, 15, 10, 10, 5],
        )[0]
    if is_acute:
        return rng.choices(
            ["OfficeVisit", "UrgentCareVisit", "EDVisit", "FollowUpVisit", "TelehealthVisit", "Procedure"],
            weights=[30, 20, 10, 20, 10, 10],
        )[0]
    if has_lab:
        return rng.choices(
            ["LabResult", "OfficeVisit", "FollowUpVisit", "LabOrder", "DiagAssess"],
            weights=[35, 25, 25, 10, 5],
        )[0]
    # Routine visit, no lab finding
    return rng.choices(
        ["FollowUpVisit", "OfficeVisit", "TelehealthVisit", "SpecialistVisit", "Vitals"],
        weights=[40, 35, 10, 10, 5],
    )[0]


def _synthesize_medical_note(
    *,
    patient_name: str,
    doctor_name: str,
    specialty: str,
    encounter: dict,
    profile: dict,
    rng: random.Random,
    encounter_index: int,
    total_encounters: int,
) -> tuple[str, str]:
    """Generate one visit note in one of four formats with varied content.

    Returns a (note_text, encounter_type) tuple.
    """
    problem_list = list(profile.get("problem_list") or ["hypertension", "hyperlipidemia"])
    medications = list(profile.get("medications") or ["lisinopril", "atorvastatin"])
    # Blend patient-specific symptoms with the generic pool for variety
    patient_symptoms: list[str] = list(profile.get("symptom_focus") or [])
    patient_acute: list[str] = list(profile.get("acute_concerns") or [])
    symptom_pool = patient_symptoms + patient_acute + _ACUTE_SYMPTOMS

    active_problems = _pick_problems(profile, rng, n=rng.randint(1, min(2, len(problem_list))))
    medication = rng.choice(medications)
    exam_finding = rng.choice(_EXAM_FINDINGS)
    status = rng.choice(_STATUS_WORDS)

    # Determine visit type: annual every ~10 encounters, acute ~25% of remaining
    is_annual = (encounter_index % 10 == 9)
    is_acute = (not is_annual) and (rng.random() < 0.25)

    if is_annual:
        visit_label = rng.choice(_ANNUAL_VISIT_LABELS)
        acute_symptom = ""
    elif is_acute:
        visit_label = "Acute / Unscheduled Visit"
        acute_symptom = rng.choice(symptom_pool)
    else:
        visit_label = rng.choice(_ROUTINE_VISIT_LABELS)
        # Routine visits sometimes include a complaint (40% chance)
        acute_symptom = rng.choice(symptom_pool) if rng.random() < 0.4 else ""

    lab_finding = rng.choice(_LAB_FINDINGS) if rng.random() < 0.5 else ""
    plan = rng.choice(_PLAN_ACTIONS)
    vitals = _make_vitals(profile, rng, encounter_index, total_encounters)

    # Cycle through all 4 styles, with occasional shuffle so runs of same style are rare
    style = _NOTE_STYLES[encounter_index % len(_NOTE_STYLES)]
    if rng.random() < 0.2:
        style = rng.choice(_NOTE_STYLES)

    kwargs: dict = dict(
        patient_name=patient_name,
        doctor_name=doctor_name,
        specialty=specialty,
        visit_label=visit_label,
        vitals=vitals,
        active_problems=active_problems,
        acute_symptom=acute_symptom,
        lab_finding=lab_finding,
        exam_finding=exam_finding,
        plan=plan,
        medication=medication,
        status=status,
    )

    if style == "soap":
        note = _note_soap(**kwargs)
    elif style == "bullets":
        note = _note_bullets(**kwargs)
    elif style == "narrative_lines":
        note = _note_narrative_lines(**kwargs)
    else:
        note = _note_structured(**kwargs)

    encounter_type = _choose_encounter_type(
        is_annual=is_annual,
        is_acute=is_acute,
        has_lab=bool(lab_finding),
        rng=rng,
    )
    return note, encounter_type


def _generate_seed_medical_records(
    seed_start: date,
    patient_rows: list[tuple],
    appointment_rows: list[tuple],
    progress_cb: Callable[[str], None] | None,
) -> list[tuple[str, int, str, str, int, str]]:
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError(
            "OPENAI_API_KEY is required for LLM-based medical_records seeding. "
            "Set OPENAI_API_KEY and run database reseed again."
        )

    try:
        from openai import OpenAI
    except ImportError as exc:
        raise RuntimeError(
            "The openai package is required for LLM-based medical_records seeding. "
            "Install dependencies and reseed."
        ) from exc

    model = os.getenv("MEDICAL_RECORDS_SEED_MODEL", "gpt-4o-mini")
    client = OpenAI(api_key=api_key, timeout=45.0, max_retries=1)

    doctor_lookup = {
        doctor_id: {
            "doctor_id": doctor_id,
            "doctor_name": f"{last_name}, {first_name}",
            "specialty": specialty,
        }
        for doctor_id, first_name, last_name, specialty in SEED_DOCTORS
    }

    encounters_by_patient = _build_medical_record_encounters(
        seed_start=seed_start,
        patient_rows=patient_rows,
        appointment_rows=appointment_rows,
        years=10,
        records_per_year=3,
    )
    patient_doctor_counts = _build_patient_doctor_counts(appointment_rows)

    rows: list[tuple[str, int, str, str]] = []
    inserted = 0
    total_patients = len(patient_rows)
    total_expected_records = sum(len(encounters) for encounters in encounters_by_patient.values())

    _emit(
        progress_cb,
        f"Step: preparing LLM medical records for {total_patients} patients (~{total_expected_records} notes total)",
    )

    for patient_idx, (patient_id, first_name, last_name, _, _, _, birthdate) in enumerate(patient_rows, start=1):
        encounters = encounters_by_patient[patient_id]
        doctor_counts = patient_doctor_counts.get(patient_id, {})
        allowed_ids = sorted({enc["doctor_id"] for enc in encounters})
        primary_doctor_id = max(allowed_ids, key=lambda d_id: doctor_counts.get(d_id, 0))
        allowed_doctors = [doctor_lookup[d_id] for d_id in allowed_ids]

        patient_name = f"{first_name} {last_name}"

        _emit(
            progress_cb,
            (
                f"Step: generating medical history for patient {patient_idx}/{total_patients} "
                f"({patient_name}) using Dr. {doctor_lookup[primary_doctor_id]['doctor_name']} "
                f"- {len(encounters)} note(s)"
            ),
        )

        profile: dict | None = None
        last_error: Exception | None = None

        for attempt in range(1, 4):
            try:
                _emit(progress_cb, f"Sub-step: sending LLM profile request for patient {patient_name} [attempt {attempt}/3]")
                age = seed_start.year - date.fromisoformat(birthdate).year
                profile = _llm_generate_patient_profile(
                    client=client,
                    model=model,
                    patient_id=patient_id,
                    patient_name=patient_name,
                    age=age,
                    allowed_doctors=allowed_doctors,
                    primary_doctor_id=primary_doctor_id,
                    encounter_count=len(encounters),
                )
                break
            except Exception as exc:  # noqa: BLE001
                last_error = exc
                _emit(progress_cb, f"Sub-step: LLM profile attempt {attempt}/3 failed for patient {patient_name}; retrying")

        if profile is None:
            raise RuntimeError(
                f"LLM profile generation failed for patient {patient_id} after retries"
            ) from last_error

        notes: list[tuple[str, str]] = []
        rng = random.Random(seed_start.toordinal() + patient_id)

        _emit(progress_cb, f"Sub-step: synthesizing {len(encounters)} note(s) locally for patient {patient_name}")
        for encounter_idx, encounter in enumerate(encounters):
            enc_doctor = doctor_lookup.get(encounter["doctor_id"], doctor_lookup[primary_doctor_id])
            note, enc_type = _synthesize_medical_note(
                patient_name=patient_name,
                doctor_name=enc_doctor["doctor_name"],
                specialty=enc_doctor["specialty"],
                encounter=encounter,
                profile=profile,
                rng=rng,
                encounter_index=encounter_idx,
                total_encounters=len(encounters),
            )
            notes.append((note, enc_type))

            if (encounter_idx + 1) % 6 == 0 or encounter_idx + 1 == len(encounters):
                _emit(
                    progress_cb,
                    (
                        f"Sub-step: completed patient {patient_idx}/{total_patients} ({patient_name}); "
                        f"{encounter_idx + 1}/{len(encounters)} notes prepared"
                    ),
                )

        for encounter_idx, encounter in enumerate(encounters):
            note_text, enc_type = notes[encounter_idx]
            rows.append(
                (
                    f"record-seed-{patient_id}-{encounter_idx + 1:02d}",
                    patient_id,
                    note_text,
                    enc_type,
                    encounter["doctor_id"],
                    encounter["created_at"],
                )
            )
            inserted += 1

        _emit(
            progress_cb,
            (
                f"Sub-step: completed patient {patient_idx}/{total_patients} "
                f"({patient_name}); {inserted}/{total_expected_records} medical notes prepared"
            ),
        )

    return rows


def seed_mock_data(
    *,
    reset: bool = False,
    seed_start: date | None = None,
    progress_cb: Callable[[str], None] | None = None,
) -> None:
    init_db()
    if not reset and is_db_seeded():
        _emit(progress_cb, "Step: seed check complete (already seeded, skipping)")
        return

    start_day = _next_business_day_on_or_after(seed_start or date.today())
    _emit(progress_cb, "Step: initializing database schema")

    with connection() as conn:
        if _has_legacy_string_ids(conn):
            _emit(progress_cb, "Step: detected legacy string IDs - forcing reset to integer IDs")
            reset = True

        if reset:
            _emit(progress_cb, "Step: reset requested - clearing existing data")
            conn.execute("DELETE FROM appointments")
            _emit(progress_cb, "Sub-step: cleared appointments")
            conn.execute("DELETE FROM medical_records")
            _emit(progress_cb, "Sub-step: cleared medical_records")
            clear_patient_history_vectors()
            _emit(progress_cb, "Sub-step: cleared medical_record vectors")
            conn.execute("DELETE FROM doctors")
            _emit(progress_cb, "Sub-step: cleared doctors")
            conn.execute("DELETE FROM patients")
            _emit(progress_cb, "Sub-step: cleared patients")
            _clear_seed_metadata(conn)
            _emit(progress_cb, "Sub-step: cleared seed metadata")

        _emit(progress_cb, "Step: seeding encounter_types reference table")
        conn.executemany(
            "INSERT OR IGNORE INTO encounter_types(category, name) VALUES (?, ?)",
            ENCOUNTER_TYPES,
        )
        _emit(progress_cb, f"Sub-step: upserted {len(ENCOUNTER_TYPES)} encounter_type row(s)")

        _emit(progress_cb, "Step: seeding patients")
        patient_rows = _build_patient_seed_rows(start_day)
        conn.executemany(
            """
            INSERT OR IGNORE INTO patients(
                patient_id, first_name, last_name, telephone, address, insurance_carrier, birthdate
            )
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            patient_rows,
        )
        _emit(progress_cb, f"Sub-step: upserted {len(SEED_PATIENTS)} patient row(s)")

        _emit(progress_cb, "Step: seeding doctors")
        conn.executemany(
            "INSERT OR IGNORE INTO doctors(doctor_id, first_name, last_name, specialty) VALUES (?, ?, ?, ?)",
            SEED_DOCTORS,
        )
        _emit(progress_cb, f"Sub-step: upserted {len(SEED_DOCTORS)} doctor row(s)")

        _emit(progress_cb, "Step: generating and seeding appointments")
        appointment_rows = _generate_seed_appointments(start_day, progress_cb)
        conn.executemany(
            """
            INSERT OR IGNORE INTO appointments(
                appointment_id, patient_id, doctor_id, start_at, end_at,
                duration_min, visit_type, status, notes
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            appointment_rows,
        )
        _emit(progress_cb, f"Sub-step: inserted {len(appointment_rows)} appointment row(s)")

        _emit(progress_cb, "Step: seeding medical records with LLM")
        medical_record_rows = _generate_seed_medical_records(
            seed_start=start_day,
            patient_rows=patient_rows,
            appointment_rows=appointment_rows,
            progress_cb=progress_cb,
        )
        conn.executemany(
            "INSERT OR IGNORE INTO medical_records(record_id, patient_id, note, encounter_type, doctor_id, created_at) VALUES (?, ?, ?, ?, ?, ?)",
            medical_record_rows,
        )
        _emit(progress_cb, f"Sub-step: upserted {len(medical_record_rows)} medical record row(s)")

        _emit(progress_cb, "Step: indexing medical records in vector store")
        index_medical_record_rows(medical_record_rows, progress_cb=progress_cb)

    _set_seed_metadata(start_day)
    _emit(progress_cb, f"Step: seeding complete (start business day: {start_day.isoformat()})")


def fetch_appointments_for_patient(patient_id: int | str) -> list[dict]:
    with connection() as conn:
        rows = conn.execute(
            """
            SELECT
                a.appointment_id,
                a.patient_id,
                p.last_name || ', ' || p.first_name AS patient_name,
                a.doctor_id,
                d.last_name || ', ' || d.first_name AS doctor_name,
                d.specialty,
                a.start_at,
                a.end_at,
                a.duration_min,
                a.visit_type,
                a.status,
                a.notes
            FROM appointments a
            JOIN patients p ON p.patient_id = a.patient_id
            JOIN doctors d ON d.doctor_id = a.doctor_id
            WHERE a.patient_id = ?
            ORDER BY a.start_at
            """,
            (patient_id,),
        ).fetchall()
    return [dict(r) for r in rows]


def fetch_appointments_for_doctor(doctor_id: int | str) -> list[dict]:
    with connection() as conn:
        rows = conn.execute(
            """
            SELECT
                a.appointment_id,
                a.patient_id,
                p.last_name || ', ' || p.first_name AS patient_name,
                a.doctor_id,
                d.last_name || ', ' || d.first_name AS doctor_name,
                d.specialty,
                a.start_at,
                a.end_at,
                a.duration_min,
                a.visit_type,
                a.status,
                a.notes
            FROM appointments a
            JOIN patients p ON p.patient_id = a.patient_id
            JOIN doctors d ON d.doctor_id = a.doctor_id
            WHERE a.doctor_id = ?
            ORDER BY a.start_at
            """,
            (doctor_id,),
        ).fetchall()
    return [dict(r) for r in rows]


def fetch_appointments_for_day(target_day: str | date) -> list[dict]:
    if isinstance(target_day, date):
        day_value = target_day.strftime("%Y-%m-%d")
    else:
        day_value = target_day
    with connection() as conn:
        rows = conn.execute(
            """
            SELECT
                a.appointment_id,
                a.patient_id,
                p.last_name || ', ' || p.first_name AS patient_name,
                a.doctor_id,
                d.last_name || ', ' || d.first_name AS doctor_name,
                d.specialty,
                a.start_at,
                a.end_at,
                a.duration_min,
                a.visit_type,
                a.status,
                a.notes
            FROM appointments a
            JOIN patients p ON p.patient_id = a.patient_id
            JOIN doctors d ON d.doctor_id = a.doctor_id
            WHERE date(a.start_at) = date(?)
            ORDER BY a.doctor_id, a.start_at
            """,
            (day_value,),
        ).fetchall()
    return [dict(r) for r in rows]


def fetch_medical_records(patient_id: str) -> list[dict]:
    with connection() as conn:
        rows = conn.execute(
            """
            SELECT
                mr.record_id,
                mr.patient_id,
                mr.note,
                mr.encounter_type,
                mr.doctor_id,
                d.last_name || ', ' || d.first_name AS doctor_name,
                mr.created_at
            FROM medical_records mr
            LEFT JOIN doctors d ON mr.doctor_id = d.doctor_id
            WHERE mr.patient_id = ?
            ORDER BY mr.created_at DESC
            """,
            (patient_id,),
        ).fetchall()
    return [dict(r) for r in rows]


def add_medical_record(
    patient_id: int | str,
    note: str,
    doctor_id: int | str | None = None,
    encounter_type: str = "OfficeVisit",
) -> str:
    """Insert a new medical record and return its record_id."""
    import uuid as _uuid
    from datetime import datetime as _dt

    record_id = f"record-{_uuid.uuid4().hex[:12]}"
    created_at = _dt.now().strftime("%Y-%m-%d %H:%M")
    with connection() as conn:
        conn.execute(
            """
            INSERT INTO medical_records(record_id, patient_id, note, encounter_type, doctor_id, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                record_id,
                int(patient_id),
                note,
                encounter_type,
                int(doctor_id) if doctor_id is not None else None,
                created_at,
            ),
        )
    return record_id


def update_medical_record(record_id: str, note: str, encounter_type: str) -> None:
    """Update the note and encounter type of an existing medical record."""
    with connection() as conn:
        conn.execute(
            "UPDATE medical_records SET note = ?, encounter_type = ? WHERE record_id = ?",
            (note, encounter_type, record_id),
        )


def delete_medical_record(record_id: str) -> None:
    """Permanently delete a medical record by its record_id."""
    with connection() as conn:
        conn.execute("DELETE FROM medical_records WHERE record_id = ?", (record_id,))


def fetch_all_appointments() -> list[dict]:
    with connection() as conn:
        rows = conn.execute(
            """
            SELECT
                a.appointment_id,
                a.patient_id,
                p.last_name || ', ' || p.first_name AS patient_name,
                a.doctor_id,
                d.last_name || ', ' || d.first_name AS doctor_name,
                d.specialty,
                a.start_at,
                a.end_at,
                a.duration_min,
                a.visit_type,
                a.status,
                a.notes
            FROM appointments a
            JOIN patients p ON p.patient_id = a.patient_id
            JOIN doctors d ON d.doctor_id = a.doctor_id
            ORDER BY a.start_at
            """
        ).fetchall()
    return [dict(r) for r in rows]


_CLINIC_SLOTS = [
    "08:00", "08:30", "09:00", "09:30",
    "10:00", "10:30", "11:00", "11:30",
    "13:00", "13:30", "14:00", "14:30",
    "15:00", "15:30", "16:00", "16:30",
]


def fetch_available_slots(doctor_id: int, date_str: str) -> list[str]:
    """Return clinic time slots (HH:MM, 24-hour) that are not yet booked for *doctor_id* on *date_str*."""
    with connection() as conn:
        rows = conn.execute(
            "SELECT start_at FROM appointments WHERE doctor_id = ? AND start_at LIKE ?",
            (doctor_id, f"{date_str}%"),
        ).fetchall()
    booked = {r["start_at"][11:16] for r in rows}   # extract HH:MM from "YYYY-MM-DD HH:MM"
    return [s for s in _CLINIC_SLOTS if s not in booked]


def fetch_counts() -> dict:
    with connection() as conn:
        patient_count = conn.execute("SELECT COUNT(*) AS c FROM patients").fetchone()["c"]
        doctor_count = conn.execute("SELECT COUNT(*) AS c FROM doctors").fetchone()["c"]
        appointment_count = conn.execute("SELECT COUNT(*) AS c FROM appointments").fetchone()["c"]
    return {
        "patients": patient_count,
        "doctors": doctor_count,
        "appointments": appointment_count,
    }


def fetch_patient_roster() -> list[str]:
    with connection() as conn:
        rows = conn.execute(
            """
            SELECT last_name || ', ' || first_name AS patient_name
            FROM patients
            ORDER BY last_name, first_name
            """
        ).fetchall()
    return [row["patient_name"] for row in rows]


def fetch_patient_list() -> list[dict]:
    with connection() as conn:
        rows = conn.execute(
            """
            SELECT
                patient_id,
                last_name || ', ' || first_name AS patient_name,
                birthdate,
                telephone,
                address,
                insurance_carrier
            FROM patients
            ORDER BY last_name, first_name
            """
        ).fetchall()
    return [dict(row) for row in rows]


def fetch_doctor_list() -> list[dict]:
    with connection() as conn:
        rows = conn.execute(
            """
            SELECT
                doctor_id,
                last_name || ', ' || first_name AS doctor_name,
                specialty
            FROM doctors
            ORDER BY last_name, first_name
            """
        ).fetchall()
    return [dict(row) for row in rows]


def create_appointment(
    patient_id: int,
    doctor_id: int,
    start_at: str,
    duration_min: int = 60,
    visit_type: str = "follow_up",
    notes: str = "",
) -> str:
    """Insert a new appointment and return the generated appointment_id."""
    start_dt = datetime.strptime(start_at, "%Y-%m-%d %H:%M")
    end_dt = start_dt + timedelta(minutes=duration_min)
    appointment_id = f"appointment-{uuid.uuid4().hex[:12]}"
    with connection() as conn:
        conn.execute(
            """
            INSERT INTO appointments(
                appointment_id, patient_id, doctor_id, start_at, end_at,
                duration_min, visit_type, status, notes
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                appointment_id,
                patient_id,
                doctor_id,
                start_at,
                end_dt.strftime("%Y-%m-%d %H:%M"),
                duration_min,
                visit_type,
                "booked",
                notes or "",
            ),
        )
    return appointment_id


def cancel_appointment(appointment_id: str, patient_id: int | str) -> bool:
    """Set an appointment's status to 'cancelled'.

    Only cancels if the appointment belongs to *patient_id* (prevents cross-patient writes).
    Returns True if the row was updated, False if not found or not owned by this patient.
    """
    with connection() as conn:
        cur = conn.execute(
            """
            UPDATE appointments
            SET status = 'cancelled'
            WHERE appointment_id = ?
              AND patient_id = ?
              AND status != 'cancelled'
            """,
            (appointment_id, patient_id),
        )
        return cur.rowcount > 0


def search_patients_by_record_keyword(keyword: str) -> list[dict]:
    """Return patients whose medical record notes contain *keyword* as an exact substring.

    Uses SQL LIKE for precise matching — suitable for drug names, diagnoses, and
    any term that must literally appear in the note text.
    Each returned dict has: patient_id, patient_name, match_count, latest_match.
    """
    safe_keyword = keyword.strip()
    if not safe_keyword:
        return []

    with connection() as conn:
        rows = conn.execute(
            """
            SELECT
                p.patient_id,
                p.last_name || ', ' || p.first_name AS patient_name,
                COUNT(mr.record_id) AS match_count,
                MAX(mr.created_at) AS latest_match
            FROM medical_records mr
            JOIN patients p ON p.patient_id = mr.patient_id
            WHERE mr.note LIKE ? ESCAPE '\\'
            GROUP BY p.patient_id, patient_name
            ORDER BY match_count DESC, patient_name
            """,
            (f"%{safe_keyword}%",),
        ).fetchall()
    return [dict(row) for row in rows]


def fetch_doctor_day_patients_with_keyword(
    doctor_name: str,
    date_str: str,
    keyword: str,
) -> list[dict]:
    """Return patients who had an appointment with *doctor_name* on *date_str* AND
    whose medical records contain *keyword* as an exact substring.

    Performs the join entirely in SQL — no LLM cross-referencing needed.
    Each returned dict has: patient_id, patient_name, appointment_count, note_match_count.
    """
    safe_keyword = keyword.strip()
    name_pattern = f"%{doctor_name.strip()}%"

    with connection() as conn:
        rows = conn.execute(
            """
            SELECT
                p.patient_id,
                p.last_name || ', ' || p.first_name AS patient_name,
                COUNT(DISTINCT a.appointment_id)     AS appointment_count,
                COUNT(DISTINCT mr.record_id)          AS note_match_count
            FROM appointments a
            JOIN patients p  ON p.patient_id  = a.patient_id
            JOIN doctors  d  ON d.doctor_id   = a.doctor_id
            JOIN medical_records mr ON mr.patient_id = a.patient_id
            WHERE DATE(a.start_at) = ?
              AND (
                    d.last_name  LIKE ? ESCAPE '\\'
                 OR d.first_name LIKE ? ESCAPE '\\'
                 OR (d.last_name || ', ' || d.first_name) LIKE ? ESCAPE '\\'
              )
              AND mr.note LIKE ? ESCAPE '\\'
            GROUP BY p.patient_id, patient_name
            ORDER BY patient_name
            """,
            (date_str, name_pattern, name_pattern, name_pattern, f"%{safe_keyword}%"),
        ).fetchall()
    return [dict(row) for row in rows]


def fetch_doctor_patients_with_keyword(doctor_name: str, keyword: str) -> list[dict]:
    """Return all distinct patients who have EVER had an appointment with *doctor_name*
    AND whose medical records contain *keyword* as an exact substring.

    Use this for general questions like 'How many of Dr. X's patients take Metformin?'
    (no date constraint).  Each returned dict has:
    patient_id, patient_name, appointment_count, note_match_count.
    """
    safe_keyword = keyword.strip()
    name_pattern = f"%{doctor_name.strip()}%"

    with connection() as conn:
        rows = conn.execute(
            """
            SELECT
                p.patient_id,
                p.last_name || ', ' || p.first_name AS patient_name,
                COUNT(DISTINCT a.appointment_id)     AS appointment_count,
                COUNT(DISTINCT mr.record_id)          AS note_match_count
            FROM appointments a
            JOIN patients p  ON p.patient_id  = a.patient_id
            JOIN doctors  d  ON d.doctor_id   = a.doctor_id
            JOIN medical_records mr ON mr.patient_id = a.patient_id
            WHERE (
                    d.last_name  LIKE ? ESCAPE '\\'
                 OR d.first_name LIKE ? ESCAPE '\\'
                 OR (d.last_name || ', ' || d.first_name) LIKE ? ESCAPE '\\'
              )
              AND mr.note LIKE ? ESCAPE '\\'
            GROUP BY p.patient_id, patient_name
            ORDER BY patient_name
            """,
            (name_pattern, name_pattern, name_pattern, f"%{safe_keyword}%"),
        ).fetchall()
    return [dict(row) for row in rows]


def update_patient_contact_info(
    patient_id: int,
    telephone: str,
    address: str,
    insurance_carrier: str,
) -> None:
    with connection() as conn:
        conn.execute(
            """
            UPDATE patients
            SET
                telephone = ?,
                address = ?,
                insurance_carrier = ?
            WHERE patient_id = ?
            """,
            (telephone.strip(), address.strip(), insurance_carrier.strip(), patient_id),
        )
