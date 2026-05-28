# Healthcare Assistant (Streamlit + CrewAI)

A capstone course project demonstrating a multi-role medical office AI workflow built with CrewAI, Streamlit, SQLite, and ChromaDB. Three distinct user roles — Patient, Doctor, and Admin — each interact with a dedicated AI agent that has access to only the tools appropriate for that role.

---

## Table of Contents

- [Architecture Overview](#architecture-overview)
- [Running the App](#running-the-app)
- [Authentication](#authentication)
- [Database](#database)
- [Agent & Task Hierarchy](#agent--task-hierarchy)
  - [Admin Agent](#admin-agent)
  - [Doctor Agent](#doctor-agent)
  - [Patient Agent](#patient-agent)
- [Tool Reference](#tool-reference)
  - [Admin Tools (12)](#admin-tools-12)
  - [Doctor Tools (8)](#doctor-tools-8)
  - [Patient Tools (5)](#patient-tools-5)
- [UI Pages](#ui-pages)
- [Project Structure](#project-structure)
- [Notes](#notes)

---

## Architecture Overview

```
Streamlit UI  ──►  run_role_crew()  ──►  CrewAI Crew
                         │                    │
                    role dispatch         Agent + Task
                         │                    │
              ┌──────────┼──────────┐     Tools call
              │          │          │         │
           Admin       Doctor    Patient   database/
          Agent        Agent      Agent    db.py (SQLite)
         (global      (per-      (per-    + ChromaDB
          tools)      doctor)   patient)   (vectors)
```

Each call to `run_role_crew()` (in `app/crew_agents.py`) builds one `Crew` with one `Agent` and one `Task`, then calls `crew.kickoff()`. All crews use `Process.sequential`. Chat history (up to 10 prior messages) is embedded in the task description so the agent has conversational context.

If CrewAI or an LLM is unavailable, `run_role_crew()` falls back to a local canned response so the UI remains usable for demos.

---

## Running the App

1. Create and activate a Python virtual environment.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. (Optional) Set the shared login password:

```bash
# Windows PowerShell
$env:CAPSTONE_PASSWORD = "yourpassword"
```

4. Start the app:

```bash
streamlit run app/main.py
```

The database is created and seeded automatically on first run.

---

## Authentication

Login is intentionally simple for class demonstration. Users enter their **last name** as their username (case-insensitive) and an optional shared password controlled by the `CAPSTONE_PASSWORD` environment variable (defaults to no password).

| Role    | Valid usernames               |
|---------|-------------------------------|
| Patient | Any of the 100 seeded patient last names (all begin with "P") |
| Doctor  | `Dawson`, `Diaz`, `Dalton`, `Duncan`, `Dorsey`, `Delgado` |
| Admin   | `admin`                       |

After login, the Streamlit session stores `role`, `user_id`, and `user_name`. Each page enforces its own role check and redirects to the login page if the check fails.

---

## Database

The app uses two data stores:

### SQLite (`database/healthcare.db`)

Auto-created and seeded on first run. Contains:

- **100 patients** (IDs 1000–1099). All last names begin with "P". Age distribution is weighted toward middle-aged and elderly patients (30 % ages 25–44, 50 % ages 45–69, 20 % ages 70–85).
- **6 doctors** (IDs 100–105):

  | ID  | Name            | Specialty        |
  |-----|-----------------|------------------|
  | 100 | Nina Dawson     | Nephrology       |
  | 101 | Sam Diaz        | Cardiology       |
  | 102 | Iris Dalton     | General Practice |
  | 103 | Leo Duncan      | Pulmonology      |
  | 104 | Maya Dorsey     | Orthopedics      |
  | 105 | Owen Delgado    | Internal Medicine|

- **Medical records** — 30 per patient (3 per year × 10 years). Notes cycle through four formats (SOAP, bullet-list, narrative, structured key-value) and are synthesised with OpenAI using per-patient profiles (conditions, medications, vitals baselines). No records are created for future dates.
- **Appointments** — 2 per patient, scheduled on weekdays within a 4-week window starting from the seeding date. 80 % of patients see a single doctor for both appointments; 20 % see two different doctors.

### ChromaDB (`database/chroma/`)

A persistent vector store with a single collection `patient_history`. Medical record notes are chunked (480-character window, 120-character overlap, word-boundary splits) and embedded using ChromaDB's default embedding function (cosine similarity). The vector store powers the semantic-search tools available to the Admin and Doctor agents.

---

## Agent & Task Hierarchy

All agent/task construction lives in `app/crew_agents.py`. The entry point is:

```python
run_role_crew(role, prompt, context_rows, chat_history, *, patient_id, patient_name, doctor_id, doctor_name)
```

### Admin Agent

**Triggered when:** `role == "admin"` (or any unrecognised role).

```
Crew  (Process.sequential)
 ├─ Agent: "Admin Agent"
 │    goal:      Support clinic operations — answer scheduling questions,
 │               look up patient data, and book appointments.
 │    backstory:  Manages clinic operations and data visibility. Collects
 │               booking details one question at a time (patient → doctor →
 │               date → available slots → time → confirm).
 │    tools:     All 12 ADMIN_TOOLS (see Tool Reference)
 │    delegation: disabled
 │
 ├─ Agent: "Doctor Agent"  (present in crew but only assigned tasks for role=="doctor" fallback)
 │    goal:      Summarize patient and appointment data for clinicians.
 │    backstory:  Concise and clinically structured.
 │    tools:     none
 │    delegation: disabled
 │
 └─ Task
      assigned to: Admin Agent
      description: Active role, today's date, recent chat history (≤10 msgs),
                   schedule data preview (≤60 rows), and the user's request.
      rules injected: use tools for exact counts; collect booking details one
                      at a time; never re-ask for information already given.
      expected_output: Concise, role-appropriate answer with actionable next steps.
```

### Doctor Agent

**Triggered when:** `role == "doctor"` and `doctor_id` is provided.

```
Crew  (Process.sequential)
 └─ Agent: "Doctor Assistant"
      goal:      Help the doctor manage their schedule, access patient records,
                 and document clinical encounters.
      backstory:  Clinical assistant for the logged-in doctor. Handles schedule
                 queries, patient chart review, new record entries, and booking/
                 cancelling patient appointments. May answer general medical
                 questions from knowledge without using tools.
      tools:     8 doctor-scoped tools (see Tool Reference)
      delegation: disabled

 └─ Task
      assigned to: Doctor Agent
      description: Today's date, logged-in doctor identity, recent chat history,
                   and the doctor's message.
      rules injected: use get_my_schedule/get_my_next_appointment for schedule;
                      use get_patient_records or search_patient_history_for_doctor
                      for charts; follow booking workflow (patient → date →
                      get_my_available_slots → time → book); answer general
                      clinical questions directly without tools.
      expected_output: Concise, clinically appropriate response.
```

### Patient Agent

**Triggered when:** `role == "patient"` and `patient_id` is provided.

```
Crew  (Process.sequential)
 └─ Agent: "Patient Assistant"
      goal:      Help the patient manage their appointments and understand
                 their medical records.
      backstory:  Friendly patient-facing assistant for the logged-in patient.
                 Hard safety rules (injected into backstory and task):
                   1. NEVER provide a medical diagnosis or prescribe treatment.
                   2. If life-threatening symptoms are described, respond
                      immediately: "Please call 911 or go to your nearest
                      emergency room right away."
                   3. For non-urgent health questions, sympathize and recommend
                      scheduling an appointment.
      tools:     5 patient-scoped tools (see Tool Reference)
      delegation: disabled

 └─ Task
      assigned to: Patient Agent
      description: Today's date, logged-in patient identity, recent chat history,
                   and the patient's message.
      rules injected: no diagnosis/prescription; 911 for emergencies; booking
                      workflow (doctor → date → get_available_slots → time →
                      book_my_appointment); cancellation workflow
                      (get_my_appointments → confirm → cancel_my_appointment).
      expected_output: Concise, empathetic, patient-appropriate response.
```

---

## Tool Reference

Tools are implemented with the `@tool` decorator from `crewai.tools`. Each tool's docstring serves as the description the LLM uses to decide when to call it.

### Admin Tools (12)

These tools are module-level globals assigned to `ADMIN_TOOLS` and given to the Admin Agent.

| Tool | Description |
|------|-------------|
| `get_database_counts` | Returns exact counts of patients, doctors, and appointments from the database. |
| `get_patient_roster` | Returns the full patient list, one patient per line in `Last, First` format. |
| `get_patient_appointments` | Lists all upcoming and past appointments for a named patient. |
| `get_doctor_schedule` | Lists all appointments for a named doctor on a specific date with an exact count. |
| `get_patient_medical_records` | Fetches the 12 most recent visit notes for a named patient (SQLite). |
| `get_available_slots` | Shows open clinic time slots for a named doctor on a specific date. |
| `schedule_appointment` | Books a new appointment for a patient with a doctor and writes it to the database. Parses flexible time formats (`15:00`, `3:00pm`, `3pm`). |
| `search_patient_history` | Semantic (vector) search through a single patient's medical history notes in ChromaDB. Returns top-8 chunks with similarity scores. |
| `search_all_patients_history` | Cross-patient semantic search across all records in ChromaDB. Returns the best-scoring chunk per patient, ranked by similarity. |
| `find_patients_with_keyword_in_records` | Exact SQL keyword/phrase search across all patient notes. Returns matching patient names, note counts, and date of most recent match. |
| `find_doctor_patients_with_keyword` | Finds all patients of a specific doctor (across any date) whose records contain an exact keyword. |
| `find_doctor_day_patients_with_keyword` | Finds patients who had an appointment with a specific doctor on a specific date AND whose records contain an exact keyword (SQL join). |

### Doctor Tools (8)

Built by `_make_doctor_tools(doctor_id, doctor_name)` and scoped to the logged-in doctor. A new set of closures is created per session.

| Tool | Description |
|------|-------------|
| `get_my_schedule` | Lists this doctor's appointments for a given date with visit types and statuses. |
| `get_my_next_appointment` | Returns the doctor's next upcoming non-cancelled appointment. |
| `get_patient_records` | Fetches the 12 most recent medical records for a named patient (chart review). |
| `search_patient_history_for_doctor` | Semantic search through a named patient's history for targeted clinical detail (medications, diagnoses, lab values, symptoms). |
| `add_patient_medical_record` | Adds a new medical record note for a named patient, linked to this doctor. Supports encounter types: `OfficeVisit`, `FollowUpVisit`, `LabResult`, etc. |
| `book_appointment_for_patient` | Books an appointment for a named patient with this doctor. Parses flexible time formats. |
| `cancel_patient_appointment` | Cancels the next upcoming appointment for a named patient with this doctor. |
| `get_my_available_slots` | Shows open appointment slots for this doctor on a specific date. |

### Patient Tools (5)

Built by `_make_patient_tools(patient_id, patient_name)` and scoped to the logged-in patient. A new set of closures is created per session.

| Tool | Description |
|------|-------------|
| `get_my_appointments` | Lists this patient's upcoming and past appointments, including appointment IDs needed for cancellation. |
| `get_my_medical_records` | Shows the 10 most recent medical records for this patient. |
| `book_my_appointment` | Books an appointment for this patient with a named doctor. Parses flexible time formats. |
| `cancel_my_appointment` | Cancels one of this patient's appointments by appointment ID. Verifies ownership before cancelling. |
| `get_available_slots` | Shows open clinic slots for a named doctor on a date (shared helper, not patient-scoped). |

---

## UI Pages

| File | Role | Description |
|------|------|-------------|
| `app/main.py` | All | Entry point. Initialises and seeds the database, then routes to the correct page based on session role. |
| `app/_pages/login.py` | — | Role selector (Patient / Doctor / Admin) and username/password form. |
| `app/_pages/patient.py` | Patient | Sidebar menu: AI assistant chat, appointments table, medical records table. |
| `app/_pages/doctor.py` | Doctor | Sidebar menu: AI assistant chat, daily schedule (interactive Gantt chart 7 AM–7 PM), per-patient chart editor (add/edit/delete records). |
| `app/_pages/admin.py` | Admin | Sidebar menu: AI assistant chat, patient list with inline edit dialog, appointment schedule grid with day navigation, analytics dashboard (Altair charts), database management (reseed, vector sync). |

---

## Project Structure

```
app/
  main.py              # Streamlit entry point; DB init and page routing
  crew_agents.py       # All CrewAI agents, tasks, tools, and run_role_crew()
  db.py                # Re-export shim: forwards database/ symbols into app/ namespace
  _pages/
    login.py           # Login page
    patient.py         # Patient UI
    doctor.py          # Doctor UI
    admin.py           # Admin UI
database/
  db.py                # SQLite schema, seeding logic, ChromaDB vector store
  healthcare.db        # SQLite database (auto-created)
  chroma/              # ChromaDB persistent vector store
requirements.txt       # streamlit, crewai, openai, chromadb
```

---

## Notes

- This is demo-only software and is **not** for real medical use.
- The Patient Agent enforces hard safety guardrails: it will never diagnose or prescribe, and it directs patients with emergency symptoms to call 911 immediately.
- Login is intentionally simple (last-name username) for classroom demonstration.
- If CrewAI or an LLM is unavailable, `run_role_crew()` returns a local fallback response so the UI continues to function for demos.
- Database seeding uses OpenAI to generate per-patient profiles (one API call per patient), then synthesises all 30 medical record notes locally from those profiles.
