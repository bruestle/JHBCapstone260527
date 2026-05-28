from __future__ import annotations

from datetime import date
from typing import Iterable

from db import (
    add_medical_record,
    cancel_appointment,
    create_appointment,
    fetch_appointments_for_day,
    fetch_appointments_for_doctor,
    fetch_appointments_for_patient,
    fetch_counts,
    fetch_doctor_day_patients_with_keyword,
    fetch_doctor_list,
    fetch_doctor_patients_with_keyword,
    fetch_available_slots,
    fetch_medical_records,
    fetch_patient_list,
    fetch_patient_roster,
    search_all_patients_history_semantic,
    search_patient_history_semantic,
    search_patients_by_record_keyword,
)

try:
    from crewai import Agent, Crew, Process, Task
    from crewai.tools import tool

    CREWAI_AVAILABLE = True
except Exception:
    CREWAI_AVAILABLE = False


if CREWAI_AVAILABLE:

    @tool("get_database_counts")
    def get_database_counts() -> str:
        """Return exact database counts for patients, doctors, and appointments."""
        counts = fetch_counts()
        return (
            f"patients={counts['patients']}, "
            f"doctors={counts['doctors']}, "
            f"appointments={counts['appointments']}"
        )


    @tool("get_patient_roster")
    def get_patient_roster() -> str:
        """Return the full patient roster as one unique patient per line in Last, First format."""
        roster = fetch_patient_roster()
        if not roster:
            return "No patients in database."
        return "\n".join(roster)


    def _find_patient(name_query: str) -> dict | None:
        """Return the first patient whose name contains name_query (case-insensitive).

        Handles inputs as: last name only, 'Last, First', or 'First Last'.
        """
        needle = name_query.strip().lower()

        # If given "First Last", also try matching as "Last, First"
        reversed_needle: str | None = None
        parts = needle.split()
        if len(parts) == 2:
            reversed_needle = f"{parts[1]}, {parts[0]}"

        for p in fetch_patient_list():
            full = (p.get("patient_name") or "").lower()
            last = full.split(",")[0].strip()
            if needle in full or needle == last:
                return p
            if reversed_needle and (reversed_needle in full or reversed_needle == full):
                return p
        return None


    @tool("get_patient_medical_records")
    def get_patient_medical_records(patient_name: str) -> str:
        """Fetch the most recent medical record notes for a patient by name.

        Input: the patient's last name or 'Last, First' full name.
        Returns the 12 most recent visit notes so you can summarise the patient's condition.
        """
        match = _find_patient(patient_name)
        if match is None:
            return f"No patient found matching '{patient_name}'."

        records = fetch_medical_records(str(match["patient_id"]))
        if not records:
            return f"No medical records found for {match['patient_name']}."

        recent = records[:12]  # already ordered DESC by created_at
        lines = [
            f"Medical records for {match['patient_name']} "
            f"(showing {len(recent)} of {len(records)} most recent):"
        ]
        for r in recent:
            note_preview = (r.get("note") or "").replace("\n", " ")[:450]
            lines.append(f"[{r['created_at']}] {note_preview}")
        return "\n".join(lines)


    @tool("search_patient_history")
    def search_patient_history(patient_name: str, query: str) -> str:
        """Semantic search through a patient's medical history notes in the vector database.

        Use this to find specific clinical information such as medications, diagnoses,
        lab results, or symptoms. More targeted than fetching all records.
        Input: patient_name (last name or full name), query (what to look for).
        """
        match = _find_patient(patient_name)
        if match is None:
            return f"No patient found matching '{patient_name}'."

        results = search_patient_history_semantic(
            patient_id=match["patient_id"], query=query, top_k=8
        )
        if not results:
            return (
                f"No relevant records found for '{query}' in {match['patient_name']}'s history. "
                "Try get_patient_medical_records for a full record list."
            )

        lines = [
            f"Semantic search results for '{query}' in {match['patient_name']}'s history:"
        ]
        for r in results:
            score = f"{r['similarity']:.2f}" if r.get("similarity") is not None else "?"
            snippet = r["text"].replace("\n", " ")[:400]
            lines.append(f"[{r['created_at']}] (score {score}): {snippet}")
        return "\n".join(lines)


    @tool("search_all_patients_history")
    def search_all_patients_history(query: str) -> str:
        """Search medical history across ALL patients in the vector database.

        Use this for cross-patient questions such as:
        - How many patients are taking a specific medication?
        - Which patients have a particular diagnosis or condition?
        - Find all patients with a specific symptom or lab result.

        Returns matching chunks grouped by patient with a count of distinct patients found.
        Input: a natural-language search query (e.g. 'Metformin', 'type 2 diabetes', 'elevated LDL').
        """
        results = search_all_patients_history_semantic(query=query, top_k=120)
        if not results:
            return f"No records found matching '{query}' across all patient histories."

        # Build a name lookup once
        name_map: dict[int, str] = {
            p["patient_id"]: p["patient_name"] for p in fetch_patient_list()
        }

        # Group chunks by patient; keep only the best-scoring chunk per patient
        best_by_patient: dict[int, dict] = {}
        for r in results:
            pid = int(r["patient_id"]) if r["patient_id"] != "" else -1
            sim = r.get("similarity") or 0.0
            if pid not in best_by_patient or sim > (best_by_patient[pid].get("similarity") or 0.0):
                best_by_patient[pid] = r

        # Sort by similarity descending
        ranked = sorted(best_by_patient.values(), key=lambda x: x.get("similarity") or 0.0, reverse=True)

        lines = [
            f"Cross-patient search for '{query}': found relevant records for {len(ranked)} distinct patient(s).\n"
        ]
        for r in ranked:
            pid = int(r["patient_id"]) if r["patient_id"] != "" else -1
            name = name_map.get(pid, f"Patient {pid}")
            score = f"{r['similarity']:.2f}" if r.get("similarity") is not None else "?"
            snippet = r["text"].replace("\n", " ")[:350]
            lines.append(f"- {name} (score {score}): {snippet}")

        return "\n".join(lines)


    @tool("find_patients_with_keyword_in_records")
    def find_patients_with_keyword_in_records(keyword: str) -> str:
        """Find all patients whose medical record notes contain an exact keyword or phrase.

        Use this — NOT semantic search — when the question asks for a specific drug name,
        diagnosis code, lab value, or any term that must literally appear in the notes.
        Examples: 'Metformin', 'type 2 diabetes', 'lisinopril', 'HbA1c'.

        Returns the list of matching patients, how many of their notes mention the term,
        and the date of their most recent matching note.
        Input: the exact word or phrase to search for (case-insensitive).
        """
        matches = search_patients_by_record_keyword(keyword)
        if not matches:
            return f"No patient records contain the exact text '{keyword}'."

        lines = [
            f"Exact keyword search for '{keyword}': "
            f"{len(matches)} patient(s) have at least one note containing this term.\n"
        ]
        for m in matches:
            lines.append(
                f"- {m['patient_name']}  "
                f"({m['match_count']} note(s) mentioning '{keyword}', "
                f"latest: {m['latest_match']})"
            )
        return "\n".join(lines)


    @tool("get_doctor_schedule")
    def get_doctor_schedule(doctor_name: str, date_str: str) -> str:
        """List all appointments for a doctor on a specific date and return the exact count.

        Always use this tool for questions like:
        - 'How many appointments does Dr. Diaz have today?'
        - 'Who does Dr. Smith see on June 10?'
        Never count from the schedule preview — query directly.
        Inputs:
          doctor_name: doctor's last name (e.g. 'Diaz')
          date_str:    ISO date (e.g. '2026-05-26')
        """
        needle = doctor_name.strip().lower()
        doctor = None
        for d in fetch_doctor_list():
            last = (d.get("doctor_name") or "").split(",")[0].strip().lower()
            if needle == last or needle in (d.get("doctor_name") or "").lower():
                doctor = d
                break
        if doctor is None:
            return f"No doctor found matching '{doctor_name}'."

        rows = [
            r for r in fetch_appointments_for_day(date_str)
            if r["doctor_id"] == doctor["doctor_id"]
        ]
        if not rows:
            return f"Dr. {doctor['doctor_name']} has no appointments on {date_str}."

        lines = [
            f"Dr. {doctor['doctor_name']} has {len(rows)} appointment(s) on {date_str}:"
        ]
        for r in rows:
            lines.append(
                f"  {r['start_at']}  {r['patient_name']}  [{r.get('visit_type', '')}]"
            )
        return "\n".join(lines)


    @tool("find_doctor_patients_with_keyword")
    def find_doctor_patients_with_keyword(doctor_name: str, keyword: str) -> str:
        """Find ALL patients of a doctor (across any date) whose medical records contain a keyword.

        Use this for general questions like:
        - 'How many of Dr. Diaz\'s patients are on Metformin?'
        - 'Which of Dr. Smith\'s patients have diabetes?'

        Use find_doctor_day_patients_with_keyword instead only when a specific date is given.
        Inputs:
          doctor_name: doctor's last name (e.g. 'Diaz')
          keyword:     exact term to search in records (e.g. 'Metformin')
        """
        matches = fetch_doctor_patients_with_keyword(doctor_name, keyword)
        if not matches:
            return (
                f"No patients of Dr. {doctor_name} have '{keyword}' in their medical records."
            )
        lines = [
            f"{len(matches)} of Dr. {doctor_name}'s patient(s) have '{keyword}' in their records:\n"
        ]
        for m in matches:
            lines.append(
                f"  - {m['patient_name']}  "
                f"({m['note_match_count']} record(s) mentioning '{keyword}')"
            )
        return "\n".join(lines)


    @tool("find_doctor_day_patients_with_keyword")
    def find_doctor_day_patients_with_keyword(doctor_name: str, date_str: str, keyword: str) -> str:
        """Find patients who had an appointment with a specific doctor on a specific date
        AND whose medical records contain an exact keyword.

        Use this for questions like:
        - 'How many of Diaz's patients today are on Metformin?'
        - 'Which patients Dr. Smith saw on 2026-05-26 have a diabetes diagnosis?'

        This performs a precise SQL join — do NOT try to cross-reference manually.
        Inputs:
          doctor_name: doctor's last name (e.g. 'Diaz')
          date_str:    ISO date string (e.g. '2026-05-26')
          keyword:     exact term to find in records (e.g. 'Metformin')
        """
        matches = fetch_doctor_day_patients_with_keyword(doctor_name, date_str, keyword)
        if not matches:
            return (
                f"No patients with an appointment with Dr. {doctor_name} on {date_str} "
                f"have '{keyword}' in their medical records."
            )

        lines = [
            f"Patients seen by Dr. {doctor_name} on {date_str} with '{keyword}' in their records: "
            f"{len(matches)} patient(s).\n"
        ]
        for m in matches:
            lines.append(
                f"- {m['patient_name']}  "
                f"({m['appointment_count']} appt(s) that day, "
                f"{m['note_match_count']} record(s) mentioning '{keyword}')"
            )
        return "\n".join(lines)


    @tool("schedule_appointment")
    def schedule_appointment(patient_name: str, doctor_name: str, date_str: str, time_str: str) -> str:
        """Schedule a new appointment for a patient with a doctor and save it to the database.

        Use this whenever the user asks to add, book, create, or schedule an appointment.
        Inputs:
          patient_name: patient's last name or 'Last, First' full name
          doctor_name:  doctor's last name (e.g. 'Diaz')
          date_str:     ISO date string (e.g. '2026-06-10')
          time_str:     time in any common format ('15:00', '3:00pm', or '3pm')
        Returns confirmation text with the full appointment details.
        """
        from datetime import datetime as _dt

        patient = _find_patient(patient_name)
        if patient is None:
            return f"No patient found matching '{patient_name}'."

        needle = doctor_name.strip().lower()
        doctor = None
        for d in fetch_doctor_list():
            last = (d.get("doctor_name") or "").split(",")[0].strip().lower()
            if needle == last or needle in (d.get("doctor_name") or "").lower():
                doctor = d
                break
        if doctor is None:
            return f"No doctor found matching '{doctor_name}'."

        # Parse flexible time formats
        clean_time = time_str.strip().lower().replace(" ", "")
        parsed_time = None
        for fmt in ("%H:%M", "%I:%M%p", "%I%p"):
            try:
                parsed_time = _dt.strptime(clean_time, fmt)
                break
            except ValueError:
                continue
        if parsed_time is None:
            return (
                f"Could not parse time '{time_str}'. "
                "Use formats like '15:00', '3:00pm', or '3pm'."
            )

        start_at = f"{date_str} {parsed_time.strftime('%H:%M')}"
        appt_id = create_appointment(
            patient_id=patient["patient_id"],
            doctor_id=doctor["doctor_id"],
            start_at=start_at,
        )

        return (
            f"Appointment successfully scheduled and saved to the database.\n"
            f"  Patient:   {patient['patient_name']}\n"
            f"  Doctor:    {doctor['doctor_name']} ({doctor.get('specialty', '')})\n"
            f"  Date/Time: {start_at}\n"
            f"  Duration:  60 min\n"
            f"  Status:    booked\n"
            f"  ID:        {appt_id}"
        )


    @tool("get_patient_appointments")
    def get_patient_appointments(patient_name: str) -> str:
        """List all appointments for a patient, separated into upcoming and past.

        Always use this tool when asked about a patient's upcoming, future, or past
        appointments. Never rely on the schedule preview for this — query directly.
        Input:
          patient_name: patient's last name or 'Last, First'
        """
        patient = _find_patient(patient_name)
        if patient is None:
            return f"No patient found matching '{patient_name}'."

        rows = fetch_appointments_for_patient(patient["patient_id"])
        if not rows:
            return f"No appointments found for {patient['patient_name']}."

        today_str = date.today().isoformat()
        upcoming, past = [], []
        for r in rows:
            (upcoming if r["start_at"] >= today_str else past).append(r)

        def _fmt_row(r: dict) -> str:
            return (
                f"  {r['start_at']}  "
                f"{r['doctor_name']} ({r.get('specialty', '')})  "
                f"[{r.get('visit_type', '')}]  status={r.get('status', '')}"
            )

        lines = [f"Appointments for {patient['patient_name']}:"]
        if upcoming:
            lines.append(f"Upcoming ({len(upcoming)}):")
            lines.extend(_fmt_row(r) for r in upcoming)
        else:
            lines.append("Upcoming: none")
        if past:
            lines.append(f"Past ({len(past)}):")
            lines.extend(_fmt_row(r) for r in past[-5:])  # last 5 only
        return "\n".join(lines)


    @tool("get_available_slots")
    def get_available_slots(doctor_name: str, date_str: str) -> str:
        """Get open appointment slots for a doctor on a specific date.

        Call this as soon as the user provides both a doctor name and a date during
        booking — before asking for the time. Shows only clinic slots not already booked.
        Inputs:
          doctor_name: doctor's last name (e.g. 'Diaz')
          date_str:    ISO date (e.g. '2026-06-10')
        """
        needle = doctor_name.strip().lower()
        doctor = None
        for d in fetch_doctor_list():
            last = (d.get("doctor_name") or "").split(",")[0].strip().lower()
            if needle == last or needle in (d.get("doctor_name") or "").lower():
                doctor = d
                break
        if doctor is None:
            return f"No doctor found matching '{doctor_name}'."

        slots = fetch_available_slots(doctor["doctor_id"], date_str)
        if not slots:
            return f"Dr. {doctor_name} has no available slots on {date_str}."

        def _fmt(hhmm: str) -> str:
            h, m = map(int, hhmm.split(":"))
            period = "am" if h < 12 else "pm"
            h12 = h % 12 or 12
            return f"{h12}:{m:02d}{period}" if m else f"{h12}{period}"

        formatted = ", ".join(_fmt(s) for s in slots)
        return (
            f"Available slots for Dr. {doctor['doctor_name']} on {date_str}:\n"
            f"{formatted}"
        )


    ADMIN_TOOLS = [
        get_database_counts,
        get_patient_roster,
        get_patient_appointments,
        get_doctor_schedule,
        get_patient_medical_records,
        get_available_slots,
        schedule_appointment,
        search_patient_history,
        search_all_patients_history,
        find_patients_with_keyword_in_records,
        find_doctor_patients_with_keyword,
        find_doctor_day_patients_with_keyword,
    ]

    # ------------------------------------------------------------------ #
    # Patient-scoped tool factory                                          #
    # ------------------------------------------------------------------ #

    def _make_patient_tools(patient_id: str, patient_name: str) -> list:
        """Return tool instances locked to a specific patient's data."""

        @tool("get_my_appointments")
        def get_my_appointments() -> str:
            """List MY upcoming and past appointments from the database.

            Always use this when the patient asks about their appointments,
            next visit, upcoming schedule, or appointment history.
            Returns appointment IDs needed to cancel an appointment.
            """
            rows = fetch_appointments_for_patient(patient_id)
            if not rows:
                return "You have no appointments on record."

            today_str = date.today().isoformat()
            upcoming, past = [], []
            for r in rows:
                (upcoming if r["start_at"] >= today_str else past).append(r)

            def _fmt(r: dict) -> str:
                return (
                    f"  {r['start_at']}  "
                    f"Dr. {r['doctor_name']} ({r.get('specialty', '')})  "
                    f"[{r.get('visit_type', '')}]  status={r.get('status', '')}  "
                    f"id={r['appointment_id']}"
                )

            lines = [f"Appointments for {patient_name}:"]
            if upcoming:
                lines.append(f"Upcoming ({len(upcoming)}):")
                lines.extend(_fmt(r) for r in upcoming)
            else:
                lines.append("Upcoming: none")
            if past:
                lines.append(f"Recent past ({min(5, len(past))}):")
                lines.extend(_fmt(r) for r in past[-5:])
            return "\n".join(lines)

        @tool("get_my_medical_records")
        def get_my_medical_records() -> str:
            """Show MY most recent medical records, visit notes, and lab results.

            Use this when the patient asks about their medical history,
            lab results, test results, diagnoses on file, or past visit notes.
            """
            records = fetch_medical_records(patient_id)
            if not records:
                return "No medical records found for you."

            recent = records[:10]
            lines = [
                f"Medical records for {patient_name} "
                f"(showing {len(recent)} most recent of {len(records)} total):"
            ]
            for r in recent:
                note_preview = (r.get("note") or "").replace("\n", " ")[:400]
                lines.append(f"[{r['created_at']}] {note_preview}")
            return "\n".join(lines)

        @tool("book_my_appointment")
        def book_my_appointment(doctor_name: str, date_str: str, time_str: str) -> str:
            """Book an appointment for ME with a specific doctor on a given date and time.

            Call get_available_slots first to confirm a slot is open.
            Inputs:
              doctor_name: doctor's last name (e.g. 'Duncan')
              date_str:    ISO date (e.g. '2026-05-28')
              time_str:    time in any common format ('15:00', '3:00pm', or '3pm')
            """
            from datetime import datetime as _dt

            needle = doctor_name.strip().lower()
            doctor = None
            for d in fetch_doctor_list():
                last = (d.get("doctor_name") or "").split(",")[0].strip().lower()
                if needle == last or needle in (d.get("doctor_name") or "").lower():
                    doctor = d
                    break
            if doctor is None:
                return f"No doctor found matching '{doctor_name}'."

            clean_time = time_str.strip().lower().replace(" ", "")
            parsed_time = None
            for fmt in ("%H:%M", "%I:%M%p", "%I%p"):
                try:
                    parsed_time = _dt.strptime(clean_time, fmt)
                    break
                except ValueError:
                    continue
            if parsed_time is None:
                return f"Could not parse time '{time_str}'. Use formats like '15:00', '3:00pm', or '3pm'."

            start_at = f"{date_str} {parsed_time.strftime('%H:%M')}"
            appt_id = create_appointment(
                patient_id=patient_id,
                doctor_id=doctor["doctor_id"],
                start_at=start_at,
            )
            return (
                f"Appointment booked successfully!\n"
                f"  Doctor:    Dr. {doctor['doctor_name']} ({doctor.get('specialty', '')})\n"
                f"  Date/Time: {start_at}\n"
                f"  Status:    booked\n"
                f"  ID:        {appt_id}"
            )

        @tool("cancel_my_appointment")
        def cancel_my_appointment(appointment_id: str) -> str:
            """Cancel one of MY upcoming appointments by its appointment ID.

            Use get_my_appointments first to see appointment IDs.
            Input:
              appointment_id: the full appointment ID string shown in get_my_appointments
            """
            ok = cancel_appointment(appointment_id, patient_id)
            if ok:
                return f"Appointment {appointment_id} has been cancelled successfully."
            return (
                f"Could not cancel appointment {appointment_id}. "
                "It may not exist, may already be cancelled, or may not belong to you."
            )

        return [
            get_my_appointments,
            get_my_medical_records,
            book_my_appointment,
            cancel_my_appointment,
            get_available_slots,
        ]

    # ------------------------------------------------------------------ #
    # Doctor-scoped tool factory                                           #
    # ------------------------------------------------------------------ #

    def _make_doctor_tools(doctor_id: str, doctor_name: str) -> list:
        """Return tool instances scoped to a specific doctor."""

        @tool("get_my_schedule")
        def get_my_schedule(date_str: str) -> str:
            """Get MY appointment schedule for a given date.

            Use this when asked about today's schedule, how many patients I have,
            or who I see on a specific date.
            Input: date_str — ISO date (e.g. '2026-05-27')
            """
            rows = [
                r for r in fetch_appointments_for_day(date_str)
                if str(r.get("doctor_id")) == str(doctor_id)
            ]
            if not rows:
                return f"No appointments on {date_str}."
            lines = [f"My schedule for {date_str} ({len(rows)} appointment(s)):"]
            for r in rows:
                lines.append(
                    f"  {r['start_at']}  {r['patient_name']}  "
                    f"[{r.get('visit_type', '')}]  status={r.get('status', '')}"
                )
            return "\n".join(lines)

        @tool("get_my_next_appointment")
        def get_my_next_appointment() -> str:
            """Get my next upcoming appointment.

            Use this when asked 'When is my next appointment?' or similar.
            """
            today_str = date.today().isoformat()
            rows = [
                r for r in fetch_appointments_for_doctor(doctor_id)
                if r.get("start_at", "") >= today_str and r.get("status") != "cancelled"
            ]
            if not rows:
                return "No upcoming appointments found."
            rows.sort(key=lambda r: r["start_at"])
            r = rows[0]
            return (
                f"Your next appointment:\n"
                f"  {r['start_at']}  {r['patient_name']}  "
                f"[{r.get('visit_type', '')}]  status={r.get('status', '')}"
            )

        @tool("get_patient_records")
        def get_patient_records(patient_name: str) -> str:
            """Fetch the most recent medical records for a named patient.

            Use this when asked for a patient's recent lab results, visit notes,
            or to review their chart before seeing them.
            Input: patient's last name, 'Last, First', or 'First Last'.
            Returns the 12 most recent records.
            """
            match = _find_patient(patient_name)
            if match is None:
                return f"No patient found matching '{patient_name}'."
            records = fetch_medical_records(str(match["patient_id"]))
            if not records:
                return f"No medical records found for {match['patient_name']}."
            recent = records[:12]
            lines = [
                f"Medical records for {match['patient_name']} "
                f"(showing {len(recent)} of {len(records)} most recent):"
            ]
            for r in recent:
                etype = r.get("encounter_type") or ""
                note_preview = (r.get("note") or "").replace("\n", " ")[:450]
                lines.append(f"[{r['created_at']}] [{etype}] {note_preview}")
            return "\n".join(lines)

        @tool("search_patient_history_for_doctor")
        def search_patient_history_for_doctor(patient_name: str, query: str) -> str:
            """Semantic search through a patient's medical history for specific clinical info.

            Use this to find medications, diagnoses, lab values, symptoms, or any
            targeted clinical detail. More precise than fetching all records.
            Inputs:
              patient_name: last name or full name
              query: what to search for (e.g. 'lab results', 'HbA1c', 'current medications')
            """
            match = _find_patient(patient_name)
            if match is None:
                return f"No patient found matching '{patient_name}'."
            results = search_patient_history_semantic(
                patient_id=match["patient_id"], query=query, top_k=8
            )
            if not results:
                return (
                    f"No relevant records found for '{query}' in {match['patient_name']}'s history. "
                    "Try get_patient_records for the full record list."
                )
            lines = [f"Search results for '{query}' in {match['patient_name']}'s history:"]
            for r in results:
                score = f"{r['similarity']:.2f}" if r.get("similarity") is not None else "?"
                snippet = r["text"].replace("\n", " ")[:400]
                lines.append(f"[{r['created_at']}] (score {score}): {snippet}")
            return "\n".join(lines)

        @tool("add_patient_medical_record")
        def add_patient_medical_record(
            patient_name: str,
            note: str,
            encounter_type: str = "OfficeVisit",
        ) -> str:
            """Add a new medical record entry for a named patient.

            Use this when asked to 'make an entry', 'add a note', 'document',
            or 'record' something in a patient's chart.
            Inputs:
              patient_name: last name or 'Last, First' or 'First Last'
              note: full note text to record (include all clinical detail)
              encounter_type: e.g. OfficeVisit, FollowUpVisit, LabResult (default: OfficeVisit)
            """
            match = _find_patient(patient_name)
            if match is None:
                return f"No patient found matching '{patient_name}'."
            record_id = add_medical_record(
                patient_id=match["patient_id"],
                note=note,
                doctor_id=doctor_id,
                encounter_type=encounter_type,
            )
            return (
                f"Medical record added successfully.\n"
                f"  Patient:        {match['patient_name']}\n"
                f"  Encounter type: {encounter_type}\n"
                f"  Record ID:      {record_id}\n"
                f"  Note preview:   {note[:120]}{'...' if len(note) > 120 else ''}"
            )

        @tool("book_appointment_for_patient")
        def book_appointment_for_patient(
            patient_name: str,
            date_str: str,
            time_str: str,
        ) -> str:
            """Book an appointment for a named patient with ME (this doctor).

            Call get_my_available_slots first to confirm a slot is open.
            Inputs:
              patient_name: patient's last name or full name
              date_str: ISO date (e.g. '2026-06-10')
              time_str: time in any common format ('15:00', '3:00pm', '3pm')
            """
            from datetime import datetime as _dt

            patient = _find_patient(patient_name)
            if patient is None:
                return f"No patient found matching '{patient_name}'."

            clean_time = time_str.strip().lower().replace(" ", "")
            parsed_time = None
            for fmt in ("%H:%M", "%I:%M%p", "%I%p"):
                try:
                    parsed_time = _dt.strptime(clean_time, fmt)
                    break
                except ValueError:
                    continue
            if parsed_time is None:
                return f"Could not parse time '{time_str}'. Use formats like '15:00', '3:00pm', or '3pm'."

            start_at = f"{date_str} {parsed_time.strftime('%H:%M')}"
            appt_id = create_appointment(
                patient_id=patient["patient_id"],
                doctor_id=doctor_id,
                start_at=start_at,
            )
            return (
                f"Appointment booked.\n"
                f"  Patient:   {patient['patient_name']}\n"
                f"  Doctor:    {doctor_name}\n"
                f"  Date/Time: {start_at}\n"
                f"  Status:    booked\n"
                f"  ID:        {appt_id}"
            )

        @tool("cancel_patient_appointment")
        def cancel_patient_appointment(patient_name: str) -> str:
            """Cancel the next upcoming appointment for a named patient with THIS doctor.

            Use this when asked to cancel an appointment for a specific patient.
            Input: patient's last name or full name.
            """
            patient = _find_patient(patient_name)
            if patient is None:
                return f"No patient found matching '{patient_name}'."

            today_str = date.today().isoformat()
            rows = [
                r for r in fetch_appointments_for_patient(patient["patient_id"])
                if r.get("start_at", "") >= today_str
                and str(r.get("doctor_id", "")) == str(doctor_id)
                and r.get("status") != "cancelled"
            ]
            if not rows:
                return (
                    f"No upcoming appointments found for {patient['patient_name']} with {doctor_name}."
                )
            rows.sort(key=lambda r: r["start_at"])
            target = rows[0]
            ok = cancel_appointment(target["appointment_id"], patient["patient_id"])
            if ok:
                return (
                    f"Cancelled: {patient['patient_name']} on {target['start_at']}  "
                    f"(ID: {target['appointment_id']})"
                )
            return f"Could not cancel appointment {target['appointment_id']}."

        @tool("get_my_available_slots")
        def get_my_available_slots(date_str: str) -> str:
            """Get open appointment slots for ME on a specific date.

            Call this before booking to show what times are available.
            Input: date_str — ISO date (e.g. '2026-06-10')
            """
            slots = fetch_available_slots(doctor_id, date_str)
            if not slots:
                return f"No available slots on {date_str}."

            def _fmt(hhmm: str) -> str:
                h, m = map(int, hhmm.split(":"))
                period = "am" if h < 12 else "pm"
                h12 = h % 12 or 12
                return f"{h12}:{m:02d}{period}" if m else f"{h12}{period}"

            formatted = ", ".join(_fmt(s) for s in slots)
            return f"Available slots for {doctor_name} on {date_str}:\n{formatted}"

        return [
            get_my_schedule,
            get_my_next_appointment,
            get_patient_records,
            search_patient_history_for_doctor,
            add_patient_medical_record,
            book_appointment_for_patient,
            cancel_patient_appointment,
            get_my_available_slots,
        ]


else:
    ADMIN_TOOLS = []

    def _make_patient_tools(patient_id: str, patient_name: str) -> list:  # type: ignore[misc]
        return []

    def _make_doctor_tools(doctor_id: str, doctor_name: str) -> list:  # type: ignore[misc]
        return []


def _rows_preview(rows: Iterable[dict], limit: int | None = None) -> str:
    all_rows = list(rows)
    if limit is None:
        selected = all_rows
    else:
        selected = all_rows[:limit]

    if not selected:
        return "No rows available."

    lines = []
    for row in selected:
        lines.append(
            f"- {row.get('start_at', '')} | {row.get('patient_name', row.get('patient_id', ''))} | "
            f"{row.get('doctor_name', row.get('doctor_id', ''))} | {row.get('visit_type', '')} | {row.get('status', '')}"
        )

    if limit is not None and len(all_rows) > limit:
        lines.append(f"... ({len(all_rows) - limit} additional rows omitted)")

    return "\n".join(lines)


def _fallback_response(role: str, prompt: str, context: str) -> str:
    return (
        f"[{role} agent fallback] CrewAI was not available, so this is a local response.\n\n"
        f"You asked: {prompt}\n\n"
        f"Relevant data preview:\n{context}\n\n"
        "This is a course demo app and does not provide medical diagnosis or treatment advice."
    )


def run_role_crew(
    role: str,
    prompt: str,
    context_rows: Iterable[dict],
    chat_history: list[dict] | None = None,
    patient_id: str | None = None,
    patient_name: str | None = None,
    doctor_id: str | None = None,
    doctor_name: str | None = None,
) -> str:
    # Admin/doctor schedule queries need full context; patient chat can be lightly bounded.
    row_limit = 60 if role == "patient" else None
    context = _rows_preview(context_rows, limit=row_limit)

    # Build prior-conversation text (all messages before the current user prompt)
    history_text = ""
    if chat_history and len(chat_history) > 1:
        prior = chat_history[:-1]  # exclude the current message already in `prompt`
        recent = prior[-10:]       # cap at 10 prior messages
        lines = []
        for msg in recent:
            label = "User" if msg["role"] == "user" else "Assistant"
            lines.append(f"{label}: {msg['content']}")
        history_text = "Recent conversation:\n" + "\n".join(lines) + "\n\n"

    if not CREWAI_AVAILABLE:
        return _fallback_response(role, prompt, context)

    try:
        # ── Doctor role: dedicated agent with doctor-scoped tools ───────────
        if role == "doctor" and doctor_id:
            dname = doctor_name or doctor_id
            doctor_tools = _make_doctor_tools(doctor_id, dname)

            doctor_agent = Agent(
                role="Doctor Assistant",
                goal="Help the doctor manage their schedule, access patient records, and document clinical encounters.",
                backstory=(
                    f"You are a clinical assistant for Dr. {dname}. "
                    "You help with: checking your schedule, reviewing and summarizing patient "
                    "medical records, adding new record entries, and booking or cancelling "
                    "patient appointments. "
                    "You may also answer general medical and clinical questions using your knowledge — "
                    "for symptom questions or differential diagnoses, answer directly and concisely. "
                    "IMPORTANT: When the conversation shows you asked a question and the doctor's "
                    "latest message is a short reply (a name, date, or time), treat it as the "
                    "direct answer to your question and proceed."
                ),
                tools=doctor_tools,
                allow_delegation=False,
                verbose=False,
            )

            task = Task(
                description=(
                    f"Today's date is {date.today().isoformat()}.\n"
                    f"The logged-in doctor is: Dr. {dname} (doctor_id={doctor_id}).\n\n"
                    f"{history_text}"
                    f"Doctor's message: {prompt}\n\n"
                    "Rules:\n"
                    "- For schedule questions, use get_my_schedule or get_my_next_appointment.\n"
                    "- For patient records or lab results, use get_patient_records or "
                    "search_patient_history_for_doctor.\n"
                    "- When adding a record entry, ask for all details then call "
                    "add_patient_medical_record.\n"
                    "- When booking: ask patient → date → call get_my_available_slots → "
                    "confirm time → call book_appointment_for_patient.\n"
                    "- When cancelling: call cancel_patient_appointment with the patient name.\n"
                    "- For general medical/clinical questions (symptoms, differentials, treatment), "
                    "answer directly from your medical knowledge without using tools.\n"
                    "- Never re-ask for something already given in the conversation."
                ),
                expected_output="A concise, clinically appropriate response.",
                agent=doctor_agent,
            )

            crew = Crew(
                agents=[doctor_agent],
                tasks=[task],
                process=Process.sequential,
                verbose=False,
            )
            result = crew.kickoff()
            return str(result)

        # ── Patient role: dedicated agent with patient-scoped tools ──────────
        if role == "patient" and patient_id:
            pname = patient_name or patient_id
            patient_tools = _make_patient_tools(patient_id, pname)

            patient_agent = Agent(
                role="Patient Assistant",
                goal="Help the patient manage their appointments and understand their medical records.",
                backstory=(
                    f"You are a friendly patient-facing healthcare assistant for {pname}. "
                    "You help with: checking appointments, booking and cancelling appointments, "
                    "and reviewing medical records and lab results. "
                    "SAFETY RULES — follow these without exception:\n"
                    "1. NEVER provide a medical diagnosis or prescribe treatment.\n"
                    "2. If the patient describes symptoms that could be life-threatening "
                    "(chest pain, difficulty breathing, signs of stroke, severe bleeding, "
                    "loss of consciousness, or any other emergency), respond IMMEDIATELY with: "
                    "'Please call 911 or go to your nearest emergency room right away.' "
                    "Do not offer to book an appointment in this case.\n"
                    "3. For non-urgent symptoms or health questions, sympathize briefly and "
                    "recommend scheduling an appointment — then offer to help them book one.\n"
                    "4. When the conversation shows you asked a question and the patient's latest "
                    "message is a short reply (a name, date, time, or doctor), treat it as the "
                    "direct answer to your question and proceed."
                ),
                tools=patient_tools,
                allow_delegation=False,
                verbose=False,
            )

            task = Task(
                description=(
                    f"Today's date is {date.today().isoformat()}.\n"
                    f"The logged-in patient is: {pname} (patient_id={patient_id}).\n\n"
                    f"{history_text}"
                    f"Patient's message: {prompt}\n\n"
                    "Rules:\n"
                    "- NEVER diagnose or prescribe treatment.\n"
                    "- For life-threatening symptoms, say IMMEDIATELY: "
                    "'Please call 911 or go to your nearest emergency room right away.' "
                    "Do not suggest scheduling an appointment.\n"
                    "- For non-urgent health questions, sympathize and recommend booking "
                    "an appointment; offer to help book one.\n"
                    "- When booking: ask for doctor → date → call get_available_slots → "
                    "confirm time → call book_my_appointment.\n"
                    "- When cancelling: call get_my_appointments to get appointment IDs, "
                    "confirm which one to cancel, then call cancel_my_appointment.\n"
                    "- Never re-ask for something already given in the conversation.\n"
                    "- If the conversation shows you asked a question and the patient's current "
                    "message is a short reply, treat it as the answer to that question."
                ),
                expected_output="A concise, empathetic, patient-appropriate response.",
                agent=patient_agent,
            )

            crew = Crew(
                agents=[patient_agent],
                tasks=[task],
                process=Process.sequential,
                verbose=False,
            )
            result = crew.kickoff()
            return str(result)

        # ── Admin / doctor roles ─────────────────────────────────────────────
        doctor_agent = Agent(
            role="Doctor Agent",
            goal="Summarize patient and appointment data for clinicians.",
            backstory="You are concise and clinically structured.",
            allow_delegation=False,
            verbose=False,
        )
        admin_agent = Agent(
            role="Admin Agent",
            goal="Support clinic operations: answer scheduling questions, look up patient data, and book appointments.",
            backstory=(
                "You manage clinic operations and data visibility. "
                "When the user wants to book an appointment, collect missing details "
                "one question at a time in this order: patient name → doctor → date → "
                "call get_available_slots to show open times → confirm the chosen time → "
                "call schedule_appointment to save it. "
                "Never ask for information already provided in the conversation. "
                "IMPORTANT: If the conversation history shows you asked a question and the "
                "user's latest message is a short reply (a name, date, doctor, or time), "
                "treat it as the direct answer to your last question — do NOT ask for clarification."
            ),
            tools=ADMIN_TOOLS,
            allow_delegation=False,
            verbose=False,
        )

        task = Task(
            description=(
                f"The active user role is '{role}'.\n"
                f"Today's date is {date.today().isoformat()}.\n\n"
                f"{history_text}"
                f"Schedule data preview:\n{context}\n\n"
                f"User request: {prompt}\n\n"
                "Rules:\n"
                "- Keep answers practical and brief; avoid diagnosis.\n"
                "- For exact counts or patient lists, call the provided tools.\n"
                "- When booking an appointment, collect missing details ONE question at a time "
                "(patient → doctor → date → call get_available_slots → time → schedule_appointment).\n"
                "- Never re-ask for something already given in the conversation above.\n"
                "- If the conversation shows the assistant asked a question and the user's current "
                "message is a short reply (a name, date, time, or doctor), treat it as the direct "
                "answer to that question and proceed — do NOT ask what the reply means.\n"
                "- After scheduling, confirm with full appointment details."
            ),
            expected_output="A concise, role-appropriate answer with actionable next steps.",
            agent={
                "doctor": doctor_agent,
                "admin": admin_agent,
            }.get(role, admin_agent),
        )

        crew = Crew(
            agents=[doctor_agent, admin_agent],
            tasks=[task],
            process=Process.sequential,
            verbose=False,
        )
        result = crew.kickoff()
        return str(result)
    except Exception:
        return _fallback_response(role, prompt, context)
