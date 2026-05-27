from __future__ import annotations

import sys
from pathlib import Path

# Ensure workspace root is on sys.path when Streamlit runs from app/.
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
	sys.path.insert(0, str(ROOT_DIR))

from database.db import (  # noqa: E402
	clear_patient_history_vectors,
	DOCTOR_SEED,
	PATIENT_SEED,
	cancel_appointment,
	connection,
	create_appointment,
	fetch_all_appointments,
	fetch_appointments_for_day,
	fetch_appointments_for_doctor,
	fetch_appointments_for_patient,
	fetch_available_slots,
	fetch_counts,
	fetch_doctor_day_patients_with_keyword,
	fetch_doctor_list,
	fetch_doctor_patients_with_keyword,
	fetch_patient_list,
	fetch_medical_records,
	fetch_patient_roster,
	index_medical_record_rows,
	init_db,
	is_db_seeded,
	search_all_patients_history_semantic,
	search_patient_history_semantic,
	search_patients_by_record_keyword,
	seed_mock_data,
	sync_patient_history_vectors,
	update_patient_contact_info,
)

__all__ = [
	"DOCTOR_SEED",
	"PATIENT_SEED",
	"cancel_appointment",
	"clear_patient_history_vectors",
	"connection",
	"create_appointment",
	"fetch_all_appointments",
	"fetch_appointments_for_day",
	"fetch_appointments_for_doctor",
	"fetch_appointments_for_patient",
	"fetch_available_slots",
	"fetch_counts",
	"fetch_doctor_day_patients_with_keyword",
	"fetch_doctor_list",
	"fetch_doctor_patients_with_keyword",
	"fetch_patient_list",
	"fetch_medical_records",
	"fetch_patient_roster",
	"index_medical_record_rows",
	"init_db",
	"is_db_seeded",
	"search_all_patients_history_semantic",
	"search_patient_history_semantic",
	"search_patients_by_record_keyword",
	"seed_mock_data",
	"sync_patient_history_vectors",
	"update_patient_contact_info",
]
