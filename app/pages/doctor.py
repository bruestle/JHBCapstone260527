from __future__ import annotations

from datetime import date

import streamlit as st

from crew_agents import run_role_crew
from db import fetch_appointments_for_day, fetch_appointments_for_doctor


def _enforce_doctor_login() -> str:
    if not st.session_state.get("authenticated"):
        st.switch_page("pages/login.py")
    if st.session_state.get("role") != "doctor":
        st.error("This page is for the Doctor role.")
        st.stop()
    return st.session_state.get("user_id", "")


def _logout() -> None:
    if st.button("Log out"):
        st.session_state.authenticated = False
        st.session_state.role = ""
        st.session_state.user_id = ""
        st.switch_page("pages/login.py")


doctor_id = _enforce_doctor_login()
st.title("Doctor View")
st.caption("Review your schedule and query the Doctor Agent.")
_logout()

doctor_appts = fetch_appointments_for_doctor(doctor_id)
available_days = sorted({row["start_at"][:10] for row in doctor_appts})
if not available_days:
    available_days = [date.today().isoformat()]

selected_day = st.selectbox("Select date", options=available_days)
day_rows = [row for row in fetch_appointments_for_day(selected_day) if row["doctor_id"] == doctor_id]

st.subheader("Appointments")
appointment_table = [
    {
        "patient": row.get("patient_name", ""),
        "doctor": row.get("doctor_name", ""),
        "specialty": row.get("specialty", ""),
        "start_at": row.get("start_at", ""),
        "end_at": row.get("end_at", ""),
        "visit_type": row.get("visit_type", ""),
        "status": row.get("status", ""),
    }
    for row in day_rows
]
st.dataframe(appointment_table, use_container_width=True, hide_index=True)

st.subheader("Doctor Agent")
query = st.text_area("Ask for a summary, patient overview, or schedule insight")
if st.button("Run Doctor Agent", type="primary"):
    if not query.strip():
        st.warning("Please enter a query.")
    else:
        with st.spinner("Doctor Agent is summarizing..."):
            answer = run_role_crew("doctor", query, day_rows)
        st.write(answer)
