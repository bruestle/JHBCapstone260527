from __future__ import annotations

from datetime import date

import streamlit as st

from crew_agents import run_role_crew
from db import fetch_appointments_for_patient, fetch_medical_records, fetch_patient_list


# ── Auth helpers ──────────────────────────────────────────────────────────────

def _enforce_patient_login() -> str:
    if not st.session_state.get("authenticated"):
        st.switch_page("pages/login.py")
    if st.session_state.get("role") != "patient":
        st.error("This page is for the Patient role.")
        st.stop()
    return st.session_state.get("user_id", "")


def _get_patient_name(patient_id: str) -> str:
    for p in fetch_patient_list():
        if str(p["patient_id"]) == str(patient_id):
            return p["patient_name"]
    return patient_id


def _logout() -> None:
    st.session_state.authenticated = False
    st.session_state.role = ""
    st.session_state.user_id = ""


def _set_patient_screen(screen: str) -> None:
    st.session_state.patient_screen = screen


# ── Page setup ────────────────────────────────────────────────────────────────

patient_id = _enforce_patient_login()
patient_name = _get_patient_name(patient_id)
first_name = patient_name.split(",")[-1].strip()

st.title("Patient View")
st.caption(f"Welcome, {first_name}")

if "patient_screen" not in st.session_state:
    st.session_state.patient_screen = "assistant"

# ── Layout ────────────────────────────────────────────────────────────────────

left_col, right_col = st.columns([0.8, 5.2], gap="medium")

with left_col:
    st.markdown(
        """
        <style>
        div[data-testid="stVerticalBlock"] .st-key-patient_menu_assistant,
        div[data-testid="stVerticalBlock"] .st-key-patient_menu_appointments,
        div[data-testid="stVerticalBlock"] .st-key-patient_menu_records,
        div[data-testid="stVerticalBlock"] .st-key-patient_menu_logout {
            margin-bottom: 0.2rem;
        }

        div[data-testid="stVerticalBlock"] .st-key-patient_menu_assistant button,
        div[data-testid="stVerticalBlock"] .st-key-patient_menu_appointments button,
        div[data-testid="stVerticalBlock"] .st-key-patient_menu_records button,
        div[data-testid="stVerticalBlock"] .st-key-patient_menu_logout button {
            min-height: 1.9rem;
            padding: 0.2rem 0.5rem;
            font-size: 0.82rem;
            line-height: 1.1;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("**Menu**")
    st.button(
        "Assistant",
        key="patient_menu_assistant",
        use_container_width=True,
        type="primary" if st.session_state.patient_screen == "assistant" else "secondary",
        on_click=_set_patient_screen,
        args=("assistant",),
    )
    st.button(
        "Appointments",
        key="patient_menu_appointments",
        use_container_width=True,
        type="primary" if st.session_state.patient_screen == "appointments" else "secondary",
        on_click=_set_patient_screen,
        args=("appointments",),
    )
    st.button(
        "Medical Records",
        key="patient_menu_records",
        use_container_width=True,
        type="primary" if st.session_state.patient_screen == "records" else "secondary",
        on_click=_set_patient_screen,
        args=("records",),
    )
    st.button(
        "Log out",
        key="patient_menu_logout",
        use_container_width=True,
        on_click=_logout,
    )

# ── Right-column content ──────────────────────────────────────────────────────

with right_col:
    screen = st.session_state.patient_screen

    # ── Assistant ────────────────────────────────────────────────────────────
    if screen == "assistant":
        st.subheader("Patient Assistant")

        all_appointments = fetch_appointments_for_patient(patient_id)
        history_key = f"chat_patient_{patient_id}"
        if history_key not in st.session_state:
            st.session_state[history_key] = [
                {
                    "role": "assistant",
                    "content": (
                        f"Hi {first_name}! I can help you check your appointments, "
                        "book or cancel visits, review your medical records, or answer "
                        "general questions. How can I help you today?"
                    ),
                }
            ]

        with st.container(height=430):
            for message in st.session_state[history_key]:
                with st.chat_message(message["role"]):
                    st.write(message["content"])

        prompt = st.chat_input("Ask about your appointments, records, or health concerns...")
        if prompt:
            st.session_state[history_key].append({"role": "user", "content": prompt})
            with st.spinner("Patient Assistant is thinking..."):
                reply = run_role_crew(
                    "patient",
                    prompt,
                    all_appointments,
                    chat_history=st.session_state[history_key],
                    patient_id=patient_id,
                    patient_name=patient_name,
                )
            st.session_state[history_key].append({"role": "assistant", "content": reply})
            st.rerun()

    # ── Appointments ─────────────────────────────────────────────────────────
    elif screen == "appointments":
        st.subheader("My Appointments")

        all_appointments = fetch_appointments_for_patient(patient_id)
        today_str = date.today().isoformat()
        upcoming = [r for r in all_appointments if r.get("start_at", "") >= today_str]
        past = [r for r in all_appointments if r.get("start_at", "") < today_str]

        st.markdown("**Upcoming**")
        if upcoming:
            st.dataframe(
                [
                    {
                        "Date / Time": r.get("start_at", ""),
                        "Doctor": r.get("doctor_name", ""),
                        "Specialty": r.get("specialty", ""),
                        "Type": r.get("visit_type", ""),
                        "Status": r.get("status", ""),
                    }
                    for r in upcoming
                ],
                use_container_width=True,
                hide_index=True,
            )
        else:
            st.info("No upcoming appointments.")

        st.markdown("**Past**")
        if past:
            st.dataframe(
                [
                    {
                        "Date / Time": r.get("start_at", ""),
                        "Doctor": r.get("doctor_name", ""),
                        "Specialty": r.get("specialty", ""),
                        "Type": r.get("visit_type", ""),
                        "Status": r.get("status", ""),
                    }
                    for r in reversed(past)
                ],
                use_container_width=True,
                hide_index=True,
            )
        else:
            st.info("No past appointments on record.")

    # ── Medical Records ───────────────────────────────────────────────────────
    elif screen == "records":
        st.subheader("My Medical Records")

        records = fetch_medical_records(patient_id)
        if not records:
            st.info("No medical records on file.")
        else:
            from html import escape as _esc

            rows_html = ""
            for rec in records:
                created    = rec.get("created_at") or ""
                doctor_val = rec.get("doctor_name") or "—"
                etype_val  = rec.get("encounter_type") or "OfficeVisit"
                note_val   = (rec.get("note") or "").replace("\n", "<br>")
                rows_html += (
                    f"<tr>"
                    f"<td class='rc nowrap'>{_esc(created)}</td>"
                    f"<td class='rc nowrap'>{_esc(doctor_val)}</td>"
                    f"<td class='rc nowrap'>{_esc(etype_val)}</td>"
                    f"<td class='rc wrap'>{note_val}</td>"
                    f"</tr>"
                )

            st.markdown(
                f"""
                <style>
                .rec-table-wrap {{
                    overflow-x: auto;
                    border: 1px solid rgba(255,255,255,0.2);
                    border-radius: 6px;
                }}
                .rec-table {{
                    width: 100%;
                    border-collapse: collapse;
                    font-size: 0.85rem;
                }}
                .rec-table th {{
                    text-align: left;
                    padding: 0.35rem 0.5rem;
                    border-bottom: 1px solid rgba(255,255,255,0.2);
                    white-space: nowrap;
                }}
                .rc {{
                    padding: 0.35rem 0.5rem;
                    border-bottom: 1px solid rgba(255,255,255,0.1);
                    vertical-align: top;
                }}
                .rc.nowrap {{ white-space: nowrap; }}
                .rc.wrap   {{ white-space: normal; overflow-wrap: anywhere; min-width: 28rem; }}
                .rec-table tr:last-child .rc {{ border-bottom: none; }}
                </style>
                <div class="rec-table-wrap">
                  <table class="rec-table">
                    <thead>
                      <tr>
                        <th>Date / Time</th>
                        <th>Doctor</th>
                        <th>Event Type</th>
                        <th>Notes</th>
                      </tr>
                    </thead>
                    <tbody>{rows_html}</tbody>
                  </table>
                </div>
                """,
                unsafe_allow_html=True,
            )
