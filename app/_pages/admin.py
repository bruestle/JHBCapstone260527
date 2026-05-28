from __future__ import annotations

from datetime import date, datetime, timedelta
from html import escape

import altair as alt
import pandas as pd
import streamlit as st
import re

from crew_agents import run_role_crew, CREWAI_AVAILABLE
from db import (
    DOCTOR_SEED,
    fetch_all_appointments,
    fetch_appointments_for_day,
    fetch_appointments_for_patient,
    fetch_medical_records,
    fetch_patient_list,
    search_patient_history_semantic,
    seed_mock_data,
    sync_patient_history_vectors,
    update_patient_contact_info,
)


def _enforce_admin_login() -> None:
    if not st.session_state.get("authenticated"):
        st.switch_page("pages/login.py")
    if st.session_state.get("role") != "admin":
        st.error("This page is for the Admin role.")
        st.stop()


def _logout() -> None:
    st.session_state.authenticated = False
    st.session_state.role = ""
    st.session_state.user_id = ""


def _set_admin_screen(screen: str) -> None:
    st.session_state.admin_screen = screen


def _next_weekday(day_value: date, step: int) -> date:
    current = day_value
    while True:
        current = current + timedelta(days=step)
        if current.weekday() < 5:
            return current


def _normalize_weekday(day_value: date) -> date:
    current = day_value
    while current.weekday() >= 5:
        current += timedelta(days=1)
    return current


def _doctor_name_domain() -> list[str]:
    return [f"{last_name}, {first_name}" for _, first_name, last_name, _ in DOCTOR_SEED]


def _render_wrapped_history_table(columns: list[str], rows: list[dict], wrap_columns: set[str]) -> None:
    if not rows:
        return

    table_rows: list[str] = []
    for row in rows:
        cells: list[str] = []
        for col in columns:
            value = escape(str(row.get(col, "") or "")).replace("\n", "<br>")
            wrap_class = " wrap" if col in wrap_columns else ""
            cells.append(f'<td class="history-cell{wrap_class}">{value}</td>')
        table_rows.append(f"<tr>{''.join(cells)}</tr>")

    header_html = "".join(f'<th class="history-head">{escape(col)}</th>' for col in columns)
    body_html = "".join(table_rows)

    st.markdown(
        f"""
        <style>
        .history-table-wrap {{
            overflow-x: auto;
            border: 1px solid rgba(255, 255, 255, 0.2);
            border-radius: 6px;
        }}
        .history-table {{
            width: 100%;
            border-collapse: collapse;
            font-size: 0.85rem;
        }}
        .history-head {{
            text-align: left;
            padding: 0.35rem 0.45rem;
            border-bottom: 1px solid rgba(255, 255, 255, 0.2);
            white-space: nowrap;
        }}
        .history-cell {{
            padding: 0.35rem 0.45rem;
            border-bottom: 1px solid rgba(255, 255, 255, 0.12);
            white-space: nowrap;
            vertical-align: top;
        }}
        .history-cell.wrap {{
            white-space: normal;
            overflow-wrap: anywhere;
            min-width: 26rem;
        }}
        .history-table tr:last-child .history-cell {{
            border-bottom: none;
        }}
        </style>
        <div class="history-table-wrap">
            <table class="history-table">
                <thead><tr>{header_html}</tr></thead>
                <tbody>{body_html}</tbody>
            </table>
        </div>
        """,
        unsafe_allow_html=True,
    )


@st.dialog("Edit Patient")
def _patient_edit_dialog() -> None:
    patient = st.session_state.get("admin_edit_patient")
    if not patient:
        return

    st.write(f"Patient ID: {patient['patient_id']}")
    st.write(f"Name: {patient['patient_name']}")

    telephone = st.text_input("Telephone", value=patient.get("telephone") or "")
    address = st.text_input("Address", value=patient.get("address") or "")
    insurance = st.text_input("Insurance carrier", value=patient.get("insurance_carrier") or "")

    action_col_1, action_col_2 = st.columns(2)
    with action_col_1:
        if st.button("Save", type="primary", key="admin_save_patient_edit"):
            update_patient_contact_info(patient["patient_id"], telephone, address, insurance)
            st.session_state.admin_edit_patient = None
            st.rerun()
    with action_col_2:
        if st.button("Cancel", key="admin_cancel_patient_edit"):
            st.session_state.admin_edit_patient = None
            st.rerun()


@st.dialog("Patient History")
def _patient_history_dialog() -> None:
    patient = st.session_state.get("admin_history_patient")
    if not patient:
        return

    st.markdown(
        """
        <style>
        div[data-testid="stDialog"] div[role="dialog"] {
            width: min(1100px, 96vw) !important;
            max-width: min(1100px, 96vw) !important;
            background: #111111 !important;
        }
        div[data-testid="stDialog"] div[role="dialog"] > div {
            max-height: 88vh !important;
            background: #111111 !important;
        }
        div[data-testid="stDialog"] div[role="dialog"] * {
            background-color: transparent;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    patient_id = patient["patient_id"]
    st.write(f"Patient ID: {patient_id}")
    st.write(f"Name: {patient['patient_name']}")

    with st.container(height=560):
        st.markdown("**Semantic Search**")
        semantic_query = st.text_input(
            "Search this patient's history",
            key=f"admin_history_search_query_{patient_id}",
            placeholder="e.g., blood pressure trend, medication changes, respiratory symptoms",
        )
        if semantic_query.strip():
            semantic_rows = search_patient_history_semantic(patient_id=patient_id, query=semantic_query, top_k=8)
            if semantic_rows:
                search_results = [
                    {
                        "Created At": row["created_at"],
                        "Record ID": row["record_id"],
                        "Score": f"{row['similarity']:.3f}" if row.get("similarity") is not None else "",
                        "Snippet": row["text"],
                    }
                    for row in semantic_rows
                ]
                _render_wrapped_history_table(
                    columns=["Created At", "Record ID", "Score", "Snippet"],
                    rows=search_results,
                    wrap_columns={"Snippet"},
                )
            else:
                st.caption("No semantic matches found for this patient yet.")

        st.markdown("**Medical Records**")
        records = fetch_medical_records(str(patient_id))
        if records:
            record_rows = [
                {
                    "Created At": record["created_at"],
                    "Doctor": record.get("doctor_name") or "—",
                    "Encounter Type": record.get("encounter_type") or "OfficeVisit",
                    "Note": record["note"],
                }
                for record in records
            ]
            _render_wrapped_history_table(
                columns=["Created At", "Doctor", "Encounter Type", "Note"],
                rows=record_rows,
                wrap_columns={"Note"},
            )
        else:
            st.caption("No medical records found.")

        st.markdown("**Appointment History**")
        appointments = fetch_appointments_for_patient(patient_id)
        if appointments:
            appointment_rows = [
                {
                    "Start": row["start_at"],
                    "Doctor": row["doctor_name"],
                    "Specialty": row["specialty"],
                    "Visit": row["visit_type"],
                    "Status": row["status"],
                    "Notes": row["notes"],
                }
                for row in appointments
            ]
            _render_wrapped_history_table(
                columns=["Start", "Doctor", "Specialty", "Visit", "Status", "Notes"],
                rows=appointment_rows,
                wrap_columns={"Notes"},
            )
        else:
            st.caption("No appointments found.")

    if st.button("Close", key="admin_close_patient_history"):
        st.session_state.admin_history_patient = None
        st.rerun()


def _render_schedule_gantt(rows: list[dict], selected_day: date) -> None:
    doctor_domain = _doctor_name_domain()

    def _fmt_time(dt: datetime) -> str:
        h = dt.hour % 12 or 12
        ampm = "AM" if dt.hour < 12 else "PM"
        return f"{h}:{dt.minute:02d} {ampm}"

    def _to_hours(dt: datetime) -> float:
        return dt.hour + dt.minute / 60.0

    gantt_rows = []
    for row in rows:
        start_dt = datetime.strptime(row["start_at"], "%Y-%m-%d %H:%M")
        end_dt = datetime.strptime(row["end_at"], "%Y-%m-%d %H:%M")
        gantt_rows.append(
            {
                "doctor_name": row["doctor_name"],
                "patient_name": row["patient_name"],
                "visit_type": row["visit_type"],
                "status": row["status"],
                "start_hr": _to_hours(start_dt),
                "end_hr": _to_hours(end_dt),
                "start_label": _fmt_time(start_dt),
                "end_label": _fmt_time(end_dt),
            }
        )

    chart_df = pd.DataFrame(gantt_rows)

    _hour_domain = [8.0, 17.0]
    # Pure JS math — no date/timezone objects involved
    _hour_axis = alt.Axis(
        values=list(range(8, 18)),
        labelExpr=(
            "datum.value < 12 ? (datum.value + ' AM') : "
            "(datum.value === 12 ? '12 PM' : (datum.value - 12) + ' PM')"
        ),
    )

    base = alt.Chart(pd.DataFrame({"doctor_name": doctor_domain})).mark_bar(opacity=0).encode(
        y=alt.Y("doctor_name:N", sort=doctor_domain, title="Doctor")
    )

    hour_marks = pd.DataFrame({"hour": list(range(8, 18))})

    hour_lines = (
        alt.Chart(hour_marks)
        .mark_rule(color="white", opacity=0.65)
        .encode(
            x=alt.X("hour:Q", scale=alt.Scale(domain=_hour_domain), axis=_hour_axis),
        )
    )

    bars = (
        alt.Chart(chart_df)
        .mark_bar(cornerRadius=3)
        .encode(
            y=alt.Y("doctor_name:N", sort=doctor_domain, title="Doctor"),
            x=alt.X("start_hr:Q", title="Time", scale=alt.Scale(domain=_hour_domain), axis=_hour_axis),
            x2="end_hr:Q",
            color=alt.Color(
                "visit_type:N",
                legend=alt.Legend(title="Visit Type", orient="top", direction="horizontal"),
            ),
            tooltip=[
                alt.Tooltip("doctor_name:N", title="Doctor"),
                alt.Tooltip("patient_name:N", title="Patient"),
                alt.Tooltip("start_label:N", title="Start"),
                alt.Tooltip("end_label:N", title="End"),
                alt.Tooltip("visit_type:N", title="Visit"),
                alt.Tooltip("status:N", title="Status"),
            ],
        )
    )

    labels = (
        alt.Chart(chart_df)
        .mark_text(align="left", baseline="middle", dx=3, fontSize=12, color="white")
        .encode(
            y=alt.Y("doctor_name:N", sort=doctor_domain, title="Doctor"),
            x=alt.X("start_hr:Q", scale=alt.Scale(domain=_hour_domain), axis=_hour_axis),
            text=alt.Text("patient_name:N"),
        )
    )

    chart = (
        (base + hour_lines + bars + labels)
        .properties(height=360)
        .configure_view(fill="#3a3a3a", strokeOpacity=0)
        .configure_axis(labelColor="white", titleColor="white", tickColor="white", domainColor="white", grid=False)
        .configure_legend(labelColor="white", titleColor="white")
    )

    st.altair_chart(chart, use_container_width=True)


_enforce_admin_login()
st.title("Admin Console")
st.caption("Use the menu on the left to switch admin screens.")

if "admin_screen" not in st.session_state:
    st.session_state.admin_screen = "assistant"

# Approximate a narrower menu rail on common desktop widths.
left_col, right_col = st.columns([0.8, 5.2], gap="medium")

with left_col:
    st.markdown(
        """
        <style>
        div[data-testid="stVerticalBlock"] .st-key-admin_menu_assistant,
        div[data-testid="stVerticalBlock"] .st-key-admin_menu_today_schedule,
        div[data-testid="stVerticalBlock"] .st-key-admin_menu_patient_list,
        div[data-testid="stVerticalBlock"] .st-key-admin_menu_database_reset,
        div[data-testid="stVerticalBlock"] .st-key-admin_menu_logout {
            margin-bottom: 0.2rem;
        }

        div[data-testid="stVerticalBlock"] .st-key-admin_menu_assistant button,
        div[data-testid="stVerticalBlock"] .st-key-admin_menu_today_schedule button,
        div[data-testid="stVerticalBlock"] .st-key-admin_menu_patient_list button,
        div[data-testid="stVerticalBlock"] .st-key-admin_menu_database_reset button,
        div[data-testid="stVerticalBlock"] .st-key-admin_menu_logout button {
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
        key="admin_menu_assistant",
        use_container_width=True,
        type="primary" if st.session_state.admin_screen == "assistant" else "secondary",
        on_click=_set_admin_screen,
        args=("assistant",),
    )
    st.button(
        "Today's Schedule",
        key="admin_menu_today_schedule",
        use_container_width=True,
        type="primary" if st.session_state.admin_screen == "today_schedule" else "secondary",
        on_click=_set_admin_screen,
        args=("today_schedule",),
    )
    st.button(
        "Patient List",
        key="admin_menu_patient_list",
        use_container_width=True,
        type="primary" if st.session_state.admin_screen == "patient_list" else "secondary",
        on_click=_set_admin_screen,
        args=("patient_list",),
    )
    st.button(
        "Database Reset",
        key="admin_menu_database_reset",
        use_container_width=True,
        type="primary" if st.session_state.admin_screen == "database_reset" else "secondary",
        on_click=_set_admin_screen,
        args=("database_reset",),
    )
    st.button(
        "Log out",
        key="admin_menu_logout",
        use_container_width=True,
        on_click=_logout,
    )
    st.divider()
    if CREWAI_AVAILABLE:
        st.caption("🟢 LLM available")
    else:
        st.caption("🔴 LLM unavailable")

with right_col:
    # counts = fetch_counts()
    # c1, c2, c3 = st.columns(3)
    # c1.metric("Patients", counts["patients"])
    # c2.metric("Doctors", counts["doctors"])
    # c3.metric("Appointments", counts["appointments"])

    screen = st.session_state.admin_screen

    if screen == "assistant":
        st.subheader("Admin Assistant")
        all_rows = fetch_all_appointments()

        history_key = "chat_admin_assistant"
        if history_key not in st.session_state:
            st.session_state[history_key] = [
                {
                    "role": "assistant",
                    "content": "Hi, I can help with operations, schedule utilization, and data-management questions.",
                }
            ]

        with st.container(height=430):
            for message in st.session_state[history_key]:
                with st.chat_message(message["role"]):
                    st.write(message["content"])

        prompt = st.chat_input("Ask about operations, utilization, or data-management tasks")
        if prompt:
            st.session_state[history_key].append({"role": "user", "content": prompt})

            with st.spinner("Admin Agent is working..."):
                result = run_role_crew("admin", prompt, all_rows, chat_history=st.session_state[history_key])

            st.session_state[history_key].append({"role": "assistant", "content": result})
            st.rerun()

    elif screen == "today_schedule":
        st.subheader("Today's Schedule")

        if "admin_schedule_day" not in st.session_state:
            st.session_state.admin_schedule_day = _normalize_weekday(date.today())

        nav_left, nav_mid, nav_right = st.columns([1, 3, 1])
        with nav_left:
            if st.button("◀", key="admin_schedule_prev"):
                st.session_state.admin_schedule_day = _next_weekday(st.session_state.admin_schedule_day, -1)
        with nav_mid:
            selected_day = st.session_state.admin_schedule_day
            st.markdown(f"**{selected_day.strftime('%A, %Y-%m-%d')}**")
        with nav_right:
            if st.button("▶", key="admin_schedule_next"):
                st.session_state.admin_schedule_day = _next_weekday(st.session_state.admin_schedule_day, 1)

        selected_day = st.session_state.admin_schedule_day
        with st.spinner("Loading schedule..."):
            today_rows = fetch_appointments_for_day(selected_day.isoformat())
            _render_schedule_gantt(today_rows, selected_day)

    elif screen == "patient_list":
        st.subheader("Patient List")
        patients = fetch_patient_list()

        st.markdown(
            """
            <style>
            .st-key-admin_patient_table_body {
                background: #222222;
                border: none;
                border-radius: 0;
                padding-top: 0;
            }

            .admin-patient-table-cell {
                color: #f3f3f3;
                border: none;
                padding: 0;
                line-height: 1.1;
                font-size: 0.82rem;
                white-space: nowrap;
                min-height: 0.9rem;
                display: flex;
                align-items: center;
            }

            .admin-patient-table-head {
                background: #222222;
                color: #ffffff;
                border: none;
                padding: 0;
                line-height: 1.1;
                font-size: 0.82rem;
                font-weight: 600;
                min-height: 0.9rem;
                display: flex;
                align-items: center;
            }

            .admin-patient-table-last {
                border-right: none;
            }

            [class*="st-key-admin_patient_edit_"] button,
            [class*="st-key-admin_patient_history_"] button {
                background: transparent;
                color: #e6e6e6;
                border: none;
                padding: 0;
                min-height: 0.9rem;
                font-size: 0.82rem;
                line-height: 1.1;
                text-decoration: underline;
                text-underline-offset: 2px;
                border-radius: 0;
                width: 100%;
            }

            [class*="st-key-admin_patient_edit_"] button:hover,
            [class*="st-key-admin_patient_history_"] button:hover {
                color: #ffffff;
                background: transparent;
            }

            [class*="st-key-admin_patient_edit_header"] button {
                background: #222222;
                color: #ffffff;
                border: none;
                border-radius: 0;
                padding: 0;
                min-height: 0.9rem;
                font-size: 0.82rem;
                font-weight: 600;
                width: 100%;
                pointer-events: none;
            }

            [class*="st-key-admin_patient_row_"] {
                margin-bottom: 0;
            }
            </style>
            """,
            unsafe_allow_html=True,
        )

        header_cols = st.columns([1.0, 2.6, 1.9, 1.9, 0.8, 0.95], gap="small")
        header_cols[0].markdown('<div class="admin-patient-table-head">ID</div>', unsafe_allow_html=True)
        header_cols[1].markdown('<div class="admin-patient-table-head">Name</div>', unsafe_allow_html=True)
        header_cols[2].markdown('<div class="admin-patient-table-head">Telephone</div>', unsafe_allow_html=True)
        header_cols[3].markdown('<div class="admin-patient-table-head">Insurance</div>', unsafe_allow_html=True)
        header_cols[4].markdown('<div class="admin-patient-table-head">Edit</div>', unsafe_allow_html=True)
        header_cols[5].markdown('<div class="admin-patient-table-head">History</div>', unsafe_allow_html=True)

        with st.container(key="admin_patient_table_body"):
            for patient in patients:
                row_cols = st.columns([1.0, 2.6, 1.9, 1.9, 0.8, 0.95], gap="small")
                row_cols[0].markdown(
                    f'<div class="admin-patient-table-cell">{patient["patient_id"]}</div>',
                    unsafe_allow_html=True,
                )
                row_cols[1].markdown(
                    f'<div class="admin-patient-table-cell">{patient.get("patient_name") or ""}</div>',
                    unsafe_allow_html=True,
                )
                row_cols[2].markdown(
                    f'<div class="admin-patient-table-cell">{patient.get("telephone") or ""}</div>',
                    unsafe_allow_html=True,
                )
                row_cols[3].markdown(
                    f'<div class="admin-patient-table-cell">{patient.get("insurance_carrier") or ""}</div>',
                    unsafe_allow_html=True,
                )

                if row_cols[4].button(
                    "Edit",
                    key=f"admin_patient_edit_{patient['patient_id']}",
                    use_container_width=True,
                ):
                    st.session_state.admin_edit_patient = patient

                if row_cols[5].button(
                    "History",
                    key=f"admin_patient_history_{patient['patient_id']}",
                    use_container_width=True,
                ):
                    st.session_state.admin_history_patient = patient

        if st.session_state.get("admin_edit_patient"):
            _patient_edit_dialog()

        if st.session_state.get("admin_history_patient"):
            _patient_history_dialog()

    elif screen == "database_reset":
        st.subheader("Database Reset")
        st.warning("This will clear and reseed all demo data.")
        if st.button("Reseed Database", type="primary"):
            progress_bar = st.progress(0, text="Preparing reseed...")
            progress_text = st.empty()
            progress_state = {"value": 0}

            def _set_progress(value: int, message: str) -> None:
                clamped_value = max(0, min(100, value))
                if clamped_value != progress_state["value"]:
                    progress_state["value"] = clamped_value
                    progress_bar.progress(clamped_value, text=message)
                progress_text.caption(message)

            def _handle_seed_progress(message: str) -> None:
                if message.startswith("Step: initializing"):
                    _set_progress(2, message)
                elif message.startswith("Step: reset requested"):
                    _set_progress(5, message)
                elif message.startswith("Sub-step: cleared appointments"):
                    _set_progress(8, message)
                elif message.startswith("Sub-step: cleared medical_records"):
                    _set_progress(10, message)
                elif message.startswith("Sub-step: cleared medical_record vectors"):
                    _set_progress(11, message)
                elif message.startswith("Sub-step: cleared doctors"):
                    _set_progress(12, message)
                elif message.startswith("Sub-step: cleared patients"):
                    _set_progress(14, message)
                elif message.startswith("Sub-step: cleared seed metadata"):
                    _set_progress(16, message)
                elif message.startswith("Step: seeding patients"):
                    _set_progress(20, message)
                elif message.startswith("Sub-step: upserted") and "patient row" in message:
                    _set_progress(28, message)
                elif message.startswith("Step: seeding doctors"):
                    _set_progress(32, message)
                elif message.startswith("Sub-step: upserted") and "doctor row" in message:
                    _set_progress(40, message)
                elif message.startswith("Step: generating and seeding appointments"):
                    _set_progress(45, message)
                elif message.startswith("Step: seeding medical records"):
                    _set_progress(58, message)
                elif message.startswith("Step: preparing LLM medical records"):
                    _set_progress(60, message)
                elif message.startswith("Step: indexing medical records in vector store"):
                    _set_progress(95, message)
                else:
                    patient_match = re.search(r"patient (\d+)/(\d+)", message)
                    completed_match = re.search(r"completed patient (\d+)/(\d+)", message)
                    vector_index_match = re.search(r"vector indexing (\d+)/(\d+)", message)
                    if completed_match:
                        current = int(completed_match.group(1))
                        total = int(completed_match.group(2))
                        value = 62 + int((current / total) * 36)
                        _set_progress(value, message)
                    elif vector_index_match:
                        current = int(vector_index_match.group(1))
                        total = int(vector_index_match.group(2))
                        value = 95 + int((current / total) * 4)
                        _set_progress(value, message)
                    elif patient_match:
                        current = int(patient_match.group(1))
                        total = int(patient_match.group(2))
                        value = 60 + int((current / total) * 32)
                        _set_progress(value, message)
                    elif message.startswith("Sub-step: sending LLM request"):
                        _set_progress(max(progress_state["value"], 61), message)
                    elif message.startswith("Sub-step: LLM attempt") and "failed" in message:
                        progress_text.caption(message)
                    elif message.startswith("Step: seeding complete"):
                        _set_progress(100, message)
                    else:
                        progress_text.caption(message)

            with st.status("Reseeding database...", expanded=True):
                seed_mock_data(reset=True, seed_start=date.today(), progress_cb=_handle_seed_progress)
                progress_bar.progress(100, text="Reseeding complete.")
                progress_text.caption("Reseeding complete.")
            st.success("Database reseeded.")
            st.rerun()

        st.divider()
        st.markdown("**Sync Vector Store** — re-index existing patient records into the vector database without reseeding.")
        if st.button("Sync Vector Store", type="secondary"):
            sync_bar = st.progress(0, text="Preparing sync...")
            sync_text = st.empty()
            sync_state = {"value": 0}

            def _set_sync_progress(value: int, message: str) -> None:
                clamped_value = max(0, min(100, value))
                if clamped_value != sync_state["value"]:
                    sync_state["value"] = clamped_value
                    sync_bar.progress(clamped_value, text=message)
                sync_text.caption(message)

            def _handle_sync_progress(message: str) -> None:
                if "clearing existing" in message:
                    _set_sync_progress(5, message)
                elif "indexing" in message and "vector store" in message:
                    _set_sync_progress(10, message)
                elif "vector sync complete" in message:
                    _set_sync_progress(100, message)
                else:
                    vector_index_match = re.search(r"vector indexing (\d+)/(\d+)", message)
                    if vector_index_match:
                        current = int(vector_index_match.group(1))
                        total = int(vector_index_match.group(2))
                        value = 10 + int((current / total) * 88)
                        _set_sync_progress(value, message)
                    else:
                        sync_text.caption(message)

            with st.status("Syncing vector store...", expanded=True):
                sync_patient_history_vectors(progress_cb=_handle_sync_progress)
                sync_bar.progress(100, text="Sync complete.")
                sync_text.caption("Sync complete.")
            st.success("Vector store synced.")
