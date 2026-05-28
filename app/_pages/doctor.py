from __future__ import annotations

from datetime import date, timedelta

import streamlit as st

from crew_agents import run_role_crew
from db import (
    ENCOUNTER_TYPES,
    add_medical_record,
    delete_medical_record,
    fetch_appointments_for_doctor,
    fetch_appointments_for_day,
    fetch_medical_records,
    fetch_patient_list,
    update_medical_record,
)


# ── Auth helpers ──────────────────────────────────────────────────────────────

def _enforce_doctor_login() -> tuple[str, str]:
    if not st.session_state.get("authenticated"):
        st.switch_page("_pages/login.py")
    if st.session_state.get("role") != "doctor":
        st.error("This page is for the Doctor role.")
        st.stop()
    return st.session_state.get("user_id", ""), st.session_state.get("user_name", "")


def _logout() -> None:
    st.session_state.authenticated = False
    st.session_state.role = ""
    st.session_state.user_id = ""
    st.session_state.user_name = ""


def _set_screen(screen: str) -> None:
    st.session_state.doctor_screen = screen


def _prev_weekday(d: date) -> date:
    d -= timedelta(days=1)
    while d.weekday() >= 5:
        d -= timedelta(days=1)
    return d


def _next_weekday(d: date) -> date:
    d += timedelta(days=1)
    while d.weekday() >= 5:
        d += timedelta(days=1)
    return d


def _nearest_weekday(d: date) -> date:
    """Return d unchanged if it is Mon–Fri; otherwise advance to next Monday."""
    if d.weekday() == 5:   # Saturday
        return d + timedelta(days=2)
    if d.weekday() == 6:   # Sunday
        return d + timedelta(days=1)
    return d


def _render_gantt(appointments: list[dict]) -> str:
    """Return self-contained HTML for a vertical Gantt chart (7 AM – 7 PM)."""
    from html import escape as _esc

    START_H, END_H = 7, 19
    PX_PER_MIN: float = 1.2
    TOTAL_MIN = (END_H - START_H) * 60      # 720
    TOTAL_PX = int(TOTAL_MIN * PX_PER_MIN)  # 864

    def _colors(visit_type: str, cancelled: bool) -> tuple[str, str]:
        if cancelled:
            return "#1a1a24", "#3a3a4a"
        v = (visit_type or "").lower()
        if any(w in v for w in ("urgent", "emergenc")):
            return "#3a0f0f", "#ef4444"
        if any(w in v for w in ("procedure", "surgery", "operat")):
            return "#3a2408", "#f59e0b"
        if any(w in v for w in ("lab", "imaging", "radiol", "xray")):
            return "#1a2a3a", "#60a5fa"
        if "telehealth" in v or "virtual" in v:
            return "#1e0e40", "#8b5cf6"
        if any(w in v for w in ("wellness", "annual", "physical", "preventive", "follow")):
            return "#0b2e24", "#10b981"
        return "#0d1f3a", "#3b82f6"

    # Time-axis labels and grid lines
    axis_parts: list[str] = []
    grid_parts: list[str] = []
    for h in range(START_H, END_H + 1):
        off = int((h - START_H) * 60 * PX_PER_MIN)
        lbl = f"{h % 12 or 12}&thinsp;{'am' if h < 12 else 'pm'}"
        axis_parts.append(f'<div class="gl-tl" style="top:{off}px">{lbl}</div>')
        grid_parts.append(f'<div class="gl-hr" style="top:{off}px"></div>')
        if h < END_H:
            half_off = off + int(30 * PX_PER_MIN)
            grid_parts.append(f'<div class="gl-hf" style="top:{half_off}px"></div>')

    # Appointment blocks
    block_parts: list[str] = []
    for appt in appointments:
        raw = appt.get("start_at", "")
        try:
            _, t = raw.split(" ", 1)
            ah, am_v = map(int, t.split(":")[:2])
        except (ValueError, AttributeError):
            continue
        start_min = (ah - START_H) * 60 + am_v
        if not (0 <= start_min < TOTAL_MIN):
            continue
        dur = max(int(appt.get("duration_min") or 30), 15)
        end_min = min(start_min + dur, TOTAL_MIN)
        top_px = int(start_min * PX_PER_MIN)
        blk_h = max(int((end_min - start_min) * PX_PER_MIN), 40)
        cancelled = appt.get("status") == "cancelled"
        patient = _esc(appt.get("patient_name") or "Unknown")
        vtype_raw = appt.get("visit_type") or ""
        vtype = _esc(vtype_raw)
        bg, border = _colors(vtype_raw, cancelled)
        fade = "opacity:0.38;" if cancelled else ""
        vline = f'<div class="gl-vt">{vtype}</div>' if blk_h >= 52 and vtype else ""
        block_parts.append(
            f'<div class="gl-blk" style="top:{top_px}px;height:{blk_h}px;'
            f'background:{bg};border-left:3px solid {border};{fade}">'
            f'<div class="gl-ts">{t}</div>'
            f'<div class="gl-pt">{patient}</div>'
            f'{vline}</div>'
        )

    empty_overlay = "" if appointments else (
        '<div style="position:absolute;top:50%;left:50%;transform:translate(-50%,-50%);'
        'color:#2e2e4e;font-size:1rem;font-weight:500;pointer-events:none">'
        'No appointments scheduled</div>'
    )

    axis_html   = "".join(axis_parts)
    grid_html   = "".join(grid_parts)
    blocks_html = "".join(block_parts)
    scroll_top  = int(60 * PX_PER_MIN)  # auto-scroll to show 8 AM on load

    # Static CSS — plain string, no f-string escaping needed
    css_static = (
        "*{box-sizing:border-box;margin:0;padding:0}"
        "body{background:#0e1117;font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif}"
        "#glwrap{display:flex;height:510px;overflow-y:auto;background:#0e1117}"
        "#glwrap::-webkit-scrollbar{width:5px}"
        "#glwrap::-webkit-scrollbar-track{background:#0e1117}"
        "#glwrap::-webkit-scrollbar-thumb{background:#2a2a3a;border-radius:3px}"
        ".gl-tl{position:absolute;right:6px;font-size:0.62rem;color:#4a4a6a;"
        "transform:translateY(-50%);user-select:none}"
        ".gl-hr{position:absolute;left:0;right:0;border-top:1px solid #1a1a2a}"
        ".gl-hf{position:absolute;left:0;right:0;border-top:1px dashed #141420}"
        ".gl-blk{position:absolute;left:6px;right:6px;border-radius:6px;padding:5px 9px;"
        "overflow:hidden;transition:filter .15s}"
        ".gl-blk:hover{filter:brightness(1.3)}"
        ".gl-ts{font-size:0.6rem;color:rgba(255,255,255,0.5);margin-bottom:1px}"
        ".gl-pt{font-size:0.8rem;font-weight:600;color:#e2e8f0;white-space:nowrap;"
        "overflow:hidden;text-overflow:ellipsis}"
        ".gl-vt{font-size:0.68rem;color:rgba(255,255,255,0.55);white-space:nowrap;"
        "overflow:hidden;text-overflow:ellipsis;margin-top:2px}"
    )
    # Dynamic CSS — f-string, uses TOTAL_PX
    css_dynamic = (
        f".gl-axis{{width:52px;flex-shrink:0;position:relative;"
        f"height:{TOTAL_PX}px;border-right:1px solid #1e1e2e}}"
        f".gl-body{{flex:1;position:relative;height:{TOTAL_PX}px}}"
    )

    return (
        "<!DOCTYPE html><html><head><meta charset='utf-8'>"
        f"<style>{css_static}{css_dynamic}</style></head><body>"
        f'<div id="glwrap">'
        f'<div class="gl-axis">{axis_html}</div>'
        f'<div class="gl-body">{grid_html}{blocks_html}{empty_overlay}</div>'
        f'</div>'
        f"<script>document.getElementById('glwrap').scrollTop={scroll_top};</script>"
        f"</body></html>"
    )


# ── Page setup ────────────────────────────────────────────────────────────────

doctor_id, doctor_name = _enforce_doctor_login()
first_name = doctor_name.split(",")[-1].strip() if doctor_name else "Doctor"

st.title("Doctor View")
st.caption(f"Welcome, Dr. {first_name}")

if "doctor_screen" not in st.session_state:
    st.session_state.doctor_screen = "assistant"

# ── Layout ────────────────────────────────────────────────────────────────────

left_col, right_col = st.columns([0.8, 5.2], gap="medium")

with left_col:
    st.markdown(
        """
        <style>
        div[data-testid="stVerticalBlock"] .st-key-doc_menu_assistant button,
        div[data-testid="stVerticalBlock"] .st-key-doc_menu_schedule button,
        div[data-testid="stVerticalBlock"] .st-key-doc_menu_medical_records button,
        div[data-testid="stVerticalBlock"] .st-key-doc_menu_logout button {
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
        key="doc_menu_assistant",
        use_container_width=True,
        type="primary" if st.session_state.doctor_screen == "assistant" else "secondary",
        on_click=_set_screen,
        args=("assistant",),
    )
    st.button(
        "Schedule",
        key="doc_menu_schedule",
        use_container_width=True,
        type="primary" if st.session_state.doctor_screen == "schedule" else "secondary",
        on_click=_set_screen,
        args=("schedule",),
    )
    st.button(
        "Medical Records",
        key="doc_menu_medical_records",
        use_container_width=True,
        type="primary" if st.session_state.doctor_screen == "medical_records" else "secondary",
        on_click=_set_screen,
        args=("medical_records",),
    )
    st.button(
        "Log out",
        key="doc_menu_logout",
        use_container_width=True,
        on_click=_logout,
    )

# ── Right-column content ──────────────────────────────────────────────────────

with right_col:
    screen = st.session_state.doctor_screen

    # ── Assistant ────────────────────────────────────────────────────────────
    if screen == "assistant":
        st.subheader("Doctor Assistant")

        today_appts = [
            r for r in fetch_appointments_for_day(date.today().isoformat())
            if str(r.get("doctor_id")) == str(doctor_id)
        ]

        history_key = f"chat_doctor_{doctor_id}"
        if history_key not in st.session_state:
            st.session_state[history_key] = [
                {
                    "role": "assistant",
                    "content": (
                        f"Hello Dr. {first_name}! I can check your schedule, look up patient "
                        "records, add chart entries, manage appointments, and answer clinical "
                        "questions. How can I help you today?"
                    ),
                }
            ]

        with st.container(height=430):
            for message in st.session_state[history_key]:
                with st.chat_message(message["role"]):
                    st.write(message["content"])

        prompt = st.chat_input("Ask about your schedule, a patient, or a clinical question...")
        if prompt:
            st.session_state[history_key].append({"role": "user", "content": prompt})
            with st.spinner("Doctor Assistant is thinking..."):
                reply = run_role_crew(
                    "doctor",
                    prompt,
                    today_appts,
                    chat_history=st.session_state[history_key],
                    doctor_id=doctor_id,
                    doctor_name=doctor_name,
                )
            st.session_state[history_key].append({"role": "assistant", "content": reply})
            st.rerun()

    # ── Schedule ─────────────────────────────────────────────────────────────
    elif screen == "schedule":
        import streamlit.components.v1 as _comp

        st.subheader("My Schedule")

        today = _nearest_weekday(date.today())
        if "doc_schedule_date" not in st.session_state:
            st.session_state.doc_schedule_date = today

        current_date = st.session_state.doc_schedule_date
        is_today = current_date == today

        # Chevron + Today navigation
        nav_l, nav_c, nav_r, nav_td = st.columns([0.5, 4, 0.5, 1])
        with nav_l:
            if st.button("◀", key="doc_sched_prev", use_container_width=True):
                st.session_state.doc_schedule_date = _prev_weekday(current_date)
                st.rerun()
        with nav_c:
            today_badge = (
                "&ensp;<span style='color:#60a5fa;font-size:0.68rem;font-weight:400;vertical-align:middle'>"
                "TODAY</span>"
            ) if is_today else ""
            st.markdown(
                f"<div style='text-align:center;font-size:1.05rem;font-weight:600;padding:0.3rem 0'>"
                f"{current_date.strftime('%A, %B %d, %Y')}{today_badge}</div>",
                unsafe_allow_html=True,
            )
        with nav_r:
            if st.button("▶", key="doc_sched_next", use_container_width=True):
                st.session_state.doc_schedule_date = _next_weekday(current_date)
                st.rerun()
        with nav_td:
            if st.button("↩ Today", key="doc_sched_today", use_container_width=True, disabled=is_today):
                st.session_state.doc_schedule_date = today
                st.rerun()

        # Appointments for the selected weekday
        all_appts = fetch_appointments_for_doctor(doctor_id)
        selected_day = current_date.isoformat()
        day_rows = sorted(
            [r for r in all_appts if r.get("start_at", "")[:10] == selected_day],
            key=lambda r: r.get("start_at", ""),
        )
        active_count = sum(1 for r in day_rows if r.get("status") != "cancelled")
        cancelled_count = len(day_rows) - active_count
        if day_rows:
            parts = [f"{active_count} active"]
            if cancelled_count:
                parts.append(f"{cancelled_count} cancelled")
            st.caption(" · ".join(parts))

        _comp.html(_render_gantt(day_rows), height=530)

    # ── Medical Records (CRUD) ────────────────────────────────────────────────
    elif screen == "medical_records":
        st.subheader("Medical Records")

        patients = fetch_patient_list()
        if not patients:
            st.info("No patients in the system.")
            st.stop()

        patient_options = {p["patient_name"]: p["patient_id"] for p in patients}
        sorted_names = sorted(patient_options.keys())

        selected_name = st.selectbox("Patient", options=sorted_names, key="doc_mr_patient")
        selected_patient_id = patient_options[selected_name]

        # Reset per-record state when the selected patient changes
        if st.session_state.get("doc_mr_last_patient") != selected_patient_id:
            st.session_state.doc_mr_last_patient = selected_patient_id
            st.session_state.doc_mr_editing = None
            st.session_state.doc_mr_deleting = None
            st.session_state.doc_mr_adding = False

        for _k, _default in (("doc_mr_editing", None), ("doc_mr_deleting", None), ("doc_mr_adding", False)):
            if _k not in st.session_state:
                st.session_state[_k] = _default

        records = fetch_medical_records(selected_patient_id)
        enc_names = [name for _, name in ENCOUNTER_TYPES]
        default_enc_idx = enc_names.index("OfficeVisit") if "OfficeVisit" in enc_names else 0

        # ── New-record toggle ────────────────────────────────────────────────
        add_lbl = "✖ Cancel" if st.session_state.doc_mr_adding else "➕ New Record"
        if st.button(add_lbl, key="doc_mr_toggle_add"):
            st.session_state.doc_mr_adding = not st.session_state.doc_mr_adding
            st.session_state.doc_mr_editing = None
            st.session_state.doc_mr_deleting = None
            st.rerun()

        if st.session_state.doc_mr_adding:
            with st.form("doc_mr_add_form", clear_on_submit=False):
                new_enc = st.selectbox("Encounter Type", enc_names, index=default_enc_idx)
                new_note_add = st.text_area(
                    "Note", height=160,
                    placeholder="Chief complaint, exam findings, assessment, plan, medications…",
                )
                btn_col1, btn_col2, _ = st.columns([1, 1, 5])
                with btn_col1:
                    submitted_add = st.form_submit_button("💾 Save", type="primary", use_container_width=True)
                with btn_col2:
                    cancelled_add = st.form_submit_button("Cancel", use_container_width=True)

            if submitted_add:
                if not new_note_add.strip():
                    st.error("Note cannot be empty.")
                else:
                    add_medical_record(selected_patient_id, new_note_add.strip(), doctor_id, new_enc)
                    st.session_state.doc_mr_adding = False
                    st.success("Record added.")
                    st.rerun()
            if cancelled_add:
                st.session_state.doc_mr_adding = False
                st.rerun()

        st.divider()

        # ── Records list ─────────────────────────────────────────────────────
        if not records:
            st.info(f"No medical records on file for **{selected_name}**.")
        else:
            st.caption(f"{len(records)} record(s) for {selected_name}")

            for rec in records:
                rid = rec["record_id"]
                created = rec.get("created_at", "")
                etype = rec.get("encounter_type", "OfficeVisit")
                doc_nm = rec.get("doctor_name") or "—"
                note_text = rec.get("note", "")

                editing_id = st.session_state.doc_mr_editing
                deleting_id = st.session_state.doc_mr_deleting

                # Row header with Edit / Delete buttons
                hdr_col, btn_edit_col, btn_del_col = st.columns([5.5, 0.6, 0.6])
                with hdr_col:
                    st.markdown(
                        f"**{created}** &nbsp;·&nbsp; {etype} &nbsp;·&nbsp; *{doc_nm}*",
                        unsafe_allow_html=True,
                    )
                with btn_edit_col:
                    if st.button(
                        "✏️", key=f"edit_btn_{rid}", help="Edit this record",
                        type="primary" if editing_id == rid else "secondary",
                    ):
                        st.session_state.doc_mr_editing = None if editing_id == rid else rid
                        st.session_state.doc_mr_deleting = None
                        st.session_state.doc_mr_adding = False
                        st.rerun()
                with btn_del_col:
                    if st.button(
                        "🗑️", key=f"del_btn_{rid}", help="Delete this record",
                        type="primary" if deleting_id == rid else "secondary",
                    ):
                        st.session_state.doc_mr_deleting = None if deleting_id == rid else rid
                        st.session_state.doc_mr_editing = None
                        st.session_state.doc_mr_adding = False
                        st.rerun()

                # Inline edit form
                if editing_id == rid:
                    cur_idx = enc_names.index(etype) if etype in enc_names else 0
                    with st.form(f"doc_mr_edit_{rid}", clear_on_submit=False):
                        upd_enc = st.selectbox("Encounter Type", enc_names, index=cur_idx)
                        upd_note = st.text_area("Note", value=note_text, height=200)
                        sv_col, cn_col, _ = st.columns([1, 1, 5])
                        with sv_col:
                            save_edit = st.form_submit_button("💾 Save", type="primary", use_container_width=True)
                        with cn_col:
                            cancel_edit = st.form_submit_button("Cancel", use_container_width=True)
                    if save_edit:
                        if not upd_note.strip():
                            st.error("Note cannot be empty.")
                        else:
                            update_medical_record(rid, upd_note.strip(), upd_enc)
                            st.session_state.doc_mr_editing = None
                            st.success("Record updated.")
                            st.rerun()
                    if cancel_edit:
                        st.session_state.doc_mr_editing = None
                        st.rerun()

                # Delete confirmation
                elif deleting_id == rid:
                    st.warning("⚠️ Delete this record? This action cannot be undone.")
                    conf_col, canc_col, _ = st.columns([1.5, 1, 4])
                    with conf_col:
                        if st.button("🗑️ Confirm Delete", key=f"confirm_del_{rid}", type="primary"):
                            delete_medical_record(rid)
                            st.session_state.doc_mr_deleting = None
                            st.success("Record deleted.")
                            st.rerun()
                    with canc_col:
                        if st.button("Cancel", key=f"cancel_del_{rid}"):
                            st.session_state.doc_mr_deleting = None
                            st.rerun()

                # Default: note preview
                else:
                    preview = note_text[:280] + " …" if len(note_text) > 280 else note_text
                    st.caption(preview)

                st.divider()

