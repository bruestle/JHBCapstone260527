from __future__ import annotations

import streamlit as st

from db import DOCTOR_SEED, PATIENT_SEED


def _set_selected_role(role: str) -> None:
    st.session_state.selected_role = role


def _username_to_user_id(role: str, username: str) -> str | None:
    entered = username.strip().lower()

    if role == "Patient":
        for patient_id, _first_name, last_name in PATIENT_SEED:
            if last_name.lower() == entered:
                return patient_id
        return None

    if role == "Doctor":
        for doctor_id, _first_name, last_name, _specialty in DOCTOR_SEED:
            if last_name.lower() == entered:
                return doctor_id
        return None

    if entered == "admin":
        return "admin"
    return None


def _allowed_users_line(role: str) -> str:
    if role == "Patient":
        users = [p[2] for p in PATIENT_SEED]
    elif role == "Doctor":
        users = [d[2] for d in DOCTOR_SEED]
    else:
        users = ["admin"]
    return f"Available Users: {', '.join(users)}"


st.title("Healthcare Assistant")
st.caption("Select your role to continue")

st.markdown(
    """
    <style>
    div[data-testid="stButton"] {
        display: inline-block;
        margin-right: 10px;
        margin-bottom: 0;
        vertical-align: top;
        white-space: nowrap;
    }
    div[data-testid="stButton"]:last-of-type {
        margin-right: 0;
    }
    div[data-testid="stButton"] > button {
        width: 120px;
        height: 2.5rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

if "selected_role" not in st.session_state:
    st.session_state.selected_role = ""

selected_role = st.session_state.selected_role

col_patient, col_doctor, col_admin, _spacer = st.columns([1, 1, 1, 6], gap="small")
with col_patient:
    st.button(
        "🧑 Patient",
        key="login_role_patient",
        type="primary" if selected_role == "Patient" else "secondary",
        on_click=_set_selected_role,
        args=("Patient",),
    )
with col_doctor:
    st.button(
        "👨 Doctor",
        key="login_role_doctor",
        type="primary" if selected_role == "Doctor" else "secondary",
        on_click=_set_selected_role,
        args=("Doctor",),
    )
with col_admin:
    st.button(
        "🛡️ Admin",
        key="login_role_admin",
        type="primary" if selected_role == "Admin" else "secondary",
        on_click=_set_selected_role,
        args=("Admin",),
    )

selected_role = st.session_state.selected_role

submitted = False
username = ""

if selected_role:
    with st.form("login_form"):
        username = st.text_input("Username")
        submitted = st.form_submit_button("Log In")

if submitted:
    resolved_user_id = _username_to_user_id(selected_role, username)

    if not selected_role:
        st.error("Select a role first.")
    elif resolved_user_id is not None:
        st.session_state.authenticated = True
        st.session_state.role = selected_role.lower()
        st.session_state.user_id = resolved_user_id
        st.rerun()
    else:
        st.error("Invalid credentials")

if selected_role:
    st.markdown(
        f"<div style='opacity:0.45;font-size:0.85rem'>{_allowed_users_line(selected_role)}</div>",
        unsafe_allow_html=True,
    )
