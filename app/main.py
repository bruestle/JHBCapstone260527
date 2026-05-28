from __future__ import annotations

import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="requests")

from datetime import date

import streamlit as st

from db import init_db, is_db_seeded, seed_mock_data

st.set_page_config(page_title="Healthcare Assistant", page_icon="🏥", layout="wide")

init_db()
if not is_db_seeded():
    with st.spinner("Seeding demo database..."):
        seed_mock_data(reset=False, seed_start=date.today())

if "authenticated" not in st.session_state:
    st.session_state.authenticated = False

login_page = st.Page("_pages/login.py", title="Login", icon="🔐")
patient_page = st.Page("_pages/patient.py", title="Patient", icon="🧑")
doctor_page = st.Page("_pages/doctor.py", title="Doctor", icon="👨‍⚕️")
admin_page = st.Page("_pages/admin.py", title="Admin", icon="🛡️")

if not st.session_state.authenticated:
    navigation = st.navigation([login_page], position="hidden")
else:
    role = st.session_state.get("role")
    if role == "patient":
        navigation = st.navigation([patient_page], position="sidebar")
    elif role == "doctor":
        navigation = st.navigation([doctor_page], position="sidebar")
    elif role == "admin":
        navigation = st.navigation([admin_page], position="sidebar")
    else:
        st.session_state.authenticated = False
        navigation = st.navigation([login_page], position="hidden")

navigation.run()
