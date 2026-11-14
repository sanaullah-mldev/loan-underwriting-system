
import os
import sqlite3
import joblib
import pandas as pd
import streamlit as st

# Page configuration
st.set_page_config(page_title="Loan Underwriting Portal", layout="wide")

# Use relative paths so app works on both local machine and Streamlit Cloud
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "loan_underwriting_pipeline.joblib")
DB_PATH = os.path.join(BASE_DIR, "loan_underwriting.db")

def get_db_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS applications (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            full_name TEXT NOT NULL,
            phone_number TEXT NOT NULL,
            cnic_or_id TEXT NOT NULL,
            gender TEXT,
            married TEXT,
            dependents TEXT,
            education TEXT,
            self_employed TEXT,
            applicant_income REAL,
            coapplicant_income REAL,
            loan_amount REAL,
            loan_amount_term REAL,
            credit_history REAL,
            property_area TEXT,
            loan_status TEXT,
            approval_probability REAL
        )
    """)
    conn.commit()
    return conn

@st.cache_resource
def load_pipeline():
    return joblib.load(MODEL_PATH)

def login_screen():
    st.title("🔐 Loan Underwriting & Credit Scoring Portal")
    st.subheader("Authorized Staff Login")
    with st.form("login_form"):
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        submitted = st.form_submit_button("Login")
        if submitted:
            # Accept any non-empty username and password
            if username.strip() and password.strip():
                st.session_state["authenticated"] = True
                st.session_state["current_user"] = username.strip()
                st.rerun()
            else:
                st.error("Please enter both username and password.")

def main_portal():
    col_title, col_logout = st.columns([5, 1])
    with col_title:
        st.title("🏦 Automated Loan Underwriting & Credit Scoring System")
        st.caption(f"Logged in as: {st.session_state.get('current_user', 'staff')}")
    with col_logout:
        if st.button("Logout"):
            st.session_state["authenticated"] = False
            st.rerun()

    pipeline = load_pipeline()
    conn = get_db_connection()

    tab1, tab2 = st.tabs(["📝 New Loan Application", "📊 Manager Audit & History"])

    with tab1:
        st.subheader("Applicant Identity & Financial Details")
        with st.form("loan_form"):
            c1, c2, c3 = st.columns(3)
            with c1:
                full_name = st.text_input("Full Name", value="Ali Raza")
                phone_number = st.text_input("Phone Number", value="0300-1234567")
                cnic_or_id = st.text_input("CNIC / Applicant ID", value="32304-1234567-1")
                gender = st.selectbox("Gender", ["Male", "Female"])
                married = st.selectbox("Married", ["Yes", "No"])
            with c2:
                dependents = st.selectbox("Dependents", ["0", "1", "2", "3+"])
                education = st.selectbox("Education", ["Graduate", "Not Graduate"])
                self_employed = st.selectbox("Self Employed", ["No", "Yes"])
                property_area = st.selectbox("Property Area", ["Urban", "Semiurban", "Rural"])
            with c3:
                applicant_income = st.number_input("Applicant Income", min_value=0.0, value=6500.0, step=100.0)
                coapplicant_income = st.number_input("Coapplicant Income", min_value=0.0, value=2000.0, step=100.0)
                loan_amount = st.number_input("Loan Amount (in thousands)", min_value=1.0, value=150.0, step=5.0)
                loan_amount_term = st.number_input("Loan Amount Term (months)", min_value=12.0, value=360.0, step=12.0)
                credit_history = st.selectbox("Credit History", [1.0, 0.0])

            submit_app = st.form_submit_button("Evaluate Loan Application")

        if submit_app:
            # Prepare ML input DataFrame (only model features)
            input_df = pd.DataFrame([{
                "Gender": gender,
                "Married": married,
                "Dependents": dependents,
                "Education": education,
                "Self_Employed": self_employed,
                "ApplicantIncome": applicant_income,
                "CoapplicantIncome": coapplicant_income,
                "LoanAmount": loan_amount,
                "Loan_Amount_Term": loan_amount_term,
                "Credit_History": credit_history,
                "Property_Area": property_area
            }])

            pred_class = int(pipeline.predict(input_df)[0])
            pred_prob = round(float(pipeline.predict_proba(input_df)[0][1]) * 100, 2)
            status_label = "Approved" if pred_class == 1 else "Rejected"

            # Save full record (identity + features + result) to SQLite
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO applications (
                    full_name, phone_number, cnic_or_id, gender, married, dependents,
                    education, self_employed, applicant_income, coapplicant_income,
                    loan_amount, loan_amount_term, credit_history, property_area,
                    loan_status, approval_probability
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                full_name, phone_number, cnic_or_id, gender, married, dependents,
                education, self_employed, applicant_income, coapplicant_income,
                loan_amount, loan_amount_term, credit_history, property_area,
                status_label, pred_prob
            ))
            conn.commit()

            # Display final decision with applicant identity
            st.markdown("---")
            st.subheader("Final Underwriting Decision")
            m1, m2, m3 = st.columns(3)
            m1.metric("Applicant", f"{full_name} ({cnic_or_id})")
            m2.metric("Decision", status_label)
            m3.metric("Approval Probability", f"{pred_prob}%")

            if status_label == "Approved":
                st.success(f"{full_name} ({phone_number}) — Loan Approved with {pred_prob}% probability.")
            else:
                st.error(f"{full_name} ({phone_number}) — Loan Rejected ({pred_prob}% approval probability).")

    with tab2:
        st.subheader("Saved Applications in SQLite Database")
        history_df = pd.read_sql_query("SELECT * FROM applications ORDER BY id DESC", conn)
        if history_df.empty:
            st.info("No applications recorded yet.")
        else:
            c_kpi1, c_kpi2, c_kpi3 = st.columns(3)
            c_kpi1.metric("Total Applications", len(history_df))
            c_kpi2.metric("Approved", int((history_df["loan_status"] == "Approved").sum()))
            c_kpi3.metric("Rejected", int((history_df["loan_status"] == "Rejected").sum()))
            st.dataframe(history_df, use_container_width=True)

    conn.close()

if not st.session_state.get("authenticated", False):
    login_screen()
else:
    main_portal()
