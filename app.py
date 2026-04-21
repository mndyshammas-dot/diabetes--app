import streamlit as st
import numpy as np
import joblib
import sqlite3
import pandas as pd
from datetime import datetime

# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="Diabetes AI", layout="wide")

# ---------------- STYLE ----------------
st.markdown("""
<style>
.block-container {
    padding-top: 2rem;
}
.card {
    background-color: #ffffff;
    padding: 20px;
    border-radius: 12px;
    box-shadow: 0px 4px 12px rgba(0,0,0,0.1);
}
h1, h2, h3 {
    color: #0f172a;
}
</style>
""", unsafe_allow_html=True)

# ---------------- LOAD MODEL ----------------
@st.cache_resource
def load_model():
    return joblib.load("model/svm_model.pkl")

model = load_model()

# ---------------- DATABASE ----------------
def init_db():
    conn = sqlite3.connect("predictions.db")
    c = conn.cursor()

    c.execute("""
    CREATE TABLE IF NOT EXISTS predictions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        pregnancies INTEGER,
        glucose INTEGER,
        blood_pressure INTEGER,
        skin_thickness INTEGER,
        insulin INTEGER,
        bmi REAL,
        dpf REAL,
        age INTEGER,
        prediction INTEGER,
        confidence REAL,
        timestamp TEXT
    )
    """)

    conn.commit()
    conn.close()

init_db()

def save_to_db(data, prediction, confidence):
    conn = sqlite3.connect("predictions.db")
    c = conn.cursor()

    c.execute("""
    INSERT INTO predictions (
        pregnancies, glucose, blood_pressure, skin_thickness,
        insulin, bmi, dpf, age, prediction, confidence, timestamp
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (*data, prediction, confidence, datetime.now()))

    conn.commit()
    conn.close()

# ---------------- SIDEBAR ----------------
st.sidebar.title("🧠 Diabetes AI")
page = st.sidebar.radio("Menu", ["🏥 Predict", "📊 Dashboard", "📜 History"])

st.sidebar.markdown("---")
st.sidebar.info("AI-based diabetes risk prediction system")

# ---------------- PREDICT PAGE ----------------
if page == "🏥 Predict":

    st.title("🏥 Diabetes Risk Checker")
    st.caption("Fill patient details to assess diabetes risk")

    st.markdown("### 🧾 Patient Information")

    col1, col2 = st.columns(2)

    with col1:
        pregnancies = st.number_input("Pregnancies", 0, 20)
        glucose = st.number_input("Glucose (mg/dL)", 0, 300)
        blood_pressure = st.number_input("Blood Pressure (mm Hg)", 0, 200)
        skin_thickness = st.number_input("Skin Thickness", 0, 100)

    with col2:
        insulin = st.number_input("Insulin", 0, 900)
        bmi = st.number_input("BMI", 0.0, 60.0)
        dpf = st.number_input("Diabetes Pedigree Function", 0.0, 3.0)
        age = st.number_input("Age", 1, 120)

    st.markdown("---")

    input_data = np.array([
        pregnancies, glucose, blood_pressure,
        skin_thickness, insulin, bmi, dpf, age
    ]).reshape(1, -1)

    if st.button("🔍 Analyze Risk"):

        if glucose < 50 or bmi < 10:
            st.error("⚠️ Please enter realistic medical values")
        else:
            pred = model.predict(input_data)[0]
            prob = model.predict_proba(input_data)[0][pred]

            save_to_db(
                [pregnancies, glucose, blood_pressure,
                 skin_thickness, insulin, bmi, dpf, age],
                int(pred), float(prob)
            )

            st.markdown("## 🧾 Result")

            if pred == 1:
                st.error("⚠️ High Risk of Diabetes")
            else:
                st.success("✅ Low Risk")

            st.metric("Confidence Score", f"{prob*100:.2f}%")

            # Progress bar
            st.progress(float(prob))

            # -------- EXPLANATION --------
            st.markdown("### 🧠 Health Insights")

            if glucose > 140:
                st.write("🔴 High glucose level detected")
            if bmi > 30:
                st.write("🔴 Obesity risk (high BMI)")
            if age > 45:
                st.write("🟡 Age increases risk")
            if dpf > 0.5:
                st.write("🟡 Family history risk")

            st.info("⚠️ This is not a medical diagnosis. Consult a doctor.")

# ---------------- DASHBOARD ----------------
elif page == "📊 Dashboard":

    st.title("📊 Analytics Dashboard")

    conn = sqlite3.connect("predictions.db")
    df = pd.read_sql("SELECT * FROM predictions", conn)
    conn.close()

    if not df.empty:

        total = len(df)
        diabetic = df["prediction"].sum()
        healthy = total - diabetic

        col1, col2, col3 = st.columns(3)

        col1.metric("Total Records", total)
        col2.metric("Diabetic Cases", diabetic)
        col3.metric("Healthy Cases", healthy)

        st.markdown("### 📊 Distribution")
        st.bar_chart(df["prediction"].value_counts())

        st.markdown("### 📈 Confidence Trend")
        st.line_chart(df["confidence"])

    else:
        st.warning("No data available yet")

# ---------------- HISTORY ----------------
elif page == "📜 History":

    st.title("📜 Prediction History")

    conn = sqlite3.connect("predictions.db")
    df = pd.read_sql("SELECT * FROM predictions ORDER BY id DESC", conn)
    conn.close()

    if not df.empty:
        st.dataframe(df, use_container_width=True)
    else:
        st.warning("No history found")