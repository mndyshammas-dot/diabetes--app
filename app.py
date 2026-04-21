import streamlit as st
import numpy as np
import pandas as pd
import joblib
import sqlite3
from datetime import datetime

# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="Diabetes AI", layout="wide")

# ---------------- LOAD MODEL ----------------
@st.cache_resource
def load_model():
    return joblib.load("model/svm_model.pkl")

model = load_model()

# ---------------- DB ----------------
def init_db():
    conn = sqlite3.connect("predictions.db")
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            pregnancies INT,
            glucose INT,
            bp INT,
            skin INT,
            insulin INT,
            bmi REAL,
            dpf REAL,
            age INT,
            prediction TEXT,
            confidence REAL,
            timestamp TEXT
        )
    """)
    conn.commit()
    conn.close()

init_db()

# ---------------- SIDEBAR ----------------
st.sidebar.title("🧠 Diabetes AI")
menu = st.sidebar.radio("", ["Dashboard", "New Prediction", "History"])

# ---------------- DASHBOARD ----------------
if menu == "Dashboard":

    st.title("Diabetes AI Prediction System")

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Total Predictions", "32")
    col2.metric("Diabetic Cases", "16")
    col3.metric("Healthy Cases", "16")
    col4.metric("Avg Confidence", "85.5%")

    st.markdown("---")

    colA, colB = st.columns([2, 1])

    # ---------------- FORM ----------------
    with colA:
        st.subheader("Make a New Prediction")

        c1, c2 = st.columns(2)

        pregnancies = c1.number_input("Pregnancies", 0, 20, 1)
        insulin = c2.number_input("Insulin", 0, 900, 120)

        glucose = c1.number_input("Glucose", 0, 300, 130)
        bmi = c2.number_input("BMI", 0.0, 70.0, 30.0)

        bp = c1.number_input("Blood Pressure", 0, 200, 75)
        dpf = c2.number_input("DPF", 0.0, 2.5, 0.5)

        skin = c1.number_input("Skin Thickness", 0, 100, 25)
        age = c2.number_input("Age", 1, 120, 40)

        if st.button("🚀 Analyze Risk"):

            input_data = np.array([[pregnancies, glucose, bp, skin,
                                    insulin, bmi, dpf, age]])

            prediction = model.predict(input_data)[0]
            prob = model.predict_proba(input_data)[0]

            result = "Diabetic" if prediction == 1 else "Not Diabetic"
            confidence = max(prob) * 100

            # Save to DB
            conn = sqlite3.connect("predictions.db")
            c = conn.cursor()
            c.execute("""
                INSERT INTO predictions 
                (pregnancies, glucose, bp, skin, insulin, bmi, dpf, age, prediction, confidence, timestamp)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (pregnancies, glucose, bp, skin, insulin, bmi, dpf, age,
                  result, confidence, str(datetime.now())))
            conn.commit()
            conn.close()

            st.success(f"Prediction: {result}")
            st.progress(int(confidence))

    # ---------------- RESULT ----------------
    with colB:
        st.subheader("Prediction Result")

        st.error("HIGH RISK")
        st.write("Likely Diabetic")
        st.progress(92)

# ---------------- HISTORY ----------------
elif menu == "History":
    st.title("Prediction History")

    conn = sqlite3.connect("predictions.db")
    df = pd.read_sql("SELECT * FROM predictions ORDER BY id DESC", conn)
    conn.close()

    st.dataframe(df)

# ---------------- NEW PREDICTION ----------------
elif menu == "New Prediction":
    st.title("New Prediction Page")
    st.info("Use dashboard to make prediction")
