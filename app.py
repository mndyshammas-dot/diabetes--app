import streamlit as st
import numpy as np
import pandas as pd
import joblib
import sqlite3
from datetime import datetime

# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="Diabetes AI", layout="wide")

# ---------------- STYLE ----------------
st.markdown("""
<style>
body {background-color: #0e1117;}
.card {
    background: #1c1f26;
    padding: 20px;
    border-radius: 12px;
    text-align: center;
}
.metric-title {
    color: gray;
    font-size: 14px;
}
.metric-value {
    font-size: 26px;
    font-weight: bold;
}
</style>
""", unsafe_allow_html=True)

# ---------------- LOAD MODEL ----------------
@st.cache_resource
def load_model():
    return joblib.load("model/svm_model.pkl")

model = load_model()

# ---------------- DATABASE ----------------
def get_connection():
    return sqlite3.connect("predictions.db", check_same_thread=False)

def init_db():
    conn = get_connection()
    c = conn.cursor()

    # 🔥 FIX: reset table (important)
    c.execute("DROP TABLE IF EXISTS predictions")

    c.execute("""
        CREATE TABLE predictions (
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

def load_data():
    conn = get_connection()
    df = pd.read_sql("SELECT * FROM predictions ORDER BY id DESC", conn)
    conn.close()
    return df

# ---------------- SIDEBAR ----------------
st.sidebar.title("🧠 Diabetes AI")
menu = st.sidebar.radio("", ["Dashboard", "History"])

# ---------------- DASHBOARD ----------------
if menu == "Dashboard":

    st.title("Diabetes AI Prediction System")
    st.caption("Smart health risk analysis using Machine Learning")

    df = load_data()

    total = len(df)
    diabetic = len(df[df["prediction"] == "Diabetic"])
    healthy = len(df[df["prediction"] == "Not Diabetic"])
    avg_conf = df["confidence"].mean() if total > 0 else 0

    # ----------- METRICS -----------
    col1, col2, col3, col4 = st.columns(4)

    def card(title, value):
        st.markdown(f"""
        <div class="card">
            <div class="metric-title">{title}</div>
            <div class="metric-value">{value}</div>
        </div>
        """, unsafe_allow_html=True)

    with col1: card("Total Predictions", total)
    with col2: card("Diabetic Cases", diabetic)
    with col3: card("Healthy Cases", healthy)
    with col4: card("Avg Confidence", f"{avg_conf:.1f}%")

    st.markdown("---")

    colA, colB = st.columns([2, 1])

    # ----------- INPUT FORM -----------
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

            try:
                input_data = np.array([[pregnancies, glucose, bp, skin,
                                        insulin, bmi, dpf, age]])

                prediction = model.predict(input_data)[0]
                prob = model.predict_proba(input_data)[0]

                result = "Diabetic" if prediction == 1 else "Not Diabetic"
                confidence = float(max(prob) * 100)

                # SAVE TO DB (SAFE)
                conn = get_connection()
                c = conn.cursor()

                c.execute("""
                    INSERT INTO predictions 
                    (pregnancies, glucose, bp, skin, insulin, bmi, dpf, age, prediction, confidence, timestamp)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    int(pregnancies),
                    int(glucose),
                    int(bp),
                    int(skin),
                    int(insulin),
                    float(bmi),
                    float(dpf),
                    int(age),
                    result,
                    confidence,
                    str(datetime.now())
                ))

                conn.commit()
                conn.close()

                st.success(f"Prediction: {result}")

            except Exception as e:
                st.error(f"Error: {e}")

    # ----------- RESULT PANEL -----------
    with colB:
        st.subheader("Prediction Result")

        if total > 0:
            last = df.iloc[0]

            if last["prediction"] == "Diabetic":
                st.error("HIGH RISK")
            else:
                st.success("LOW RISK")

            st.write(last["prediction"])
            st.progress(int(last["confidence"]))
        else:
            st.info("No predictions yet")

# ---------------- HISTORY ----------------
elif menu == "History":

    st.title("Prediction History")

    df = load_data()

    if len(df) > 0:
        st.dataframe(df, use_container_width=True)
    else:
        st.info("No data available")