import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import joblib

# -------------------- CONFIGURATION --------------------
DATA_PATH = "StudentsPerformance.csv"
MODEL_PATH = "student_performance_model.pkl"
SCALER_PATH = "scaler.pkl"

st.set_page_config(
    page_title="Student Performance Predictor",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -------------------- STYLING --------------------
st.markdown("""
<style>
[data-testid="stAppViewContainer"] {
    background: linear-gradient(135deg, #1e293b, #0f172a);
    color: white !important;
}
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0f172a, #1e293b);
    color: #e2e8f0;
}
.main-header {
    font-size: 2.8rem;
    font-weight: 800;
    text-align: center;
    color: #38bdf8;
    margin-bottom: 1rem;
    text-shadow: 0px 0px 8px rgba(56,189,248,0.5);
}
.section-header {
    font-size: 1.6rem;
    font-weight: 700;
    color: #93c5fd;
    margin-top: 1.5rem;
    margin-bottom: 0.5rem;
    border-left: 4px solid #3b82f6;
    padding-left: 8px;
}
div[data-testid="stMetricValue"] {
    color: #38bdf8 !important;
}
button[kind="secondary"] {
    background: linear-gradient(135deg, #2563eb, #3b82f6);
    color: white;
    border-radius: 10px;
    font-weight: 600;
    border: none;
    box-shadow: 0px 0px 6px rgba(37,99,235,0.6);
}
button[kind="secondary"]:hover {
    background: linear-gradient(135deg, #1d4ed8, #2563eb);
}
.footer {
    text-align: center;
    color: #94a3b8;
    font-size: 0.9rem;
    padding: 10px;
    margin-top: 30px;
}
</style>
""", unsafe_allow_html=True)

# -------------------- LOAD & PREPROCESS --------------------
@st.cache_data
def load_and_preprocess_data():
    if not os.path.exists(DATA_PATH):
        return None
    df = pd.read_csv(DATA_PATH)
    df.columns = [c.strip() for c in df.columns]
    for col in ["math score", "reading score", "writing score"]:
        df[col] = pd.to_numeric(df[col], errors="coerce") / 2.0
    df.dropna(inplace=True)
    df["average_score"] = df[["math score", "reading score", "writing score"]].mean(axis=1)
    df["performance_level"] = df["average_score"].apply(
        lambda x: "High" if x >= 40 else "Medium" if x >= 30 else "Low"
    )
    return df


def train_and_save_model(df):
    categorical = ["gender", "parental level of education", "lunch", "test preparation course"]
    numeric = ["math score", "reading score", "writing score"]

    X = pd.get_dummies(df[categorical + numeric])
    y = df["performance_level"]

    scaler = StandardScaler()
    X[numeric] = scaler.fit_transform(X[numeric])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = RandomForestClassifier(n_estimators=200, random_state=42)
    model.fit(X_train, y_train)

    # Save model and scaler
    joblib.dump(model, MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    return model, scaler, X, X_test, y_test, y_pred, acc, numeric


def load_saved_model():
    if os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH):
        model = joblib.load(MODEL_PATH)
        scaler = joblib.load(SCALER_PATH)
        return model, scaler
    return None, None


def prepare_student_row(student_dict, X_cols, scaler, numeric_cols):
    row = pd.DataFrame([student_dict])
    row_encoded = pd.get_dummies(row)
    row_encoded = row_encoded.reindex(columns=X_cols, fill_value=0)
    row_encoded[numeric_cols] = scaler.transform(row_encoded[numeric_cols])
    return row_encoded


# -------------------- HEADER --------------------
st.markdown("<h1 class='main-header'>🎓 Student Performance Predictor</h1>", unsafe_allow_html=True)
st.caption("📘 Smart ML Dashboard to Analyze and Predict Student Performance")

# -------------------- LOAD MODEL OR TRAIN --------------------
df = load_and_preprocess_data()

if df is not None:
    model, scaler, X, X_test, y_test, y_pred, accuracy, numeric_cols = train_and_save_model(df)
else:
    model, scaler = load_saved_model()
    if model is None:
        st.error("❌ No dataset or saved model found. Please upload the dataset first.")
        st.stop()
    else:
        st.warning("⚠️ Dataset not found — using previously saved model.")
        X, X_test, y_test, y_pred, accuracy, numeric_cols = None, None, None, None, 0.0, ["math score", "reading score", "writing score"]

# -------------------- SIDEBAR --------------------
with st.sidebar:
    st.markdown("## 🔍 Navigation")
    page = st.radio("Select a Page", [
        "📄 Overview",      # NEW TAB
        "🏠 Home",
        "📊 Data Analysis",
        "🤖 Model Performance",
        "🔮 Predict Student"
    ])

# -------------------- PROJECT OVERVIEW --------------------
if page == "📄 Overview":
    st.markdown("<h2 class='section-header'>Project Overview</h2>", unsafe_allow_html=True)
    st.markdown("""
    ### 🎓 Student Performance Predictor

    **Objective:** Predict student performance levels (Low, Medium, High) based on their characteristics and test scores.

    **Features:**
    - Analyze student data and visualize trends
    - Train a Random Forest machine learning model
    - Predict performance for new students
    - Save and reuse trained models
    - Interactive dashboard with dark theme

    **Technology Stack:**
    - Python
    - Streamlit
    - Pandas, NumPy
    - Scikit-learn
    - Matplotlib, Seaborn

    **How it works:**
    1. Load the dataset (or use saved model if dataset unavailable)
    2. Train the Random Forest model on student data
    3. Predict performance for any student
    4. Visualize data trends and model insights

    This tab provides an at-a-glance understanding of the project and its functionality.
    """)

# -------------------- HOME PAGE --------------------
elif page == "🏠 Home":
    st.markdown("<h2 class='section-header'>Overview</h2>", unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    if df is not None:
        c1.metric("Total Students", len(df))
    c2.metric("Model Accuracy", f"{accuracy*100:.2f}%")
    if df is not None:
        c3.metric("Features Used", df.shape[1])
    st.info("This AI model predicts student performance levels (Low, Medium, High) using Random Forest.")

    if df is not None:
        st.markdown("<h3 class='section-header'>Preview of Data</h3>", unsafe_allow_html=True)
        st.dataframe(df.head(10), use_container_width=True)

# -------------------- DATA ANALYSIS --------------------
elif page == "📊 Data Analysis":
    if df is None:
        st.warning("⚠️ Dataset not available for analysis.")
    else:
        st.markdown("<h2 class='section-header'>Exploratory Data Analysis</h2>", unsafe_allow_html=True)
        tab1, tab2, tab3 = st.tabs(["Distribution", "Gender-wise", "Correlation"])

        with tab1:
            st.subheader("Average Score Distribution")
            fig, ax = plt.subplots(figsize=(8, 5))
            sns.histplot(df["average_score"], bins=16, kde=True, color="#3b82f6", ax=ax)
            ax.set_title("Distribution of Average Scores (/50)", color="white")
            st.pyplot(fig)

        with tab2:
            st.subheader("Performance Level by Gender")
            fig, ax = plt.subplots(figsize=(8, 5))
            sns.countplot(x="gender", hue="performance_level", data=df, palette="coolwarm", ax=ax)
            ax.set_title("Performance by Gender", color="white")
            st.pyplot(fig)

        with tab3:
            st.subheader("Correlation Heatmap")
            fig, ax = plt.subplots(figsize=(6, 5))
            corr = df[["math score", "reading score", "writing score", "average_score"]].corr()
            sns.heatmap(corr, annot=True, cmap="mako", fmt=".2f", ax=ax)
            ax.set_title("Correlation Between Scores", color="white")
            st.pyplot(fig)

# -------------------- MODEL PERFORMANCE --------------------
elif page == "🤖 Model Performance":
    st.markdown("<h2 class='section-header'>Model Evaluation</h2>", unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    col1.metric("Test Accuracy", f"{accuracy*100:.2f}%")
    if y_test is not None and y_pred is not None:
        col2.metric("Samples Tested", len(y_test))

        st.subheader("Confusion Matrix")
        fig, ax = plt.subplots(figsize=(6, 4))
        cm = confusion_matrix(y_test, y_pred, labels=["Low", "Medium", "High"])
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        st.pyplot(fig)
    else:
        st.info("ℹ️ Model loaded from saved version — training data unavailable for metrics display.")

# -------------------- PREDICT STUDENT --------------------
elif page == "🔮 Predict Student":
    st.markdown("<h2 class='section-header'>Predict Student Performance</h2>", unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        gender = st.selectbox("Gender", ["female", "male"])
        parental_education = st.selectbox("Parental Education", [
            "some high school", "high school", "some college", "associate's degree", "bachelor's degree", "master's degree"
        ])
        lunch = st.selectbox("Lunch Type", ["standard", "free/reduced"])

    with col2:
        test_prep = st.selectbox("Test Prep Course", ["none", "completed"])
        math = st.slider("Math Score (/50)", 0.0, 50.0, 25.0)
        reading = st.slider("Reading Score (/50)", 0.0, 50.0, 25.0)
        writing = st.slider("Writing Score (/50)", 0.0, 50.0, 25.0)

    if st.button("🎯 Predict Performance", use_container_width=True):
        student = {
            "gender": gender,
            "parental level of education": parental_education,
            "lunch": lunch,
            "test preparation course": test_prep,
            "math score": math,
            "reading score": reading,
            "writing score": writing
        }

        if df is not None:
            X_cols = pd.get_dummies(df[["gender", "parental level of education", "lunch", "test preparation course",
                                        "math score", "reading score", "writing score"]]).columns
        else:
            X_cols = model.feature_names_in_

        student_row = prepare_student_row(student, X_cols, scaler, ["math score", "reading score", "writing score"])
        prediction = model.predict(student_row)[0]
        prob = model.predict_proba(student_row)[0]

        avg_score = (math + reading + writing) / 3
        st.success(f"### 🎓 Predicted Performance: **{prediction}**")
        st.metric("Average Score", f"{avg_score:.2f}/50")
        st.metric("Confidence", f"{max(prob)*100:.1f}%")

        prob_df = pd.DataFrame({
            "Performance Level": model.classes_,
            "Probability": prob
        }).sort_values("Probability", ascending=False)

        fig, ax = plt.subplots(figsize=(6, 3))
        sns.barplot(x="Performance Level", y="Probability", data=prob_df, palette="viridis", ax=ax)
        ax.set_ylim([0, 1])
        st.pyplot(fig)

# -------------------- FOOTER --------------------
st.markdown("""
<hr style="border: 1px solid #334155; margin-top: 3rem; margin-bottom: 1rem;">
<div style='text-align: center; color: #94a3b8; font-size: 14px;'>
    <p>🎓 Advanced Student Performance Predictor | Built with Streamlit & Scikit-learn</p>
    <p>💡 Developed by <b>Muhammad Furqan</b></p>
</div>
""", unsafe_allow_html=True)