import os, joblib
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Heart Disease Prediction", page_icon="❤️", layout="wide")

MODELS_DIR = "models"
MODEL_FILES = {
    "Logistic Regression": "model_logistic_regression.pkl",
    "KNN": "model_knn.pkl",
    "Decision Tree": "model_decision_tree.pkl",
    "Random Forest": "model_random_forest.pkl",
    "SVM": "model_svm.pkl",
}

@st.cache_resource
def load_assets():
    pre = joblib.load(os.path.join(MODELS_DIR, "preprocessor.pkl"))
    models = {n: joblib.load(os.path.join(MODELS_DIR, f)) for n, f in MODEL_FILES.items()}
    return pre, models

try:
    preprocessor, models = load_assets()
except FileNotFoundError:
    st.error("Model files not found. Run `train.py` first to create the `models/` folder.")
    st.stop()

# ---------- Friendly labels ----------
CP = {"1": "Typical angina", "2": "Atypical angina", "3": "Non-anginal pain", "4": "Asymptomatic"}
ECG = {"0": "Normal", "1": "ST-T wave abnormality", "2": "Left ventricular hypertrophy"}
SLOPE = {"1": "Upsloping", "2": "Flat", "3": "Downsloping"}
THAL = {"3": "Normal", "6": "Fixed defect", "7": "Reversible defect"}
YES_NO = {0: "No", 1: "Yes"}

# ---------- Sidebar ----------
st.sidebar.title("Settings")
model_name = st.sidebar.selectbox("Choose a model", list(models.keys()))
st.sidebar.info("This app is for **educational use only** and is not a medical diagnosis.")

# ---------- Header ----------
st.title("❤️ Heart Disease Prediction")
st.caption("Fill in the patient details below, then click Predict.")

# ---------- Inputs ----------
with st.form("patient_form"):
    st.subheader("Patient Details")
    c1, c2, c3 = st.columns(3)

    with c1:
        st.markdown("**Basic Info**")
        age = st.number_input("Age", 1, 120, 50)
        sex = st.radio("Sex", [1, 0], format_func=lambda x: "Male" if x else "Female", horizontal=True)
        trestbps = st.number_input("Resting Blood Pressure (mm Hg)", 50, 250, 120)
        chol = st.number_input("Cholesterol (mg/dl)", 100, 600, 200)
        fbs = st.radio("Fasting Blood Sugar > 120 mg/dl", [0, 1], format_func=YES_NO.get, horizontal=True)

    with c2:
        st.markdown(" Symptoms & ECG")
        cp = st.selectbox("Chest Pain Type", list(CP), format_func=CP.get)
        restecg = st.selectbox("Resting ECG", list(ECG), format_func=ECG.get)
        exang = st.radio("Exercise-Induced Angina", [0, 1], format_func=YES_NO.get, horizontal=True)
        thalach = st.number_input("Max Heart Rate Achieved", 50, 250, 150)

    with c3:
        st.markdown(" Stress Test")
        oldpeak = st.number_input("ST Depression (oldpeak)", 0.0, 10.0, 1.0, step=0.1)
        slope = st.selectbox("Slope of Peak Exercise ST", list(SLOPE), format_func=SLOPE.get)
        ca = st.slider("Major Vessels Colored (ca)", 0, 3, 0)
        thal = st.selectbox("Thallium Stress Test", list(THAL), format_func=THAL.get)

    submitted = st.form_submit_button(" Predict", type="primary", use_container_width=True)

# ---------- Prediction ----------
if submitted:
    input_data = pd.DataFrame([{
        "age": age, "sex": sex, "cp": cp, "trestbps": trestbps, "chol": chol,
        "fbs": fbs, "restecg": restecg, "thalach": thalach, "exang": exang,
        "oldpeak": oldpeak, "slope": slope, "ca": ca, "thal": thal,
    }])
    X_in = preprocessor.transform(input_data)

    model = models[model_name]
    pred = model.predict(X_in)[0]
    prob = model.predict_proba(X_in)[0][1]  # probability of disease

    st.markdown("---")
    tab1, tab2, tab3 = st.tabs([" Result", " Compare Models", "Data Inspection"])

    with tab1:
        if pred == 1:
            st.error(f"Heart disease likely ({model_name})")
        else:
            st.success(f"No heart disease detected ({model_name})")

        m1, m2 = st.columns(2)
        m1.metric("Disease probability", f"{prob * 100:.1f}%")
        m2.metric("No-disease probability", f"{(1 - prob) * 100:.1f}%")
        st.progress(float(prob), text="Risk level")

    with tab2:
        rows = []
        for n, m in models.items():
            p = m.predict_proba(X_in)[0][1]
            rows.append({
                "Model": n,
                "Prediction": "Disease" if p >= 0.5 else "No disease",
                "Disease probability (%)": round(p * 100, 1),
            })
        st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
        st.caption("If the models disagree, treat the result with extra caution.")

    with tab3:
        st.write("Raw input:")
        st.dataframe(input_data, hide_index=True)
        st.write("Preprocessed (scaled + encoded) input:")
        try:
            cols = preprocessor.get_feature_names_out()
            st.dataframe(pd.DataFrame(X_in, columns=cols), hide_index=True)
        except Exception:
            st.write(X_in)