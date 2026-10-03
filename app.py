import os, joblib
import streamlit as st
import numpy as np
import pandas as pd

st.set_page_config(page_title="Heart Disease Prediction", layout="wide")

st.title("Heart Disease Prediction App")

MODELS_DIR = "models"

@st.cache_resource
def load_assets():
    preprocessor = joblib.load(os.path.join(MODELS_DIR, "preprocessor.pkl"))
    models = {
        "Logistic Regression": joblib.load(os.path.join(MODELS_DIR, "model_logistic_regression.pkl")),
        "KNN": joblib.load(os.path.join(MODELS_DIR, "model_knn.pkl")),
        "Decision Tree": joblib.load(os.path.join(MODELS_DIR, "model_decision_tree.pkl")),
        "Random Forest": joblib.load(os.path.join(MODELS_DIR, "model_random_forest.pkl")),
        "SVM": joblib.load(os.path.join(MODELS_DIR, "model_svm.pkl")),
    }
    return preprocessor, models

preprocessor, models = load_assets()

## Sidebar
st.sidebar.header("Choose the model")
selected_model_name = st.sidebar.selectbox("Select a model", list(models.keys()))


# ---------------------------------------------------------
# Main Page: Patient Data Inputs
# ---------------------------------------------------------
st.header("Patient Data Input")

col1, col2, col3 = st.columns(3)

with col1:
    age = st.number_input("Age", min_value=1, max_value=120, value=50)
    sex = st.selectbox("Sex", options=[1, 0], format_func=lambda x: "Male" if x == 1 else "Female")
    cp = st.selectbox("Chest Pain Type (cp)", options=["1", "2", "3", "4"])
    trestbps = st.number_input("Resting Blood Pressure (mm Hg)", min_value=50, max_value=250, value=120)

with col2:
    chol = st.number_input("Serum Cholesterol (mg/dl)", min_value=100, max_value=600, value=200)
    fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl (fbs)", options=[0, 1])
    restecg = st.selectbox("Resting ECG Results", options=["0", "1", "2"])
    thalach = st.number_input("Max Heart Rate Achieved", min_value=50, max_value=250, value=150)

with col3:
    exang = st.selectbox("Exercise Induced Angina", options=[0, 1])
    oldpeak = st.number_input("ST Depression (oldpeak)", min_value=0.0, max_value=10.0, value=1.0)
    slope = st.selectbox("Slope of Peak Exercise ST", options=["1", "2", "3"])
    ca = st.number_input("Major Vessels Colored by Fluoroscopy (ca)", min_value=0, max_value=4, value=0)
    thal = st.selectbox("Thallium Stress Test (thal)", options=["3", "6", "7"])

# Construct raw input DataFrame
input_data = pd.DataFrame([{
    "age": age, "sex": sex, "cp": str(cp), "trestbps": trestbps, "chol": chol,
    "fbs": fbs, "restecg": str(restecg), "thalach": thalach, "exang": exang,
    "oldpeak": oldpeak, "slope": str(slope), "ca": ca, "thal": str(thal)
}])

st.markdown("-----")

if st.button("Predict", type="primary"):

    # Preprocess the input data
    input_preprocessed = preprocessor.transform(input_data)

    # Run prediction with selected model
    model = models[selected_model_name]
    prediction = model.predict(input_preprocessed)[0]
    probability =model.predict_proba(input_preprocessed)[0][1]  # Probability of class 1 (disease present)

    # Display the prediction result
    st.subheader("Prediction Result")
    if prediction == 1:
        st.error(f"Heart disease: Probability of disease is {probability:.2f}%.")
    else:
        st.success(f"No heart disease. Probability of disease is {1 - probability:.2f}%.")


    # Display inspection details at the bottom
    st.markdown("---")
    st.subheader("Data Inspection & Pipeline Diagnostics")
    
    exp1, exp2 = st.tabs(["Raw Received Data", "Preprocessed Array Sent to Model"])
    
    with exp1:
        st.write("This is the exact DataFrame constructed from your inputs:")
        st.dataframe(input_data)
        
    with exp2:
        st.write("This is the scaled and One-Hot Encoded feature matrix fed directly to the model:")
        # Attempt to retrieve encoded feature names if available from ColumnTransformer
        try:
            feature_names = preprocessor.get_feature_names_out()
            preprocessed_df = pd.DataFrame(input_preprocessed, columns=feature_names)
            st.dataframe(preprocessed_df)
        except Exception:
            st.write(input_preprocessed)