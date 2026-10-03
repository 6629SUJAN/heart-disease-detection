import os, joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os 
from ucimlrepo import fetch_ucirepo

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC




from ucimlrepo import fetch_ucirepo







heart_disease = fetch_ucirepo(id=45)

X = heart_disease.data.features
y = heart_disease.data.targets.rename(
    columns={"num":"target"}
)
y["target"] = (y["target"]>0).astype(int)
print(heart_disease.metadata)







X.isna().sum()

X = X.dropna()



X["thal"] = X["thal"].astype(int)
y = y.loc[X.index]



df = pd.concat([X, y], axis=1)
df.head()

# Plot target variable distribution
plt.figure(figsize=(12, 12))

# Age vs Heart Disease
plt.subplot(3, 3, 1)
sns.histplot(x='age', hue='target', data=df, multiple="stack")
plt.title("Age")
plt.xlabel("Age")
plt.ylabel("Count")

# Sex vs Heart Disease
plt.subplot(3, 3, 2)
sns.histplot(x='sex', hue='target', data=df, multiple="stack")
plt.title("Sex")
plt.xlabel("Sex (0: Female, 1: Male)")
plt.ylabel("Count")

# Chest Pain vs Heart Disease
plt.subplot(3, 3, 3)
sns.countplot(x='cp', hue='target', data=df)
plt.title("Chest Pain Type (cp)")
plt.xlabel("Chest Pain Type (1: Typical, 2: Atypical, 3: Non-anginal, 4: Asymptomatic)")
plt.ylabel("Count")

# Fasting Blood Sugar vs Heart Disease
plt.subplot(3, 3, 4)
sns.countplot(x='fbs', hue='target', data=df)
plt.title("Target Distribution (0 = No Disease, 1 = Disease)")
plt.xlabel("Fasting Blood Sugar > 120 mg/dl (0: No, 1: Yes)")
plt.ylabel("Count")

# Resting ECG Type vs Heart Disease
plt.subplot(3, 3, 5)
sns.countplot(x='restecg', hue='target', data=df)
plt.title("Resting ECG (restecg)")
plt.xlabel("Resting ECG (0: Normal, 1: ST-T Wave Abnormality, 2: Left Ventricular Hypertrophy)")
plt.ylabel("Count")

# Exercise-Induced Angina vs Heart Disease
plt.subplot(3, 3, 6)
sns.countplot(x='exang', hue='target', data=df)
plt.title("Exercise-Induced Angina (exang)")
plt.xlabel("Exercise-Induced Angina (0: No, 1: Yes)")
plt.ylabel("Count")

# Slope of Peak exercise ST segment vs Heart Disease
plt.subplot(3, 3, 7)
sns.countplot(x='slope', hue='target', data=df)
plt.title("Slope of Peak exercise ST Segment (slope)")
plt.xlabel("Slope of Peak exercise ST Segment (0: Upsloping, 1: Flat, 2: Downsloping)")
plt.ylabel("Count")

# Number of Major Vessels vs Heart Disease
plt.subplot(3, 3, 8)
sns.countplot(x='ca', hue='target', data=df)
plt.title("Number of Major Vessels (ca)")
plt.xlabel("Number of Major Vessels (0-3)")
plt.ylabel("Count")

# Thal Type vs Heart Disease
plt.subplot(3, 3, 9)
sns.countplot(x='thal', hue='target', data=df)
plt.title("Thal Type (thal)")
plt.xlabel("Thal Type (3: Normal, 6: Fixed Defect, 7: Reversible Defect)")
plt.ylabel("Count")

plt.tight_layout()
plt.show()





categorical_cols = [ 'cp', 'restecg','slope','thal']
X[categorical_cols] = X[categorical_cols].astype(str)

X_encoded = pd.get_dummies(X, columns=categorical_cols, drop_first=True, dtype=int)

print("Original Features Shape:", X.shape)
print("Encoded Features Shape:", X_encoded.shape)
print(X_encoded.head())


from sklearn.preprocessing import OneHotEncoder

continuous_cols = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak','ca']
binary_cols = ['sex', 'fbs', 'exang']

scaler = StandardScaler()
X_encoded[continuous_cols] = scaler.fit_transform(X_encoded[continuous_cols])

preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), continuous_cols),
        ("cat", OneHotEncoder(drop="first", sparse_output=False, handle_unknown="ignore"), categorical_cols),
        ("bin", "passthrough", binary_cols)
    ]
)
##Train Test Split and Modeling
X_train, X_test, y_train, y_test = train_test_split(X, y["target"], test_size=0.2, random_state=42)

X_train_preprocessed = preprocessor.fit_transform(X_train)

##Save preprocessor artifact
Models_dir = "models"
if not os.path.exists(Models_dir):
    os.makedirs(Models_dir)
preprocessor_path = os.path.join(Models_dir, "preprocessor.pkl")
joblib.dump(preprocessor, preprocessor_path)
print(f"Preprocessor saved to {preprocessor_path}")

## Grid Search for Hyperparameter Tuning
model_params = {
    "logistic_regression": {
        "model": LogisticRegression(max_iter=1000, random_state=42),
        "params": {"C": [0.01, 0.1, 1.0, 10.0], "solver": ["liblinear", "lbfgs"]}
    },
    "knn": {
        "model": KNeighborsClassifier(),
        "params": {"n_neighbors": [3, 5, 7, 9, 11], "weights": ["uniform", "distance"]}
    },
    "decision_tree": {
        "model": DecisionTreeClassifier(random_state=42),
        "params": {"max_depth": [3, 5, 7, 10, None], "criterion": ["gini", "entropy"]}
    },
    "random_forest": {
        "model": RandomForestClassifier(random_state=42),
        "params": {"n_estimators": [50, 100, 200], "max_depth": [3, 5, 10, None]}
    },
    "svm": {
        "model": SVC(probability=True, random_state=42),
        "params": {"C": [0.1, 1, 10], "kernel": ["linear", "rbf"]}
    }
}

# Transform test data using the fitted preprocessor
X_test_preprocessed = preprocessor.transform(X_test)


from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix, classification_report

for name, mp in model_params.items():

    clf = GridSearchCV(mp["model"], mp["params"], cv=5, scoring="accuracy", n_jobs=-1)
    clf.fit(X_train_preprocessed, y_train)
    best_model = clf.best_estimator_
    y_pred = best_model.predict(X_test_preprocessed)
    model_path = os.path.join(Models_dir, f"model_{name}.pkl")
    joblib.dump(best_model, model_path)

    print(f"Saved {name} model to '{model_path}' (CV Accuracy: {clf.best_score_:.4f})")
    print(f"Model: {name}")
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print("Classification Report:\n", classification_report(y_test, y_pred))