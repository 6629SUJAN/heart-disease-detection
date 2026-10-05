import os
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from ucimlrepo import fetch_ucirepo

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, learning_curve, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier


RANDOM_STATE = 42
PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "notebooks"


def load_data():
    heart_disease = fetch_ucirepo(id=45)
    X = heart_disease.data.features.copy()
    y = heart_disease.data.targets.rename(columns={"num": "target"}).copy()
    y["target"] = (y["target"] > 0).astype(int)

    X = X.dropna().copy()
    y = y.loc[X.index]
    X["thal"] = X["thal"].astype(int)

    categorical_cols = ["cp", "restecg", "slope", "thal"]
    continuous_cols = ["age", "trestbps", "chol", "thalach", "oldpeak", "ca"]
    binary_cols = ["sex", "fbs", "exang"]
    X[categorical_cols] = X[categorical_cols].astype(str)

    return X, y["target"], categorical_cols, continuous_cols, binary_cols


def create_eda_plot(X, y):
    df = pd.concat([X, y.rename("target")], axis=1)
    plots = [
        ("age", "Age"),
        ("sex", "Sex"),
        ("cp", "Chest Pain Type"),
        ("fbs", "Fasting Blood Sugar"),
        ("restecg", "Resting ECG"),
        ("exang", "Exercise Angina"),
        ("slope", "ST Slope"),
        ("ca", "Major Vessels"),
        ("thal", "Thal Type"),
    ]
    fig, axes = plt.subplots(3, 3, figsize=(12, 12))
    for axis, (column, title) in zip(axes.ravel(), plots):
        if column == "age":
            sns.histplot(data=df, x=column, hue="target", multiple="stack", ax=axis)
        else:
            sns.countplot(data=df, x=column, hue="target", ax=axis)
        axis.set_title(title)
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "eda_plots.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def train_models(X, y, categorical_cols, continuous_cols, binary_cols):
    preprocessor = ColumnTransformer([
        ("num", StandardScaler(), continuous_cols),
        ("cat", OneHotEncoder(drop="first", sparse_output=False, handle_unknown="ignore"), categorical_cols),
        ("bin", "passthrough", binary_cols),
    ])
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )
    X_train_preprocessed = preprocessor.fit_transform(X_train)
    X_test_preprocessed = preprocessor.transform(X_test)

    model_params = {
        "logistic_regression": (
            LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
            {"C": [0.01, 0.1, 1.0, 10.0], "solver": ["liblinear", "lbfgs"]},
        ),
        "knn": (KNeighborsClassifier(), {"n_neighbors": [3, 5, 7, 9, 11], "weights": ["uniform", "distance"]}),
        "decision_tree": (
            DecisionTreeClassifier(random_state=RANDOM_STATE),
            {"max_depth": [3, 5, 7, 10, None], "criterion": ["gini", "entropy"]},
        ),
        "random_forest": (
            RandomForestClassifier(random_state=RANDOM_STATE),
            {"n_estimators": [50, 100, 200], "max_depth": [3, 5, 10, None]},
        ),
        "svm": (SVC(probability=True, random_state=RANDOM_STATE), {"C": [0.1, 1, 10], "kernel": ["linear", "rbf"]}),
    }

    MODELS_DIR.mkdir(exist_ok=True)
    joblib.dump(preprocessor, MODELS_DIR / "preprocessor.pkl")
    models = {}
    for name, (model, params) in model_params.items():
        search = GridSearchCV(model, params, cv=5, scoring="accuracy", n_jobs=-1)
        search.fit(X_train_preprocessed, y_train)
        best_model = search.best_estimator_
        models[name] = best_model
        joblib.dump(best_model, MODELS_DIR / f"model_{name}.pkl")
        y_pred = best_model.predict(X_test_preprocessed)
        print(f"\n=== {name} | CV accuracy: {search.best_score_:.4f} | best: {search.best_params_}")
        print(confusion_matrix(y_test, y_pred))
        print(classification_report(y_test, y_pred))

    return models, preprocessor, X_train, X_test, y_train, y_test, X_train_preprocessed, X_test_preprocessed


def evaluate_models(models, X_test_preprocessed, y_test):
    results = []
    confusion_results = []
    for name, model in models.items():
        y_pred = model.predict(X_test_preprocessed)
        y_prob = model.predict_proba(X_test_preprocessed)[:, 1]
        results.append({
            "Model": name.replace("_", " ").title(),
            "Accuracy": accuracy_score(y_test, y_pred),
            "Precision": precision_score(y_test, y_pred),
            "Recall": recall_score(y_test, y_pred),
            "F1-Score": f1_score(y_test, y_pred),
            "ROC-AUC": roc_auc_score(y_test, y_prob),
        })
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
        confusion_results.append({"Model": name, "TN": tn, "FP": fp, "FN": fn, "TP": tp})

    results_df = pd.DataFrame(results).round(4)
    comparison_df = results_df.copy()
    metric_columns = ["Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"]
    comparison_df[metric_columns] = (comparison_df[metric_columns] * 100).round(2)
    cm_df = pd.DataFrame(confusion_results)
    comparison_df.to_csv(RESULTS_DIR / "model_comparison_results.csv", index=False)
    cm_df.to_csv(RESULTS_DIR / "confusion_matrix_results.csv", index=False)

    print("\nModel comparison (percent):")
    print(comparison_df.to_string(index=False))
    for metric in ["Accuracy", "F1-Score", "ROC-AUC"]:
        best = results_df.loc[results_df[metric].idxmax()]
        print(f"Best {metric}: {best['Model']} ({best[metric]:.4f})")

    return results_df


def create_evaluation_plots(models, X_test_preprocessed, y_test, X_train_preprocessed, y_train, results_df):
    metric_columns = ["Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"]
    results_df.set_index("Model")[metric_columns].plot(kind="bar", figsize=(12, 6), ylim=(0, 1))
    plt.title("Performance Comparison of Machine Learning Models")
    plt.ylabel("Score")
    plt.xticks(rotation=20)
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "model_comparison.png", dpi=300, bbox_inches="tight")
    plt.close()

    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    for axis, (name, model) in zip(axes.ravel(), models.items()):
        ConfusionMatrixDisplay.from_predictions(
            y_test, model.predict(X_test_preprocessed), display_labels=["No Disease", "Disease"], values_format="d", ax=axis
        )
        axis.set_title(name.replace("_", " ").title())
    axes.ravel()[-1].axis("off")
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "confusion_matrices.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    learning_models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
        "KNN": KNeighborsClassifier(),
        "Decision Tree": DecisionTreeClassifier(random_state=RANDOM_STATE),
        "Random Forest": RandomForestClassifier(random_state=RANDOM_STATE),
        "SVM": SVC(probability=True, random_state=RANDOM_STATE),
    }
    fig, axis = plt.subplots(figsize=(10, 6))
    for name, model in learning_models.items():
        train_sizes, _, validation_scores = learning_curve(
            model, X_train_preprocessed, y_train, cv=5, scoring="accuracy", train_sizes=np.linspace(0.1, 1.0, 5), n_jobs=-1
        )
        axis.plot(train_sizes, validation_scores.mean(axis=1), marker="o", label=name)
    axis.set_title("Validation Learning Curves of Machine Learning Models")
    axis.set_xlabel("Training Set Size")
    axis.set_ylabel("Validation Accuracy")
    axis.set_ylim(0.4, 1.05)
    axis.legend()
    axis.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "combined_learning_curves.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    RESULTS_DIR.mkdir(exist_ok=True)
    X, y, categorical_cols, continuous_cols, binary_cols = load_data()
    create_eda_plot(X, y)
    models, _, X_train, X_test, y_train, y_test, X_train_p, X_test_p = train_models(
        X, y, categorical_cols, continuous_cols, binary_cols
    )
    results_df = evaluate_models(models, X_test_p, y_test)
    create_evaluation_plots(models, X_test_p, y_test, X_train_p, y_train, results_df)
    print("\nResults saved successfully.")


if __name__ == "__main__":
    main()