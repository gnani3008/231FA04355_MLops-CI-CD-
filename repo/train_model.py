"""Breast Cancer classification with MLflow experiment tracking and Model Registry.

Trains six models, logs one MLflow run per model, registers the best model as
Version 1, then changes one hyperparameter, retrains and registers Version 2.
"""
import json
import os

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from mlflow.tracking import MlflowClient
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

EXPERIMENT_NAME = "Breast_Cancer_Experiments"
REGISTERED_MODEL_NAME = "Breast_Cancer_Best_Model"
N_FEATURES = 5
RANDOM_STATE = 42


def load_data():
    """Load the Breast Cancer Wisconsin dataset as a DataFrame with a 'target' column."""
    data = load_breast_cancer(as_frame=True)
    return data.frame


def select_features(df, n=N_FEATURES):
    """Pick the n features with the highest absolute correlation with the target."""
    corr = df.corr()["target"].drop("target").abs().sort_values(ascending=False)
    return corr.head(n).index.tolist()


def split_and_scale(df, features):
    X = df[features].values
    y = df["target"].values
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )
    scaler = StandardScaler()
    return (X_train, X_test, y_train, y_test,
            scaler.fit_transform(X_train), scaler.transform(X_test))


def get_model_configs():
    """Six models: (estimator, needs_scaling, params to log)."""
    return {
        "Logistic Regression": (LogisticRegression(max_iter=1000, C=1.0, random_state=RANDOM_STATE), True, {"C": 1.0, "max_iter": 1000}),
        "Decision Tree": (DecisionTreeClassifier(max_depth=5, random_state=RANDOM_STATE), False, {"max_depth": 5}),
        "Random Forest": (RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE), False, {"n_estimators": 100}),
        "K-Nearest Neighbors": (KNeighborsClassifier(n_neighbors=5), True, {"n_neighbors": 5}),
        "Support Vector Machine": (SVC(C=1.0, kernel="rbf", probability=True, random_state=RANDOM_STATE), True, {"C": 1.0, "kernel": "rbf"}),
        "Gradient Boosting": (GradientBoostingClassifier(n_estimators=100, random_state=RANDOM_STATE), False, {"n_estimators": 100}),
    }


def evaluate(model, X_test, y_test):
    preds = model.predict(X_test)
    return {
        "accuracy": accuracy_score(y_test, preds),
        "precision": precision_score(y_test, preds, average="weighted", zero_division=0),
        "recall": recall_score(y_test, preds, average="weighted", zero_division=0),
        "f1_score": f1_score(y_test, preds, average="weighted", zero_division=0),
    }


def log_run(run_name, model, scaled, params, data):
    """Fit a model, log params/metrics/model to MLflow, return (run_id, metrics)."""
    X_train, X_test, y_train, y_test, X_train_s, X_test_s = data
    Xtr, Xte = (X_train_s, X_test_s) if scaled else (X_train, X_test)
    with mlflow.start_run(run_name=run_name) as run:
        model.fit(Xtr, y_train)
        metrics = evaluate(model, Xte, y_test)
        mlflow.log_param("model_name", run_name)
        for k, v in params.items():
            mlflow.log_param(k, v)
        for k, v in metrics.items():
            mlflow.log_metric(k, v)
        # cloudpickle avoids skops "untrusted type" errors (e.g. KNN, tree models)
        mlflow.sklearn.log_model(model, name="model", serialization_format="cloudpickle")
        return run.info.run_id, metrics


def run_pipeline(tracking_uri="file:./mlruns", results_path="results.json"):
    os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(EXPERIMENT_NAME)

    df = load_data()
    features = select_features(df)
    data = split_and_scale(df, features)

    # --- Six runs, one per model ---
    rows = []
    configs = get_model_configs()
    for name, (model, scaled, params) in configs.items():
        run_id, m = log_run(name, model, scaled, params, data)
        rows.append({"model": name, "run_id": run_id, **m})
        print(f"{name:25s} accuracy={m['accuracy']:.4f} f1={m['f1_score']:.4f}")
    results = pd.DataFrame(rows).sort_values("accuracy", ascending=False).reset_index(drop=True)
    best = results.iloc[0]

    # --- Registry Version 1 ---
    v1 = mlflow.register_model(f"runs:/{best['run_id']}/model", REGISTERED_MODEL_NAME)

    # --- Change one hyperparameter (n_estimators / C / ... depends on best model) ---
    _, scaled, params = configs[best["model"]]
    changes = {
        "Logistic Regression": ("C", 0.5),
        "Decision Tree": ("max_depth", 8),
        "Random Forest": ("n_estimators", 200),
        "K-Nearest Neighbors": ("n_neighbors", 7),
        "Support Vector Machine": ("C", 10.0),
        "Gradient Boosting": ("n_estimators", 150),
    }
    pname, new_val = changes[best["model"]]
    new_params = {**params, pname: new_val}
    builders = {
        "Logistic Regression": lambda p: LogisticRegression(max_iter=p["max_iter"], C=p["C"], random_state=RANDOM_STATE),
        "Decision Tree": lambda p: DecisionTreeClassifier(max_depth=p["max_depth"], random_state=RANDOM_STATE),
        "Random Forest": lambda p: RandomForestClassifier(n_estimators=p["n_estimators"], random_state=RANDOM_STATE),
        "K-Nearest Neighbors": lambda p: KNeighborsClassifier(n_neighbors=p["n_neighbors"]),
        "Support Vector Machine": lambda p: SVC(C=p["C"], kernel=p["kernel"], probability=True, random_state=RANDOM_STATE),
        "Gradient Boosting": lambda p: GradientBoostingClassifier(n_estimators=p["n_estimators"], random_state=RANDOM_STATE),
    }
    v2_run_id, v2_metrics = log_run(f"{best['model']}_v2", builders[best["model"]](new_params), scaled, new_params, data)
    v2 = mlflow.register_model(f"runs:/{v2_run_id}/model", REGISTERED_MODEL_NAME)

    summary = {
        "features": features,
        "best_model": best["model"],
        "best_metrics": {k: float(best[k]) for k in ("accuracy", "precision", "recall", "f1_score")},
        "changed_hyperparameter": f"{pname}: {params[pname]} -> {new_val}",
        "version_1": v1.version,
        "version_2": v2.version,
        "version_2_metrics": {k: float(v) for k, v in v2_metrics.items()},
        "all_models": results[["model", "accuracy", "precision", "recall", "f1_score"]].to_dict("records"),
    }
    with open(results_path, "w") as f:
        json.dump(summary, f, indent=2)
    print("\nBest model:", summary["best_model"], "| Registry versions:", v1.version, v2.version)
    return summary


if __name__ == "__main__":
    run_pipeline()
