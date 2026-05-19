"""
train.py  –  UR3 Predictive Maintenance | Random Forest Classifier
Run:  python src/train.py
"""

import os
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    roc_auc_score, confusion_matrix, classification_report,
)

# ── Config ────────────────────────────────────────────────────────────────────
DATA_PATH    = "data/ur3_cobotops.csv"
MODEL_DIR    = "models"
RESULTS_DIR  = "results"
WINDOW       = 5        # rolling feature window
N_ESTIMATORS = 200
RANDOM_STATE = 42

BASE_FEATURES = [
    "current_j1", "current_j2", "current_j3",
    "temperature_j1", "temperature_j2",
    "vibration",
]

# ── 1. Data ───────────────────────────────────────────────────────────────────

def load_or_generate(path: str) -> pd.DataFrame:
    """Load CSV if it exists; otherwise synthesise realistic UR3 sensor data."""
    if os.path.exists(path):
        print(f"[INFO] Loading dataset from {path}")
        return pd.read_csv(path)

    print("[INFO] Dataset not found – generating synthetic UR3 sensor data …")
    rng = np.random.default_rng(RANDOM_STATE)
    n, ratio = 5_000, 0.85
    n_normal  = int(n * ratio)
    n_anomaly = n - n_normal

    normal = {
        "current_j1":    rng.normal(2.10, 0.15, n_normal),
        "current_j2":    rng.normal(2.40, 0.18, n_normal),
        "current_j3":    rng.normal(1.80, 0.12, n_normal),
        "temperature_j1":rng.normal(45.0,  2.0, n_normal),
        "temperature_j2":rng.normal(47.0,  2.5, n_normal),
        "vibration":     rng.normal(0.05, 0.01, n_normal),
        "anomaly":       np.zeros(n_normal, dtype=int),
    }
    anomaly = {
        "current_j1":    rng.normal(3.20, 0.40, n_anomaly),
        "current_j2":    rng.normal(3.60, 0.50, n_anomaly),
        "current_j3":    rng.normal(2.80, 0.35, n_anomaly),
        "temperature_j1":rng.normal(62.0,  5.0, n_anomaly),
        "temperature_j2":rng.normal(65.0,  6.0, n_anomaly),
        "vibration":     rng.normal(0.18, 0.05, n_anomaly),
        "anomaly":       np.ones(n_anomaly, dtype=int),
    }

    df = (
        pd.concat([pd.DataFrame(normal), pd.DataFrame(anomaly)], ignore_index=True)
        .sample(frac=1, random_state=RANDOM_STATE)
        .reset_index(drop=True)
    )
    os.makedirs("data", exist_ok=True)
    df.to_csv(path, index=False)
    print(f"[INFO] Synthetic data saved → {path}  ({len(df):,} rows, "
          f"{n_anomaly} anomalies = {n_anomaly/n:.0%})")
    return df


# ── 2. Feature engineering ────────────────────────────────────────────────────

def add_rolling_features(df: pd.DataFrame, window: int = WINDOW) -> pd.DataFrame:
    """Append rolling mean and std for every base sensor column."""
    df = df.copy()
    for col in BASE_FEATURES:
        df[f"{col}_roll_mean"] = df[col].rolling(window, min_periods=1).mean()
        df[f"{col}_roll_std"]  = df[col].rolling(window, min_periods=1).std().fillna(0)
    return df


# ── 3. Training & evaluation ─────────────────────────────────────────────────

def train():
    os.makedirs(MODEL_DIR,   exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # --- Load & engineer ---
    df = load_or_generate(DATA_PATH)
    df = add_rolling_features(df)

    X = df.drop(columns=["anomaly"])
    y = df["anomaly"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    # --- Scale ---
    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc  = scaler.transform(X_test)

    # --- Fit ---
    print("[INFO] Training Random Forest …")
    clf = RandomForestClassifier(
        n_estimators=N_ESTIMATORS,
        max_depth=None,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    clf.fit(X_train_sc, y_train)

    # --- Evaluate ---
    y_pred = clf.predict(X_test_sc)
    y_prob = clf.predict_proba(X_test_sc)[:, 1]

    print("\n" + "=" * 40)
    print("  EVALUATION RESULTS")
    print("=" * 40)
    print(f"  Accuracy  : {accuracy_score(y_test, y_pred):.4f}")
    print(f"  Precision : {precision_score(y_test, y_pred):.4f}")
    print(f"  Recall    : {recall_score(y_test, y_pred):.4f}")
    print(f"  ROC AUC   : {roc_auc_score(y_test, y_prob):.4f}")
    print("=" * 40)
    print(classification_report(y_test, y_pred, target_names=["Normal", "Anomaly"]))

    # --- Save artefacts ---
    joblib.dump(clf,              f"{MODEL_DIR}/rf_model.pkl")
    joblib.dump(scaler,           f"{MODEL_DIR}/scaler.pkl")
    joblib.dump(list(X.columns),  f"{MODEL_DIR}/feature_names.pkl")
    print("[INFO] Model artefacts saved → models/")

    # --- Plots ---
    _plot_confusion_matrix(y_test, y_pred)
    _plot_feature_importance(clf, X.columns)
    print(f"[INFO] Plots saved → {RESULTS_DIR}/")


# ── 4. Plots ──────────────────────────────────────────────────────────────────

def _plot_confusion_matrix(y_test, y_pred):
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues", ax=ax,
        xticklabels=["Normal", "Anomaly"],
        yticklabels=["Normal", "Anomaly"],
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix – UR3 Anomaly Detection")
    plt.tight_layout()
    plt.savefig(f"{RESULTS_DIR}/confusion_matrix.png", dpi=150)
    plt.close()


def _plot_feature_importance(clf, feature_names, top_n: int = 15):
    importances = pd.Series(clf.feature_importances_, index=feature_names)
    top = importances.sort_values(ascending=False).head(top_n)

    fig, ax = plt.subplots(figsize=(7, 5))
    top.plot(kind="barh", ax=ax, color="steelblue")
    ax.invert_yaxis()
    ax.set_title(f"Top {top_n} Feature Importances")
    ax.set_xlabel("Mean Decrease in Impurity")
    plt.tight_layout()
    plt.savefig(f"{RESULTS_DIR}/feature_importance.png", dpi=150)
    plt.close()


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    train()
