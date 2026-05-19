import numpy as np
import pandas as pd
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# ------------------------------------------------------------
# 1. Create or load dataset
# ------------------------------------------------------------
DATA_PATH = "data/ur3_cobotops.csv"
os.makedirs("data", exist_ok=True)
os.makedirs("models", exist_ok=True)
os.makedirs("results", exist_ok=True)

if not os.path.exists(DATA_PATH):
    print("Dataset not found – generating synthetic dataset...")
    np.random.seed(42)
    n_samples = 5000
    normal_data = {
        "current_j1": np.random.normal(2.10, 0.15, n_samples),
        "current_j2": np.random.normal(2.40, 0.18, n_samples),
        "current_j3": np.random.normal(1.80, 0.12, n_samples),
        "temperature_j1": np.random.normal(45.0, 2.0, n_samples),
        "temperature_j2": np.random.normal(47.0, 2.5, n_samples),
        "vibration": np.random.normal(0.05, 0.01, n_samples),
        "anomaly": 0
    }
    n_anom = 500
    anom_data = {
        "current_j1": np.random.normal(3.20, 0.40, n_anom),
        "current_j2": np.random.normal(3.60, 0.50, n_anom),
        "current_j3": np.random.normal(2.80, 0.35, n_anom),
        "temperature_j1": np.random.normal(62.0, 5.0, n_anom),
        "temperature_j2": np.random.normal(65.0, 6.0, n_anom),
        "vibration": np.random.normal(0.18, 0.05, n_anom),
        "anomaly": 1
    }
    df_normal = pd.DataFrame(normal_data)
    df_anom = pd.DataFrame(anom_data)
    df = pd.concat([df_normal, df_anom], ignore_index=True)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    df.to_csv(DATA_PATH, index=False)
    print(f"Synthetic dataset saved to {DATA_PATH}")
else:
    df = pd.read_csv(DATA_PATH)
    print(f"Loaded dataset from {DATA_PATH}")

print(f"Dataset shape: {df.shape}")

# ------------------------------------------------------------
# 2. Feature Engineering – rolling statistics
# ------------------------------------------------------------
BASE_COLS = ["current_j1", "current_j2", "current_j3",
             "temperature_j1", "temperature_j2", "vibration"]

# Create rolling mean and std for each base column (window = 5)
df_sorted = df.sort_index()  # already shuffled, but ensure order
for col in BASE_COLS:
    df_sorted[f"{col}_roll_mean"] = df_sorted[col].rolling(window=5, min_periods=1).mean()
    df_sorted[f"{col}_roll_std"]  = df_sorted[col].rolling(window=5, min_periods=1).std().fillna(0)

# Drop the first 4 rows where rolling might be incomplete? Keep all, min_periods=1 handles it.
X = df_sorted.drop(columns=["anomaly"])
y = df_sorted["anomaly"]

# Keep feature names for later
feature_names = X.columns.tolist()
print(f"Total features after engineering: {len(feature_names)}")

# ------------------------------------------------------------
# 3. Train / test split (stratified)
# ------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ------------------------------------------------------------
# 4. Normalize features
# ------------------------------------------------------------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ------------------------------------------------------------
# 5. Train Random Forest Classifier
# ------------------------------------------------------------
print("Training Random Forest...")
rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
rf.fit(X_train_scaled, y_train)

# ------------------------------------------------------------
# 6. Evaluate
# ------------------------------------------------------------
y_pred = rf.predict(X_test_scaled)
y_proba = rf.predict_proba(X_test_scaled)[:, 1]

print("\n=== Classification Report ===")
print(classification_report(y_test, y_pred, target_names=["Normal", "Anomaly"]))
print(f"ROC AUC: {roc_auc_score(y_test, y_proba):.4f}")

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=["Normal", "Anomaly"],
            yticklabels=["Normal", "Anomaly"])
plt.title("Confusion Matrix")
plt.ylabel("True")
plt.xlabel("Predicted")
plt.savefig("results/confusion_matrix.png", dpi=150)
plt.close()
print("Confusion matrix saved to results/confusion_matrix.png")

# Feature importance
importances = rf.feature_importances_
indices = np.argsort(importances)[::-1][:15]  # top 15
plt.figure(figsize=(8,6))
plt.title("Top 15 Feature Importances")
plt.barh(range(len(indices)), importances[indices], align="center")
plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
plt.gca().invert_yaxis()
plt.xlabel("Relative Importance")
plt.tight_layout()
plt.savefig("results/feature_importance.png", dpi=150)
plt.close()
print("Feature importance plot saved to results/feature_importance.png")

# ------------------------------------------------------------
# 7. Save model, scaler, and feature names
# ------------------------------------------------------------
joblib.dump(rf, "models/rf_model.pkl")
joblib.dump(scaler, "models/scaler.pkl")
joblib.dump(feature_names, "models/feature_names.pkl")
print("\nModel artefacts saved to models/")
print("Training complete.")