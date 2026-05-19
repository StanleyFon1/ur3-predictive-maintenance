# 🤖 AI-Based Predictive Maintenance for UR3 Robotic Arm

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3-orange?logo=scikit-learn)
![Streamlit](https://img.shields.io/badge/Streamlit-1.32-red?logo=streamlit)
![License](https://img.shields.io/badge/License-MIT-green)

A machine-learning pipeline that detects anomalies in a **Universal Robots UR3** collaborative robot arm using sensor data, paired with a **real-time Streamlit dashboard** for live health monitoring.

> **Industry 4.0 context:** Unplanned downtime costs manufacturers an estimated $260,000 per hour (Aberdeen Research). This project demonstrates how predictive maintenance detecting faults *before* failure can be achieved with ML on readily available sensor data streams.

---

## 📋 Table of Contents
- [Overview](#overview)
- [Dataset](#dataset)
- [Methodology](#methodology)
- [Results](#results)
- [Dashboard Demo](#dashboard-demo)
- [How to Run](#how-to-run)
- [Repository Structure](#repository-structure)
- [Future Work](#future-work)
- [Author](#author)

---

## Overview

The system monitors six sensor channels from the UR3 arm (joint currents, joint temperatures, vibration) and classifies each reading as **Normal** or **Anomalous** using a Random Forest classifier. Rolling statistical features are engineered to capture temporal patterns across a sliding window.

**Key features:**
- Automated data generation (synthetic UR3 sensor data) with no external dependency
- Rolling-window feature engineering for temporal context
- Full evaluation suite: accuracy, precision, recall, ROC AUC, confusion matrix
- Interactive Streamlit dashboard with live anomaly score, health index, and per-sensor charts
- Anomaly injection toggle for live demonstration

---

## Dataset

**Source:** UR3 CobotOps — sensor logs from a Universal Robots UR3 arm  
**Features:** `current_j1`, `current_j2`, `current_j3`, `temperature_j1`, `temperature_j2`, `vibration`  
**Target:** `anomaly` — binary label (0 = Normal, 1 = Anomaly)

See [`data/README.md`](data/README.md) for download instructions and feature statistics.  
If no CSV is found, `train.py` auto-generates a statistically realistic synthetic dataset.

---

## Methodology

```
Raw Sensor Data
     │
     ▼
Data Loading & Validation
     │
     ▼
Rolling Feature Engineering  ←── window = 5 timesteps
(roll_mean, roll_std per channel)
     │
     ▼
Train / Test Split  (80 / 20, stratified)
     │
     ▼
StandardScaler normalisation
     │
     ▼
Random Forest Classifier  (200 trees, n_jobs=-1)
     │
     ▼
Evaluation: Accuracy · Precision · Recall · ROC AUC
     │
     ▼
Model Persistence (joblib)  →  models/
```

---

## Results

| Metric    | Score |
|-----------|-------|
| Accuracy  | 92%   |
| Precision | 91%   |
| Recall    | 89%   |
| ROC AUC   | 0.96  |

> *Note: Results shown above are from the synthetic dataset. Performance can or will vary with real UR3 sensor data.*

**Confusion Matrix**

![Confusion Matrix](results/confusion_matrix.png)

**Top 15 Feature Importances**

![Feature Importance](results/feature_importance.png)

---

## Dashboard Demo

> 🎥 **[YouTube Demo Video](#)** *(## Dashboard Demo

[![Watch the demo](https://img.youtube.com/vi/-f97VxfJFSA/0.jpg)](https://youtu.be/-f97VxfJFSA)

🎥 [Click here to watch the demo video on YouTube](https://youtu.be/-f97VxfJFSA))*

The Streamlit dashboard provides:
- Live anomaly score time-series with configurable threshold line
- Per-joint current and temperature charts
- Vibration waveform with fill
- KPI row: total readings, anomalies detected, latest score, health index (%)
- Anomaly injection toggle for live demonstration
- Adjustable update interval and history window

---

## How to Run

### Prerequisites
- Python ≥ 3.10
- Git

### 1 — Clone the repository
```bash
git clone https://github.com/StanleyFon1/ur3-predictive-maintenance.git
cd ur3-predictive-maintenance
```

### 2 — Create a virtual environment (recommended)
```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
```

### 3 — Install dependencies
```bash
pip install -r requirements.txt
```

### 4 — Train the model
```bash
python src/train.py
```
This generates `data/ur3_cobotops.csv` (if absent), trains the classifier, prints evaluation metrics, and saves:
- `models/rf_model.pkl`
- `models/scaler.pkl`
- `models/feature_names.pkl`
- `results/confusion_matrix.png`
- `results/feature_importance.png`

### 5 — Launch the dashboard
```bash
streamlit run dashboard/app.py
```
Open **http://localhost:8501** in your browser, then press **▶ Start** in the sidebar.

---

## Repository Structure

```
ur3-predictive-maintenance/
├── data/
│   └── README.md            # Dataset source & download instructions
├── notebooks/
│   └── 01_eda.ipynb         # Exploratory data analysis (coming soon)
├── src/
│   └── train.py             # Full training pipeline
├── dashboard/
│   └── app.py               # Streamlit real-time monitoring app
├── models/                  # Saved model artefacts (auto-generated)
├── results/                 # Evaluation plots (auto-generated)
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Future Work

- [ ] **Real-time streaming** – connect directly to a physical UR3 via RTDE (Real-Time Data Exchange)
- [ ] **LSTM / Transformer** – sequence models to exploit longer temporal dependencies
- [ ] **Containerisation** – Docker image for one-command deployment
- [ ] **Cloud deployment** – host dashboard on Streamlit Cloud or AWS
- [ ] **Alert system** – email / SMS notification on anomaly threshold breach

---

## License

This project is licensed under the **MIT License** see [LICENSE](LICENSE) for details.

---

## Author

**Stanley Fon**  
M.Sc. Mechatronics & Cyber-Physical Systems. Deggendorf Institute of Technology

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Stanley%20Fon-blue?logo=linkedin)](https://www.linkedin.com/in/stanley-fon-56bb4420b)
[![GitHub](https://img.shields.io/badge/GitHub-StanleyFon1-black?logo=github)](https://github.com/StanleyFon1)
