# Dataset

## Source
**UR3 CobotOps** – a publicly available dataset containing sensor readings from a Universal Robots UR3 collaborative robot arm under normal and anomalous operating conditions.

- **Features:** joint currents (J1–J3), joint temperatures (J1–J2), vibration
- **Target:** `anomaly` — binary label (0 = Normal, 1 = Anomaly)
- **Size:** ~5,000 timestamped samples (85% normal / 15% anomalous)

## Download Instructions
If you have access to the original dataset, place the CSV file here as:

```
data/ur3_cobotops.csv
```

## Synthetic Fallback
If no CSV is found, `src/train.py` will automatically generate a realistic synthetic dataset and save it to `data/ur3_cobotops.csv`. This is the default behaviour and sufficient for running all experiments in this repository.

The synthetic data matches the statistical distribution of UR3 operating characteristics:

| Feature        | Normal μ (σ)   | Anomalous μ (σ) |
|----------------|----------------|-----------------|
| current_j1 (A) | 2.10 (0.15)    | 3.20 (0.40)     |
| current_j2 (A) | 2.40 (0.18)    | 3.60 (0.50)     |
| current_j3 (A) | 1.80 (0.12)    | 2.80 (0.35)     |
| temp_j1 (°C)   | 45.0 (2.0)     | 62.0 (5.0)      |
| temp_j2 (°C)   | 47.0 (2.5)     | 65.0 (6.0)      |
| vibration (g)  | 0.05 (0.01)    | 0.18 (0.05)     |
