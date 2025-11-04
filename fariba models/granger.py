# %% [markdown]
# Granger Lasso (multivariate) for logs → binary time series
# - single cell
# - no plots
# - saves an aggregated CSV per edge (source->target) and a detailed CSV per lag
# Requirements (uncomment if needed):
# !pip install causallearn networkx matplotlib scikit-learn pandas --quiet

import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer, StandardScaler
from causallearn.search.Granger.Granger import Granger
from collections import Counter
import numpy as np
from typing import List, Tuple
import os

# =======================
# 🔧 Editable parameters
# =======================
log_path = "final_sequences_NoNoise.txt"        # path to file with one event per line
min_freq = 5                                    # minimum frequency to keep a template
maxlag = 3                                      # number of lags in Granger Lasso
threshold = 1e-6                                # L1 threshold to consider an edge present
use_signed_weight = True                        # True: signed weight; False: only magnitude
standardize = True                              # z-score before Granger (recommended)

csv_path = "granger_relations_synthetic.csv"     # aggregated CSV (1 row per edge j->i)
csv_by_lag_path = "granger_relations_by_lag.csv" # CSV with 1 row per (j->i, lag)
report_path = "granger_report.txt"               # simple report

# =======================
# 🧩 Utility functions
# =======================
def read_log_lines(log_path: str) -> List[str]:
    if not os.path.exists(log_path):
        raise FileNotFoundError(f"File not found: {log_path}")
    with open(log_path, 'r') as f:
        return [line.strip() for line in f if line.strip()]

def logs_to_time_series(log_lines: List[str], min_freq: int = 5) -> Tuple[pd.DataFrame, List[str]]:
    counts = Counter(log_lines)
    frequent_templates = {tpl for tpl, freq in counts.items() if freq >= min_freq}
    filtered_lines = [line if line in frequent_templates else None for line in log_lines]

    if len(frequent_templates) == 0:
        raise ValueError("No template survived the frequency filter (min_freq too high?).")

    mlb = MultiLabelBinarizer()
    binary_matrix = mlb.fit_transform([[line] if line is not None else [] for line in filtered_lines])
    df = pd.DataFrame(binary_matrix, columns=mlb.classes_)
    return df, list(mlb.classes_)

def run_multivar_granger(
    df: pd.DataFrame,
    variable_names: List[str],
    csv_path: str,
    report_path: str,
    maxlag: int,
    threshold: float,
    use_signed_weight: bool,
    standardize: bool,
    csv_by_lag_path: str
) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """
    Returns:
      - df_all: aggregated edges (j->i) summing over lags
      - coeff_matrix: original N x (N*maxlag) matrix from granger_lasso
      - coeff_lag: tensor [maxlag, N, N] (lag, target i, source j)
    """
    X = df.to_numpy().astype(float)
    N = len(variable_names)

    if standardize:
        scaler = StandardScaler(with_mean=True, with_std=True)
        X = scaler.fit_transform(X)

    G = Granger(maxlag=maxlag)
    coeff_matrix = G.granger_lasso(X)  # shape: [N, N*maxlag]

    # Reshape to [maxlag, N(target i), N(source j)]
    # Assuming columns are stacked by lag: [t-1 (1..N), t-2 (1..N), ..., t-maxlag (1..N)]
    coeff_lag = coeff_matrix.reshape(N, N, maxlag).transpose(2, 0, 1)

    # Aggregation per pair (j->i): strength = L1 sum across lags; sign = sign of sum
    edges = []
    for i in range(N):         # target
        for j in range(N):     # source
            if i == j:
                continue
            betas_ij = coeff_lag[:, i, j]              # vector (maxlag,)
            strength = float(np.sum(np.abs(betas_ij))) # L1 across lags
            if strength > threshold:
                signed_sum = float(np.sum(betas_ij))
                if use_signed_weight:
                    sign = 1.0 if signed_sum >= 0 else -1.0
                    weight = strength * sign
                else:
                    weight = strength
                edges.append((variable_names[j], variable_names[i], weight, strength, signed_sum))

    df_all = pd.DataFrame(edges, columns=["source", "target", "weight", "l1_strength", "sum_signed"])
    df_all.to_csv(csv_path, index=False)

    # Detailed CSV per lag (for audit)
    if csv_by_lag_path is not None:
        rows = []
        for lag in range(maxlag):  # 0..maxlag-1  (actual lag = lag+1)
            for i in range(N):
                for j in range(N):
                    if i == j:
                        continue
                    beta = float(coeff_lag[lag, i, j])
                    rows.append({
                        "lag": lag + 1,
                        "source": variable_names[j],
                        "target": variable_names[i],
                        "beta": beta
                    })
        pd.DataFrame(rows).to_csv(csv_by_lag_path, index=False)

    with open(report_path, 'w') as f:
        f.write("=== Granger Causality Report (Multivariate, Lasso) ===\n")
        f.write(f"Total variables: {N}\n")
        f.write(f"maxlag: {maxlag}\n")
        f.write(f"Standardization (z-score): {standardize}\n")
        f.write(f"L1 threshold for edge presence: {threshold}\n")
        f.write(f"Detected edges: {len(df_all)}\n")
        f.write(f"\nCSV (aggregated): {csv_path}\n")
        if csv_by_lag_path:
            f.write(f"CSV (by lag): {csv_by_lag_path}\n")

    return df_all, coeff_matrix, coeff_lag

# =======================
# ▶️ Execution
# =======================
log_lines = read_log_lines(log_path)
df, variable_names = logs_to_time_series(log_lines, min_freq=min_freq)
print(f"📊 Time series: {df.shape[0]} timestamps × {df.shape[1]} events")

df_result, coeff_matrix, coeff_lag = run_multivar_granger(
    df=df,
    variable_names=variable_names,
    csv_path=csv_path,
    report_path=report_path,
    maxlag=maxlag,
    threshold=threshold,
    use_signed_weight=use_signed_weight,
    standardize=standardize,
    csv_by_lag_path=csv_by_lag_path
)

# Quick preview in notebook
print(f"\n✅ Aggregated CSV saved to: {os.path.abspath(csv_path)}")