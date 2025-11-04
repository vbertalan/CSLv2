import argparse
import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer
from causallearn.search.ConstraintBased.PC import pc
from collections import Counter
import numpy as np

# -------------------------
# Utility functions
# -------------------------

def read_log_lines(log_path):
    with open(log_path, 'r') as f:
        return [line.strip() for line in f if line.strip()]

def choose_window_size(log_lines, frequent_templates, min_samples_per_feature=5, max_window=30):
    n_features = len(frequent_templates)
    n_lines = len(log_lines)

    if n_features == 0:
        raise ValueError("No event with sufficient frequency.")

    for window_size in range(max_window, 1, -1):
        n_samples = n_lines - window_size + 1
        if n_samples >= min_samples_per_feature * n_features:
            return window_size
    return 2

def build_windowed_dataset(log_lines, window_size=None, min_freq=5):
    counts = Counter(log_lines)
    frequent_templates = {tpl for tpl, freq in counts.items() if freq >= min_freq}
    filtered_lines = [line for line in log_lines if line in frequent_templates]

    if len(frequent_templates) == 0:
        raise ValueError("No template survived the frequency filter.")

    if window_size is None:
        window_size = choose_window_size(filtered_lines, frequent_templates)

    windows = []
    for i in range(len(filtered_lines) - window_size + 1):
        window = filtered_lines[i:i+window_size]
        windows.append(set(window))

    mlb = MultiLabelBinarizer()
    binary_matrix = mlb.fit_transform(windows)
    df = pd.DataFrame(binary_matrix, columns=mlb.classes_)

    n_samples = df.shape[0]
    n_features = df.shape[1]
    ratio = n_samples / n_features if n_features else 0

    if ratio >= 10:
        confidence = "HIGH"
    elif ratio >= 5:
        confidence = "MODERATE"
    else:
        confidence = "LOW"

    return df, mlb.classes_, window_size, n_samples, n_features, ratio, confidence

# -------------------------
# PC algorithm (CPU only)
# -------------------------
def run_pc_cpu(df, variable_names, csv_path, report_path, alpha=0.01):
    """
    Runs the PC algorithm using the CPU version from causallearn.
    """
    X = df.to_numpy().astype(int)
    cg = pc(data=X, alpha=alpha, indep_test="chisq", uc_rule=0, verbose=False)

    directed = []
    bidirectional = []
    undirected = []

    n_vars = len(variable_names)
    for i in range(n_vars):
        for j in range(n_vars):
            if i == j:
                continue
            a = cg.G.graph[i, j]
            b = cg.G.graph[j, i]
            if a == -1 and b == 1:
                directed.append((variable_names[i], variable_names[j]))
            elif a == 1 and b == 1:
                bidirectional.append((variable_names[i], variable_names[j]))
            elif a == -1 and b == -1:
                undirected.append((variable_names[i], variable_names[j]))

    df_all = pd.DataFrame(directed + bidirectional + undirected,
                          columns=["source", "target"])
    df_all["relation"] = (["directed"] * len(directed) +
                          ["bidirectional"] * len(bidirectional) +
                          ["undirected"] * len(undirected))
    df_all.to_csv(csv_path, index=False)

    with open(report_path, 'w') as f:
        f.write("=== Causality Report (CPU) ===\n")
        f.write(f"Total variables: {n_vars}\n")
        f.write(f"Directed edges: {len(directed)}\n")
        f.write(f"Bidirectional edges: {len(bidirectional)}\n")
        f.write(f"Undirected edges: {len(undirected)}\n")

    print(f"💾 CSV saved at: {csv_path}")
    print(f"📝 Report saved at: {report_path}")
    return df_all

# -------------------------
# Main
# -------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--log_path", default="final_sequences_NoNoise.txt")
    parser.add_argument("--csv_path", default="causal_relations_pc.csv")
    parser.add_argument("--report_path", default="report_pc.txt")
    parser.add_argument("--alpha", type=float, default=0.01)
    args = parser.parse_args()

    log_lines = read_log_lines(args.log_path)
    df, variable_names, window_size, n_samples, n_features, ratio, confidence = build_windowed_dataset(
        log_lines, window_size=None, min_freq=5
    )

    print(f"📏 Window size: {window_size}")
    print(f"📊 {n_samples} windows × {n_features} events → sample/feature ratio = {ratio:.2f}")
    print(f"🔎 Statistical confidence level: {confidence}")

    print("🖥️ Running PC on CPU (causallearn)...")
    df_result = run_pc_cpu(df, variable_names, args.csv_path, args.report_path, alpha=args.alpha)

    print(df_result.head())
