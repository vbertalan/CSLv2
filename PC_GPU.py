import argparse
import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer
from collections import Counter
import numpy as np

# === GPU PC (gpucsl) ===
try:
    from gpucsl.pc.pc import DiscretePC
    GPUCSL_AVAILABLE = True
except ImportError:
    GPUCSL_AVAILABLE = False

# -------------------------
# Funções utilitárias
# -------------------------
def read_log_lines(log_path):
    with open(log_path, 'r') as f:
        return [line.strip() for line in f if line.strip()]

def choose_window_size(log_lines, frequent_templates, min_samples_per_feature=5, max_window=30):
    n_features = len(frequent_templates)
    n_lines = len(log_lines)
    if n_features == 0:
        raise ValueError("Nenhum evento com frequência suficiente.")
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
        raise ValueError("Nenhum template sobreviveu ao filtro de frequência.")
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
        confidence = "ALTA"
    elif ratio >= 5:
        confidence = "MODERADA"
    else:
        confidence = "BAIXA"
    return df, mlb.classes_, window_size, n_samples, n_features, ratio, confidence

# -------------------------
# Versão CPU (causallearn)
# -------------------------
def run_pc_cpu(df, variable_names, csv_path, report_path, alpha=0.01):
    from causallearn.search.ConstraintBased.PC import pc
    X = df.to_numpy().astype(int)
    cg = pc(data=X, alpha=alpha, indep_test="chisq", uc_rule=0, verbose=False)
    directed, bidirectional, undirected = [], [], []
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
    df_all = pd.DataFrame(directed + bidirectional + undirected, columns=["source", "target"])
    df_all["relation"] = (["directed"] * len(directed) +
                          ["bidirectional"] * len(bidirectional) +
                          ["undirected"] * len(undirected))
    df_all.to_csv(csv_path, index=False)
    with open(report_path, 'w') as f:
        f.write("=== Relatório de Causalidade (CPU) ===\n")
        f.write(f"Total de variáveis: {n_vars}\n")
        f.write(f"Arestas direcionadas: {len(directed)}\n")
        f.write(f"Arestas bidirecionais: {len(bidirectional)}\n")
        f.write(f"Arestas não-direcionadas: {len(undirected)}\n")
    return df_all

# -------------------------
# Versão GPU (gpucsl) com controle de VRAM
# -------------------------
def run_pc_gpu(df, variable_names, csv_path, report_path, alpha=0.01, max_level=3, mem_fraction=None):
    if not GPUCSL_AVAILABLE:
        raise RuntimeError("gpucsl não está instalado. Instale com: pip install cupy-cudaXXX gpucsl")

    import cupy as cp
    gpu_count = cp.cuda.runtime.getDeviceCount()
    if gpu_count == 0:
        raise RuntimeError("Nenhuma GPU foi detectada pelo CuPy.")
    gpu_props = cp.cuda.runtime.getDeviceProperties(0)
    gpu_name = gpu_props['name'].decode()
    total_mem_gb = gpu_props['totalGlobalMem'] / (1024**3)
    print(f"✅ GPU detectada: {gpu_name} ({gpu_count} encontrada(s))")
    print(f"💾 VRAM total: {total_mem_gb:.2f} GB")

    if mem_fraction is None:
        n_samples, n_features = df.shape
        ratio = n_samples / max(1, n_features)
        if ratio > 50:
            mem_fraction = 0.5
        elif ratio > 20:
            mem_fraction = 0.7
        else:
            mem_fraction = 0.9
        print(f"⚙️ Memória fracionária ajustada automaticamente: {mem_fraction*100:.0f}%")

    X = df.to_numpy().astype(np.int32)
    pc_instance = DiscretePC(X, max_level, alpha)
    pc_instance.set_distribution_specific_options(memory_restriction=mem_fraction)

    ((directed_graph, sepsets, pmax,
      skel_time, orient_time, kernel_time),
     pc_time) = pc_instance.execute()

    dirs, bi, und = [], [], []
    n = directed_graph.shape[0]
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            a, b = directed_graph[i, j], directed_graph[j, i]
            if a == -1 and b == 1:
                dirs.append((variable_names[i], variable_names[j]))
            elif a == 1 and b == 1:
                bi.append((variable_names[i], variable_names[j]))
            elif a == -1 and b == -1:
                und.append((variable_names[i], variable_names[j]))

    df_all = pd.DataFrame(dirs + bi + und, columns=["source", "target"])
    df_all["relation"] = (["directed"] * len(dirs) +
                          ["bidirectional"] * len(bi) +
                          ["undirected"] * len(und))
    df_all.to_csv(csv_path, index=False)

    with open(report_path, 'w') as f:
        f.write("=== Relatório de Causalidade (GPU) ===\n")
        f.write(f"GPU usada: {gpu_name}\n")
        f.write(f"VRAM total: {total_mem_gb:.2f} GB\n")
        f.write(f"Fração de memória usada: {mem_fraction*100:.0f}%\n")
        f.write(f"Total de variáveis: {n}\n")
        f.write(f"Arestas direcionadas: {len(dirs)}\n")
        f.write(f"Arestas bidirecionais: {len(bi)}\n")
        f.write(f"Arestas não-direcionadas: {len(und)}\n")
        f.write("\n--- P-values máximos (pmax) ---\n")
        for i in range(n):
            for j in range(n):
                if i < j and pmax[i][j] != 0:
                    f.write(f"{variable_names[i]} - {variable_names[j]}: p={pmax[i][j]:.4f}\n")
        f.write(f"\nTempo total execução: {pc_time:.2f} segundos\n")

    print(f"💾 CSV salvo em: {csv_path}")
    print(f"📝 Relatório salvo em: {report_path}")
    return df_all

# -------------------------
# Main
# -------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--log_path", default="logs/logs_teste.log")
    parser.add_argument("--csv_path", default="relacoes_causais.csv")
    parser.add_argument("--report_path", default="relatorio.txt")
    parser.add_argument("--alpha", type=float, default=0.01)
    parser.add_argument("--use_gpu", action="store_true", help="Usar versão GPU (gpucsl)")
    parser.add_argument("--mem_fraction", type=float, default=None, help="Fração de VRAM a usar (0-1). Se omitido, calcula automático.")
    args = parser.parse_args()

    log_lines = read_log_lines(args.log_path)
    df, variable_names, window_size, n_samples, n_features, ratio, confidence = build_windowed_dataset(
        log_lines, window_size=None, min_freq=5
    )

    print(f"📏 Window size: {window_size}")
    print(f"📊 {n_samples} janelas × {n_features} eventos → razão amostra/feature = {ratio:.2f}")
    print(f"🔎 Nível de confiança estatística: {confidence}")

    if args.use_gpu:
        print("🚀 Rodando PC na GPU (gpucsl)...")
        df_result = run_pc_gpu(df, variable_names, args.csv_path, args.report_path,
                               alpha=args.alpha, mem_fraction=args.mem_fraction)
    else:
        print("🖥️ Rodando PC na CPU (causallearn)...")
        df_result = run_pc_cpu(df, variable_names, args.csv_path, args.report_path,
                               alpha=args.alpha)

    print(df_result.head())
