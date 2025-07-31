# === Step 0 - Imports ===

import os
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
from pgmpy.estimators import PC
import DrainMethod

# === Step 1 - Configurações Gerais ===

## General parameters 

input_dir = os.path.join(os.getcwd(), "logs") # The input directory of raw logs
output_dir = os.path.join(os.getcwd(), "parsings")  # The output directory of parsing results
vector_dir = os.path.join(os.getcwd(), "embeddings")  # The vector directory of converted logs

#logName = 'ciena-mini.txt' 
logName = 'log-lines.txt' # Name of file to be parsed

log_format = '<Content>' # Format of the file, if there are different fields
regex = [] # Regex strings for Drain execution

BIN_SIZE = 60  # Sugestão: maior valor reduz número de bins
MIN_TEMPLATE_OCCURRENCES = 5  # Filtro: templates com menos que isso são ignorados

# === Step 2 - Leitura e pré-processamento dos logs ===

def parse_logs():
    ## Drain parameters
    st = 0.5 # Drain similarity threshold
    depth = 5 # Max depth of the parsing tree

    ## Parses file, using DrainMethod
    print('\n=== Starting Drain Parsing ===')
    indir = os.path.join(input_dir, os.path.dirname(logName))
    print(indir)
    log_file = os.path.basename(logName)

    parser = DrainMethod.LogParser(log_format=log_format, indir=indir, outdir=output_dir, rex=regex, depth=depth, st=st)
    drain_results = parser.parse(log_file)
    return drain_results

# === Step 3 - Separação dos dados ===

def slice_dataset(drain_results):
    dataset_size = len(drain_results)
    num_bins = max(1, dataset_size // BIN_SIZE)  # evita divisão por zero
    unique_templates = drain_results["EventTemplate"].unique()
    print(len(unique_templates))
    return unique_templates, BIN_SIZE, num_bins

# === Step 4 - Criação da matriz de contagem ===

def create_template_matrix(templates, bin_size, num_bins, drain_results):
    drain_results = drain_results.copy()
    drain_results['Bin'] = (drain_results.index // bin_size) + 1

    df_grouped = drain_results.groupby(['EventTemplate', 'Bin']).size().unstack(fill_value=0)

    # Garante todas as colunas de bin
    for col in range(1, num_bins + 1):
        if col not in df_grouped.columns:
            df_grouped[col] = 0

    df_grouped = df_grouped[sorted(df_grouped.columns)]
    df_grouped.reset_index(inplace=True)
    df_grouped.rename(columns={"EventTemplate": "Template"}, inplace=True)

    print(f"Matriz criada com shape: {df_grouped.shape}")
    return df_grouped

# === Step 5 - Geração do grafo causal ===

def generate_graph(template_matrix):
    df_causal = template_matrix.drop(columns=['Template'])

    est = PC(df_causal)
    model_chi = est.estimate(ci_test='chi_square')
    model_gsq, _ = est.estimate(ci_test='g_sq', return_type='skeleton')

    print(f"Chi-Square edges: {len(model_chi.edges())}")
    print(f"G^2 Skeleton edges: {len(model_gsq.edges())}")

    # Geração do grafo
    graph = nx.DiGraph(model_chi.edges())
    plt.figure(figsize=(12, 8))
    nx.draw(graph, with_labels=True, node_size=700, font_size=10)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    graph_path = os.path.join(OUTPUT_DIR, "causal_graph.png")
    plt.title("Causal Graph using PC Algorithm")
    plt.savefig(graph_path, format="png", dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Grafo salvo em: {graph_path}")

# === Step 6 - Execução principal ===

def main():
    parsed_logs = parse_logs()
    templates, bin_size, num_bins = slice_dataset(parsed_logs)
    template_matrix = create_template_matrix(templates, bin_size, num_bins, parsed_logs)
    generate_graph(template_matrix)
    print("Execução terminada.")

# === Execução ===

if __name__ == "__main__":
    main()
