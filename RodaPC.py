import pandas as pd
from causallearn.search.ConstraintBased.PC import pc
from causallearn.utils.cit import CIT

# === CONFIGURAÇÃO ===
ARQUIVO_LOG = "logs_teste.log"  # seu arquivo .log
ALPHA = 0.05  # nível de significância

# === 1. Leitura do arquivo .log ===
with open(ARQUIVO_LOG, "r") as f:
    linhas = [linha.strip() for linha in f if linha.strip()]  # ignora vazias

# === 2. Processamento para matriz binária ===
linhas_templates = [linha.split(", ") for linha in linhas]
todos_templates = sorted(set(template for linha in linhas_templates for template in linha))

matriz_binaria = []
for linha in linhas_templates:
    binaria = [1 if template in linha else 0 for template in todos_templates]
    matriz_binaria.append(binaria)

df_bin = pd.DataFrame(matriz_binaria, columns=todos_templates)
df_bin.to_csv("matriz_binaria.csv", index=False)
print("[✓] Matriz binária salva como 'matriz_binaria.csv'.")

# === 3. Execução do PC com teste G² ===
print("[...] Rodando PC com teste G² (gsq)...")
data = df_bin.values
indep_test = CIT(data, method="gsq")

cg = pc(
    data,
    alpha=ALPHA,
    indep_test_func=indep_test,
    stable=True,
    show_progress=True,
    verbose=False
)

# === 4. Extração de relações causais ===
templates = df_bin.columns.tolist()
relacoes = []
for node in range(len(templates)):
    pais = [i for i in range(len(templates)) if cg.G.graph[i][node] == "-->"]
    filhos = [i for i in range(len(templates)) if cg.G.graph[node][i] == "-->"]

    ancestral = templates[pais[0]] if pais else "NA"
    descendente = templates[filhos[0]] if filhos else "NA"

    relacoes.append({
        "template": templates[node],
        "ancestral": ancestral,
        "descendente": descendente
    })

df_relacoes = pd.DataFrame(relacoes)
df_relacoes.to_csv("relacoes_detectadas.csv", index=False)
print("[✓] Relações causais salvas como 'relacoes_detectadas.csv'.")
