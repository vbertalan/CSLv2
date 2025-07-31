import random

templates = [
    "disk full",
    "compilation failed",
    "linking failed",
    "build aborted",
    "network timeout"
]

linhas = []
for _ in range(1000):
    linha = []

    # Gera eventos com causalidade simulada
    if random.random() < 0.6:
        linha.append("disk full")
        linha.append("compilation failed")
        if random.random() < 0.9:
            linha.append("linking failed")
            if random.random() < 0.8:
                linha.append("build aborted")
    elif random.random() < 0.4:
        linha.append("compilation failed")
        if random.random() < 0.5:
            linha.append("linking failed")

    # Adiciona ruído independente
    if random.random() < 0.3:
        linha.append("network timeout")

    linhas.append(", ".join(linha))

# Salvar
with open("logs_teste.log", "w") as f:
    f.write("\n".join(linhas))
