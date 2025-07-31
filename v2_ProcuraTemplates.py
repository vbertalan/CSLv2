import os
import json
import re
from collections import defaultdict

# Lista de templates a procurar (case insensitive)
templates_procurados = [
    "generating catalog error license"
]
templates_set = set(t.lower().strip() for t in templates_procurados)

# Pasta onde estão os arquivos JSON
pasta = 'split_batches_with_sequences'
regex_part_file = re.compile(r'^part_\d+\.json$')

# Dicionários para contagem e arquivos onde apareceram
contagem_templates = defaultdict(int)
arquivos_por_template = defaultdict(set)

# Percorre os arquivos da pasta
for nome_arquivo in os.listdir(pasta):
    if not regex_part_file.match(nome_arquivo):
        continue

    caminho_arquivo = os.path.join(pasta, nome_arquivo)

    try:
        with open(caminho_arquivo, "r", encoding="utf-8") as f:
            dados = json.load(f)
    except Exception as e:
        print(f"[ERRO] Falha ao abrir {nome_arquivo}: {e}")
        continue

    for bloco in dados.values():
        for evento in bloco.get("group", []):
            template = evento.get("template", "").lower().strip()
            if template in templates_set:
                contagem_templates[template] += 1
                arquivos_por_template[template].add(nome_arquivo)

# Exibe os resultados
print("\n=== Resultados ===")
for template_original in templates_procurados:
    template_normalizado = template_original.lower().strip()
    count = contagem_templates.get(template_normalizado, 0)
    arquivos = arquivos_por_template.get(template_normalizado, set())
    
    print(f"\nTemplate: '{template_original}'")
    print(f"Ocorrências: {count}")
    if arquivos:
        print(f"Arquivos onde apareceu: {', '.join(sorted(arquivos))}")
    else:
        print("Arquivos onde apareceu: [NENHUM]")
