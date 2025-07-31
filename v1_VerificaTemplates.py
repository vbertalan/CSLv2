import os
import json
import re

# Lista de dup_ids fornecida diretamente no Python
dup_ids = [
    "22a3a1e6-ee13-4b9d-8d59-ff9c24b18b06",
    "30850804-bd80-4617-bd35-cc8649d636da", 
    "fe8849dc-cf9f-404d-87a2-c833e6610dbb",
    "e9971382-a7b8-4398-87c7-3fe8c15ab503",
#     # Adicione outros event_ids conforme necessário
]
dup_ids_set = set(dup_ids)

# Caminho para a pasta onde estão os arquivos .json
pasta = 'split_batches_with_sequences'
regex_part_file = re.compile(r'^part_\d+\.json$')

# Armazena templates encontrados (sem duplicação)
templates_encontrados = {}

# Percorre todos os arquivos part_X.json da pasta
for nome_arquivo in os.listdir(pasta):
    if not regex_part_file.match(nome_arquivo):
        continue

    caminho_completo = os.path.join(pasta, nome_arquivo)

    with open(caminho_completo, "r", encoding="utf-8") as f:
        try:
            dados = json.load(f)
        except json.JSONDecodeError:
            print(f"[ERRO] Arquivo malformado ignorado: {nome_arquivo}")
            continue

    for caminho, conteudo in dados.items():
        grupo = conteudo.get("group", [])
        for evento in grupo:
            dup_id = evento.get("dup_id")
            template = evento.get("template")
            if dup_id in dup_ids_set and dup_id not in templates_encontrados:
                templates_encontrados[dup_id] = template

# Exibe os resultados na tela
print("\nTemplates encontrados:")
for dup_id in dup_ids:
    if dup_id in templates_encontrados:
        print(f"{dup_id}: {templates_encontrados[dup_id]}")
    else:
        print(f"{dup_id}: [NÃO ENCONTRADO]")
