import os
import json
import re

# Lista Python de event_ids de interesse
event_ids_procurados = [
    "ff9f4bb3-6876-46ef-a858-d8f84c7b0414",
    "4e3f21df-484d-48ce-950e-01319588822a",
    "5e81a0ea-28f1-4999-b01f-0d613d27f59d",
    "a5cf75d8-27fa-429e-9d26-fc1f04436ce8",
    "8c604fdd-53c9-4c2a-9433-fb14e1ed6362",
    "07452e03-006a-40d5-b998-522408afbc33",
    "c5932d55-fc67-4100-901b-96a1582567d7",
    "7e6153b1-3b2d-45e4-acf4-4a54c936ba3b",
    "bbcc318d-4cba-49c7-8ed0-abff91955f11",
    "7d7579e0-4951-4b6f-b4f4-2dc0ed40862e",
    "13f70665-c23c-472a-8f41-da89f65d1421",
    "9755c661-a176-4c0b-acbc-7f10528526cd",
    "fe22b357-11d7-4108-bd2f-ba5d408b9354",
    "7e498221-d92c-4785-9206-1b72575e0392",
    "29df63ce-15ee-471c-b876-4c3dd5c67fbb",
    "e2965329-61aa-429f-b1d8-d7037466bc6b",
    "805fb876-b647-4c1c-8f86-a2fb0e1f9350",
    "18a19759-16dc-4ba4-9a5c-6e2a43fc2650",
    "46069c90-8bdc-46b2-80ed-5b582ac1e302"
]
event_ids_set = set(event_ids_procurados)

# Caminho da pasta com os arquivos .json
pasta = 'split_batches_with_sequences'
regex_part_file = re.compile(r'^part_\d+\.json$')

# Dicionário para armazenar os resultados
event_info = {}

# Percorre os arquivos .json da pasta
for nome_arquivo in os.listdir(pasta):
    if not regex_part_file.match(nome_arquivo):
        continue

    caminho_arquivo = os.path.join(pasta, nome_arquivo)

    try:
        with open(caminho_arquivo, "r", encoding="utf-8") as f:
            dados = json.load(f)
    except Exception as e:
        print(f"[ERRO] Falha ao ler {nome_arquivo}: {e}")
        continue

    for bloco in dados.values():
        for evento in bloco.get("group", []):
            eid = evento.get("event_id")
            if eid in event_ids_set and eid not in event_info:
                event_info[eid] = {
                    "template": evento.get("template", "[sem template]"),
                    "dup_id": evento.get("dup_id", "[sem dup_id]"),
                    "arquivo": nome_arquivo
                }

# Exibe os resultados
print("\n=== Resultados dos event_ids buscados ===")
for eid in event_ids_procurados:
    if eid in event_info:
        info = event_info[eid]
        print(f"\nEvent ID: {eid}")
        print(f"Template: {info['template']}")
        print(f"Dup ID:   {info['dup_id']}")
        print(f"Arquivo:  {info['arquivo']}")
    else:
        print(f"\nEvent ID: {eid}")
        print(">>> [NÃO ENCONTRADO]")
