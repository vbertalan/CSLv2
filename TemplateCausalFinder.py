import os
import json
import re

# Lista de dup_ids fornecida diretamente no código
dup_ids = ["22a3a1e6-ee13-4b9d-8d59-ff9c24b18b06", "439e22f7-e59f-4acb-9dc5-6e8c37f6c5ef", 
           "30850804-bd80-4617-bd35-cc8649d636da", "6a395487-295b-4fce-91c7-50b8e18986ca", 
           "ee6a1067-e65a-467a-ad79-f0f42566de8c", "2586e64b-12a8-4cb9-9a88-fbecd635046f",  
           "032f546a-d005-463d-856d-7e502d9002de", "2465e372-1a98-4088-bc1d-4353d335f92b", 
           "107325ad-6bde-4492-b7f6-4d5a2dfb0c7c", "4b7666f4-4045-4ab2-aa4d-b7f26b7a8e54", 
           "107325ad-6bde-4492-b7f6-4d5a2dfb0c7c", "fe8849dc-cf9f-404d-87a2-c833e6610dbb", 
           "7c013cc3-79cf-4db2-8c9a-75408b1b0ccf", "0d22dca7-6bf6-4fd5-80a0-584cd2d9b9ce", 
           "9ba0835c-8344-48e3-967a-f46f0e02db8c", "e9971382-a7b8-4398-87c7-3fe8c15ab503", 
           "883d13b7-f213-4fcb-9c22-51bea05766ca", "23df7ab8-1629-4edf-93b8-9134fbdb51d7", 
           "e6b849ab-a2af-4df6-930f-784c15f5f42d", "8c604fdd-53c9-4c2a-9433-fb14e1ed6362", 
           "07452e03-006a-40d5-b998-522408afbc33", "c5932d55-fc67-4100-901b-96a1582567d7", 
           "7e6153b1-3b2d-45e4-acf4-4a54c936ba3b"]

dup_ids_set = set(dup_ids)

# Caminho para a pasta com os arquivos
pasta = "/home/vbertalan/Downloads/Projetos/CSL/CSL/split_batches_with_sequences"

# Regex para identificar arquivos do tipo part_X.json
regex_part_file = re.compile(r'^part_\d+\.json$')

# Onde armazenar os resultados encontrados
templates_encontrados = {}

# Varre todos os arquivos na pasta
for nome_arquivo in os.listdir(pasta):
    if not regex_part_file.match(nome_arquivo):
        continue  # pula arquivos que não seguem o padrão
    
    caminho_arquivo = os.path.join(pasta, nome_arquivo)
    
    with open(caminho_arquivo, "r", encoding="utf-8") as f:
        try:
            dados = json.load(f)
        except json.JSONDecodeError:
            print(f"Aviso: erro ao ler {nome_arquivo}, ignorando.")
            continue

        # Suporta arquivos com lista ou dicts
        if isinstance(dados, dict):
            dados = [dados]
        
        for entrada in dados:
            if not isinstance(entrada, dict):
                continue
            dup_id = entrada.get("dup_id")
            template = entrada.get("template")
            if dup_id in dup_ids_set and dup_id not in templates_encontrados:
                templates_encontrados[dup_id] = template

# Exibe os resultados
print("\nTemplates encontrados para os dup_ids:")
for dup_id in dup_ids:
    if dup_id in templates_encontrados:
        print(f"{dup_id}: {templates_encontrados[dup_id]}")
    else:
        print(f"{dup_id}: [NÃO ENCONTRADO]")
