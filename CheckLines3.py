import os
import json
from collections import Counter

# Diretório onde os arquivos part_X.json estão localizados
output_dir = 'split_batches_with_sequences'

# Função para contar as ocorrências de dup_ids nos arquivos part_X.json
def count_dup_ids_in_file(file_path, dup_id_counts, dup_id_data):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

        for key, value in data.items():
            group = value.get('group', [])
            for event in group:
                # Contar dup_ids
                dup_id = event.get('dup_id')
                if dup_id:
                    dup_id_counts[dup_id] += 1
                    if dup_id not in dup_id_data:
                        dup_id_data[dup_id] = {
                            "raw": event.get('raw'),
                            "template": event.get('template')
                        }

# Contadores globais para armazenar frequências e dados de dup_ids
dup_id_counts = Counter()
dup_id_data = {}

# Verificar todos os arquivos part_X.json e contar as ocorrências de dup_ids
for file_name in os.listdir(output_dir):
    if file_name.startswith('part_') and file_name.endswith('.json'):
        file_path = os.path.join(output_dir, file_name)
        count_dup_ids_in_file(file_path, dup_id_counts, dup_id_data)

# Encontrar os 20 dup_ids mais frequentes
most_common_dup_ids = dup_id_counts.most_common(20)

# Criar o arquivo para salvar as informações dos dup_ids mais frequentes
output_file_dup_ids = 'top_20_dup_ids.json'
with open(output_file_dup_ids, 'w', encoding='utf-8') as f_out_dup_ids:
    result_dup_ids = []
    for dup_id, frequency in most_common_dup_ids:
        result_dup_ids.append({
            "dup_id": dup_id,
            "frequencia": frequency,
            "raw": dup_id_data[dup_id]["raw"],
            "template": dup_id_data[dup_id]["template"]
        })
    json.dump(result_dup_ids, f_out_dup_ids, indent=2, ensure_ascii=False)

print(f"Os 20 dup_ids mais frequentes foram salvos em '{output_file_dup_ids}'")
