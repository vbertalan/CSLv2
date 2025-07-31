import os
import json

# Lista de event_ids a serem verificados
event_ids_to_check = [
    "22a3a1e6-ee13-4b9d-8d59-ff9c24b18b06", ## COM DUP_ID NULL
    "30850804-bd80-4617-bd35-cc8649d636da"
    # Adic1ione outros event_ids conforme necessário
]
# Diretório onde os arquivos part_X.json estão localizados
output_dir = 'split_batches_with_sequences'

# Função para verificar as ocorrências dos event_ids nos arquivos part_X.json
def check_event_ids_in_file(file_path, event_ids_to_check):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        event_id_counts = {event_id: 0 for event_id in event_ids_to_check}
        found_event_ids = set()

        for key, value in data.items():
            group = value.get('group', [])
            for event in group:
                event_id = event.get('event_id')
                if event_id in event_ids_to_check:
                    event_id_counts[event_id] += 1
                    found_event_ids.add(event_id)

        # Retorna True se algum event_id foi encontrado e as contagens dos event_ids encontrados
        return found_event_ids, event_id_counts

# Verificar todos os arquivos part_X.json
files_with_matching_event_ids = []
for file_name in os.listdir(output_dir):
    if file_name.startswith('part_') and file_name.endswith('.json'):
        file_path = os.path.join(output_dir, file_name)
        found_event_ids, event_id_counts = check_event_ids_in_file(file_path, event_ids_to_check)

        if found_event_ids:
            files_with_matching_event_ids.append((file_name, event_id_counts))

# Exibir arquivos que contêm pelo menos um dos event_ids e as contagens de ocorrências
print("Arquivos que contêm pelo menos um dos event_ids:")
for file_name, event_id_counts in files_with_matching_event_ids:
    print(f"\nArquivo: {file_name}")
    for event_id, count in event_id_counts.items():
        if count > 0:
            print(f"  Event ID: {event_id}, Ocorrências: {count}")
