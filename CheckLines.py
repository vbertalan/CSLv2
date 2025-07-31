import os
import json

# Lista de event_ids a serem verificados
# dup_ids_to_check = [
#     "fe8849dc-cf9f-404d-87a2-c833e6610dbb",
#     "7c013cc3-79cf-4db2-8c9a-75408b1b0ccf",
#     "0d22dca7-6bf6-4fd5-80a0-584cd2d9b9ce",
#     "9ba0835c-8344-48e3-967a-f46f0e02db8c",
#     "e9971382-a7b8-4398-87c7-3fe8c15ab503",
#     "883d13b7-f213-4fcb-9c22-51bea05766ca",
#     "23df7ab8-1629-4edf-93b8-9134fbdb51d7",
#     "e6b849ab-a2af-4df6-930f-784c15f5f42d",
#     # Adicione outros event_ids conforme necessário
# ]

dup_ids_to_check = [
    "22a3a1e6-ee13-4b9d-8d59-ff9c24b18b06",
    "30850804-bd80-4617-bd35-cc8649d636da", 
    "fe8849dc-cf9f-404d-87a2-c833e6610dbb",
    "e9971382-a7b8-4398-87c7-3fe8c15ab503",
#     # Adicione outros event_ids conforme necessário
]

# Diretório onde os arquivos part_X.json estão localizados
output_dir = 'split_batches_with_sequences'

# Função para verificar as ocorrências dos dup_ids nos arquivos part_X.json
def check_dup_ids_in_file(file_path, dup_ids_to_check):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        dup_id_counts = {dup_id: 0 for dup_id in dup_ids_to_check}
        
        for key, value in data.items():
            group = value.get('group', [])
            for event in group:
                dup_id = event.get('dup_id')
                if dup_id in dup_id_counts:
                    dup_id_counts[dup_id] += 1
        
        return dup_id_counts

# Verificar todos os arquivos part_X.json
files_with_counts = []

output_file = 'matching_files_output.txt'

with open(output_file, 'w', encoding='utf-8') as f_out:
    f_out.write("Arquivos e número total de ocorrências dos dup_ids:\n")
    
    for file_name in os.listdir(output_dir):
        if file_name.startswith('part_') and file_name.endswith('.json'):
            file_path = os.path.join(output_dir, file_name)
            dup_id_counts = check_dup_ids_in_file(file_path, dup_ids_to_check)
            
            # Somar as ocorrências dos dup_ids
            total_occurrences = sum(dup_id_counts.values())
            
            # Verificar se todos os dup_ids têm pelo menos 2 ocorrências
            if all(count >= 2 for count in dup_id_counts.values()):
                files_with_counts.append((file_name, total_occurrences, dup_id_counts))
                
    # Ordenar os arquivos com base no total de ocorrências em ordem decrescente
    files_with_counts.sort(key=lambda x: x[1], reverse=True)
    
    # Gravar os resultados no arquivo externo
    if files_with_counts:
        f_out.write("Arquivos ordenados por número total de ocorrências:\n")
        for file_name, total_occurrences, dup_id_counts in files_with_counts:
            f_out.write(f"\nArquivo: {file_name} - Total de ocorrências: {total_occurrences}\n")
            for dup_id, count in dup_id_counts.items():
                f_out.write(f"  {dup_id}: {count} ocorrências\n")
    else:
        f_out.write("\nNenhum arquivo atendeu à condição de ter pelo menos 2 ocorrências de cada dup_id.\n")

print(f"\nOs resultados foram salvos no arquivo: {output_file}")