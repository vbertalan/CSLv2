import os
import json
import csv

# Diretório onde os arquivos estão
input_dir = 'split_batches_with_sequences'

# Lista de tokens a serem buscados (case-insensitive)
tokens = ["make", "compile", "linker", "error"]

# Lista para armazenar os resultados
results = []

# Para cada arquivo JSON na pasta
for file_name in sorted(os.listdir(input_dir)):
    if file_name.startswith('part_') and file_name.endswith('.json'):
        token_counts = dict.fromkeys(tokens, 0)  # zera o contador por arquivo
        file_path = os.path.join(input_dir, file_name)
        
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            for value in data.values():
                group = value.get('group', [])
                for event in group:
                    template = event.get('template', '')
                    template_lower = template.lower()
                    for token in tokens:
                        if token in template_lower:
                            token_counts[token] += 1

        # Adiciona resultado da linha
        row = {'arquivo': file_name}
        row.update(token_counts)
        results.append(row)

# Nome do arquivo CSV de saída
output_file = 'token_counts_por_arquivo.csv'

# Escreve o CSV
with open(output_file, 'w', newline='', encoding='utf-8') as f_out:
    writer = csv.DictWriter(f_out, fieldnames=['arquivo'] + tokens)
    writer.writeheader()
    writer.writerows(results)

print(f"Contagem de tokens por arquivo salva em '{output_file}'")