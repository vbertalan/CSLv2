import os
import json
from collections import Counter
import csv

# Diretório onde os arquivos part_X.json estão localizados
input_dir = 'split_batches_with_sequences'

# Contador para armazenar as ocorrências de cada template
template_counter = Counter()

# Percorrer todos os arquivos part_X.json da pasta
for file_name in os.listdir(input_dir):
    if file_name.startswith('part_') and file_name.endswith('.json'):
        file_path = os.path.join(input_dir, file_name)
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            for value in data.values():
                group = value.get('group', [])
                for event in group:
                    template = event.get('template')
                    if template:
                        template_counter[template] += 1

# Ordenar os templates por número de ocorrências (decrescente)
sorted_templates = template_counter.most_common()

# Caminho do arquivo de saída
output_file = 'template_frequencies.csv'

# Escrever os resultados em um arquivo CSV
with open(output_file, 'w', newline='', encoding='utf-8') as f_out:
    writer = csv.writer(f_out)
    writer.writerow(['template', 'numero_de_ocorrencias'])
    for template, count in sorted_templates:
        writer.writerow([template, count])

print(f"Frequência dos templates salva em '{output_file}'")
