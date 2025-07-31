import os
import json
import re

input_dir = 'split_batches_with_sequences'
output_file = 'templates_por_grupo.log'

pattern = re.compile(r'^part_\d+\.json$')
lines = []

for file_name in sorted(os.listdir(input_dir)):
    if not pattern.match(file_name):
        continue

    file_path = os.path.join(input_dir, file_name)
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    for content in data.values():
        group = content.get('group', [])
        templates = [event.get("template", "").strip() for event in group if event.get("template")]
        if templates:
            line = ", ".join(templates)
            lines.append(line)

with open(output_file, 'w', encoding='utf-8') as f_out:
    for line in lines:
        f_out.write(line + '\n')

print(f'Templates agrupados salvos em "{output_file}"')
