import os
import json
import ijson

def extract_templates(events):
    return ["".join(event.get("template", "")) for event in events]

from decimal import Decimal

class DecimalEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, Decimal):
            return float(obj)  
        return super().default(obj)

def save_json(data, path):
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False, cls=DecimalEncoder)


# Caminho do arquivo de entrada
input_file = '/home/vbertalan/Downloads/whole_data/whole_dataset_with_special_caracters.json'  # Substitua com o caminho correto

# Diretório onde os arquivos part_X.json estão localizados
output_dir = 'split_batches_with_sequences'

# Função para encontrar o último arquivo existente
def get_last_file(output_dir):
    existing_files = [f for f in os.listdir(output_dir) if f.startswith('part_') and f.endswith('.json')]
    if existing_files:
        # Extrair o maior número de arquivo existente (part_X.json)
        file_numbers = [int(f.split('_')[1].split('.')[0]) for f in existing_files]
        return max(file_numbers)
    else:
        return -1  # Se não houver arquivos, retorna -1

# Função para retomar a execução a partir de um arquivo específico
def resume_from_file(start_file=501):
    batch_size = 500
    batch = {}
    entry_count = 0
    file_count = start_file
    template_to_id = {}
    id_to_template = {}
    next_template_id = 0

    sequence_file_path = os.path.join(output_dir, 'all_sequences.txt')
    sequence_file = open(sequence_file_path, 'a')  # Usar 'a' para adicionar à sequência existente

    with open(input_file, 'rb') as f_in:
        parser = ijson.kvitems(f_in, '')  

        for key, value in parser:
            metadata = value.get('metadata')
            group = value.get('group')

            if metadata is not None and group is not None:
                # === Build sequence ===
                templates = extract_templates(group)
                sequence = []
                for template in templates:
                    if template not in template_to_id:
                        template_to_id[template] = next_template_id
                        id_to_template[next_template_id] = template
                        next_template_id += 1
                    sequence.append(template_to_id[template])
                # Write this sequence to file
                sequence_file.write(' '.join(map(str, sequence)) + '\n')

                # Add to JSON batch
                batch[key] = {'metadata': metadata, 'group': group}
                entry_count += 1

                # Flush batch to file every 500 entries
                if entry_count == batch_size:
                    part_path = os.path.join(output_dir, f'part_{file_count}.json')
                    save_json(batch, part_path)
                    print(f'Saved {part_path}')
                    batch = {}
                    entry_count = 0
                    file_count += 1

                    # Stop after saving part_1000.json
                    if file_count > 1000:
                        break

    # Save any remaining batch
    if batch:
        part_path = os.path.join(output_dir, f'part_{file_count}.json')
        save_json(batch, part_path)
        print(f'Saved {part_path}')

    sequence_file.close()

    # Save mappings
    save_json(template_to_id, os.path.join(output_dir, 'template_to_id.json'))
    save_json(id_to_template, os.path.join(output_dir, 'id_to_template.json'))

    print("All done.")

# Verifica o último arquivo gerado e retoma a execução de onde parou
last_file = get_last_file(output_dir)
if last_file >= 500:
    print(f"Retomando a partir do arquivo part_{last_file + 1}.json...")
    resume_from_file(start_file=last_file + 1)
else:
    print("Iniciando do início...")
    resume_from_file(start_file=0)
