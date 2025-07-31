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

# ====== MAIN PIPELINE ======

input_file = '/home/vbertalan/Downloads/whole_data/whole_dataset_with_special_caracters.json'  
output_dir = 'split_batches_with_sequences'
os.makedirs(output_dir, exist_ok=True)

batch_size = 500
batch = {}
entry_count = 0
file_count = 0

template_to_id = {}
id_to_template = {}
next_template_id = 0

sequence_file_path = os.path.join(output_dir, 'all_sequences.txt')
sequence_file = open(sequence_file_path, 'w')

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
