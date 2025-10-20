import os
import json

# Pasta onde estão os arquivos JSON
input_dir = "./split_batches_with_sequences"  # altere para o seu caminho
output_dir = "./split_batches_with_sequences_fariba"

# Cria a pasta de saída, se não existir
os.makedirs(output_dir, exist_ok=True)

# Percorre todos os arquivos .json da pasta
for filename in os.listdir(input_dir):
    if filename.endswith(".json"):
        input_path = os.path.join(input_dir, filename)
        output_path = os.path.join(output_dir, filename.replace(".json", ".txt"))

        try:
            with open(input_path, "r") as f:
                data = json.load(f)
        except Exception as e:
            print(f"Erro ao ler {filename}: {e}")
            continue

        results = []

        # Extrai os templates
        for path, content in data.items():
            if "group" in content:
                templates = [g.get("template", "").strip() for g in content["group"] if "template" in g]
                if templates:
                    results.append(" | ".join(templates))

        # Escreve o arquivo de saída
        with open(output_path, "w") as out:
            for r in results:
                out.write(r + "\n")

        print(f"✅ Gerado: {output_path}")
