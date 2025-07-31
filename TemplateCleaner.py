# Arquivo com templates a remover (um por linha)
noise_file = 'NoiseTemplates.txt'

# Arquivo de entrada com os templates por grupo
input_file = 'templates_por_grupo.log'

# Arquivo de saída com templates filtrados
output_file = 'templates_filtrados.log'

# Lê os templates de ruído (removendo espaços extras)
with open(noise_file, 'r', encoding='utf-8') as f:
    noise_templates = set(line.strip() for line in f if line.strip())

# Processa o arquivo linha a linha
with open(input_file, 'r', encoding='utf-8') as f_in, open(output_file, 'w', encoding='utf-8') as f_out:
    for line in f_in:
        templates = [t.strip() for t in line.strip().split(',')]
        # Remove templates de ruído
        filtered = [t for t in templates if t not in noise_templates]
        if filtered:
            f_out.write(", ".join(filtered) + '\n')

print(f'Templates filtrados salvos em "{output_file}"')


