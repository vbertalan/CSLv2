import os
import torch
import matplotlib.pyplot as plt
from datasets import Dataset, load_from_disk, concatenate_datasets
from transformers import (
    GPT2Tokenizer,
    GPT2LMHeadModel,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
    EarlyStoppingCallback,
)
import hashlib

# === CONFIGURAÇÕES ===
model_name = "gpt2"
input_txt_path = "/home/vberta/projects/def-aloise/vberta/Paper3/templates_v4.log"
output_dir = "./gpt2_log_model"
block_size = 128
num_train_epochs = 10
train_frac = 0.2  # Usar apenas 10% para acelerar o treino
eval_ratio = 0.1
cached_dataset_path = "./tokenized_dataset"
chunk_size = 2_000_000  # Tokenizar em blocos de 2 milhões de linhas

# === VERIFICA DISPONIBILIDADE DE GPU ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Usando dispositivo: {device}")

# === TOKENIZER E PAD TOKEN ===
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

# === MODELO ===
model = GPT2LMHeadModel.from_pretrained(model_name)
model.resize_token_embeddings(len(tokenizer))

if torch.cuda.device_count() > 1:
    print(f"Usando {torch.cuda.device_count()} GPUs com Accelerate")

# === CHECAGEM DE CONSISTÊNCIA DO CACHE ===
def compute_file_hash(path):
    with open(path, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()

hash_path = cached_dataset_path + "_meta.txt"
hash_mismatch = True
current_hash = compute_file_hash(input_txt_path) + f"_block{block_size}"

if os.path.exists(cached_dataset_path) and os.path.exists(hash_path):
    with open(hash_path, "r") as f:
        saved_hash = f.read().strip()
    if saved_hash == current_hash:
        hash_mismatch = False

# === TOKENIZAÇÃO POR BLOCOS ===
if not hash_mismatch:
    print("Carregando dataset tokenizado do disco...")
    tokenized = load_from_disk(cached_dataset_path)
else:
    print("Tokenizando o dataset em blocos...")
    with open(input_txt_path, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]

    datasets_parts = []
    total = len(lines)
    for i in range(0, total, chunk_size):
        print(f"Processando linhas {i} a {min(i+chunk_size, total)}...")
        chunk = lines[i:i+chunk_size]
        dataset_chunk = Dataset.from_dict({"text": chunk})

        def tokenize_function(examples):
            tokens = tokenizer(
                examples["text"],
                truncation=True,
                padding="max_length",
                max_length=block_size,
            )
            return {
                "input_ids": tokens["input_ids"],
                "attention_mask": tokens["attention_mask"],
                "labels": tokens["input_ids"],
            }

        tokenized_chunk = dataset_chunk.map(
            tokenize_function,
            batched=True,
            remove_columns=["text"],
            num_proc=1,
            load_from_cache_file=False
        )

        datasets_parts.append(tokenized_chunk)
        del dataset_chunk, tokenized_chunk
        import gc
        gc.collect()

    full_dataset = concatenate_datasets(datasets_parts)
    tokenized = full_dataset.train_test_split(test_size=eval_ratio, seed=42)
    print("Salvando dataset tokenizado no disco...")
    tokenized.save_to_disk(cached_dataset_path)
    with open(hash_path, "w") as f:
        f.write(current_hash)

# === REDUZIR TAMANHO DO DATASET PARA TREINAMENTO RÁPIDO ===
tokenized["train"] = tokenized["train"].shuffle(seed=42).select(range(int(len(tokenized["train"]) * train_frac)))

# === FORMATAÇÃO PARA OTIMIZAR MEMÓRIA ===
tokenized.set_format("torch")

# === TESTE SIMPLES COM FALLBACK ===
print("Exemplo do dataset (teste de integridade):")
try:
    print(tokenized["train"][0])
    print("Primeiros 5 exemplos:")
    for i in range(5):
        print(tokenized["train"][i])
except Exception as e:
    print(f"[FALHA] Não foi possível acessar os exemplos diretamente: {e}")
    print("Tentando fallback mais leve:")
    try:
        print(f"Número de exemplos no dataset de treino: {len(tokenized['train'])}")
    except Exception as ee:
        print(f"Fallback também falhou: {ee}")

# === COLLATOR ===
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False,
)

# === ARGUMENTOS DE TREINAMENTO ===
training_args = TrainingArguments(
    output_dir=output_dir,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=32,
    eval_strategy="epoch",
    save_strategy="epoch",
    num_train_epochs=num_train_epochs,
    weight_decay=0.01,
    logging_dir=f"{output_dir}/logs",
    logging_steps=50,
    save_total_limit=3,
    load_best_model_at_end=True,
    fp16=torch.cuda.is_available(),
    report_to="none",
    remove_unused_columns=False,
    dataloader_num_workers=4,
)

# === CALLBACKS ===
callbacks = [EarlyStoppingCallback(early_stopping_patience=2)]

# === TREINAMENTO ===
print("Iniciando o treinamento...")
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized["train"],
    eval_dataset=tokenized["test"],
    tokenizer=tokenizer,
    data_collator=data_collator,
    callbacks=callbacks,
)

train_result = trainer.train()
if trainer.is_world_process_zero():
    trainer.save_model(f"{output_dir}/final")
    tokenizer.save_pretrained(f"{output_dir}/final")

# === PLOTAGEM DE ERRO ===
logs = trainer.state.log_history
train_loss = [x["loss"] for x in logs if "loss" in x and "eval_loss" not in x]
eval_loss = [x["eval_loss"] for x in logs if "eval_loss" in x]

plt.plot(train_loss, label="Train Loss")
plt.plot(eval_loss, label="Eval Loss")
plt.xlabel("Step")
plt.ylabel("Loss")
plt.legend()
plt.title("GPT2 Fine-Tuning Loss")
plt.savefig(f"{output_dir}/loss_plot.png")
plt.show()