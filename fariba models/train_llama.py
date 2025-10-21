import torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, TrainingArguments, Trainer

# ==== CONFIGURATION ====
#input_file = "all_sequences.txt"
input_file = "sequences_mini.txt"
model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"  # smaller model for GPU training
output_dir = "./llama_finetuned_ids"
seq_length = 30
epochs = 3
batch_size = 2
learning_rate = 2e-5
device = "cuda" if torch.cuda.is_available() else "cpu"

# ==== LOAD SEQUENCES ====
def load_sequences(path):
    sequences = []
    with open(path) as f:
        for line in f:
            tokens = [int(x) for x in line.strip().split()]
            if len(tokens) > 1:
                sequences.append(tokens)
    return sequences

data = load_sequences(input_file)
print(f"✅ Loaded {len(data)} sequences")

# ==== DETERMINE VOCAB SIZE ====
vocab_size = max(max(seq) for seq in data) + 1
print(f"✅ Vocabulary size detected: {vocab_size}")

# ==== PAD OR TRUNCATE ====
def pad_or_truncate(seq, max_len):
    if len(seq) >= max_len:
        return seq[:max_len]
    else:
        return seq + [0] * (max_len - len(seq))

X = [pad_or_truncate(seq, seq_length) for seq in data]

# ==== CREATE DATASET ====
dataset = Dataset.from_dict({
    "input_ids": X,
    "labels": X,
})

# ==== LOAD MODEL ====
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32
).to(device)


# Resize vocabulary if necessary
if model.config.vocab_size < vocab_size:
    model.resize_token_embeddings(vocab_size)
    print(f"🔧 Model resized to vocab_size={vocab_size}")

# ==== TRAINING ARGUMENTS ====
training_args = TrainingArguments(
    output_dir=output_dir,
    overwrite_output_dir=True,
    per_device_train_batch_size=batch_size,
    num_train_epochs=epochs,
    learning_rate=learning_rate,
    logging_steps=10,
    save_strategy="epoch",
    bf16=True,                   # ✅ usar bfloat16, ideal no H100
    fp16=False,                  # ❌ desativa fp16 para evitar o erro
    max_grad_norm=1.0,
    ddp_find_unused_parameters=False,
    report_to="none",
)


# ==== TRAIN ====
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
)

print("🚀 Starting training...")
trainer.train()

# ==== SAVE MODEL ====
model.save_pretrained(output_dir)
print(f"✅ Training complete. Model saved in {output_dir}")
