import os
import time
import shutil
import hashlib
from typing import List
import re  # <-- (novo) para split robusto em vírgula/linha

import torch
from datasets import Dataset, load_from_disk, concatenate_datasets
from transformers import (
    GPT2Tokenizer,
    GPT2LMHeadModel,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
)
import matplotlib.pyplot as plt

# =============================================================
# Configurações básicas
# =============================================================
model_name = "gpt2"
# Altere este caminho se necessário
input_txt_path = "/home/vberta/projects/def-aloise/vberta/Paper3/logs/part_3.log"
output_dir = "./gpt2_log_model"
block_size = 128
num_train_epochs = 3
train_frac = 0.9  # fração das linhas do arquivo de entrada para tokenizar/treinar
eval_ratio = 0.1

# Onde salvar o cache tokenizado: prioriza disco local do nó (SLURM_TMPDIR)
SCRATCH = os.environ.get("SLURM_TMPDIR")
cache_root = SCRATCH or "."
cached_dataset_path = os.path.join(cache_root, "tokenized_dataset")
hash_path = os.path.join(cached_dataset_path, ".hash")
ready_path = os.path.join(cached_dataset_path, ".ready")
tmp_path = cached_dataset_path + ".tmp"

# Tamanho do chunk em linhas para tokenização incremental (IO/ram friendly)
chunk_size = 2_000_000

# =============================================================
# Utilidades de rank/processo
# =============================================================

def is_main_proc() -> bool:
    try:
        if "RANK" in os.environ:
            return int(os.environ["RANK"]) == 0
    except Exception:
        pass
    try:
        import torch.distributed as dist
        if dist.is_available() and dist.is_initialized():
            return dist.get_rank() == 0
    except Exception:
        pass
    return True

# =============================================================
# Funções auxiliares para cache atômico + marcador .ready
# =============================================================

def cached_dataset_ok(path: str) -> bool:
    if not os.path.isdir(path):
        return False
    if os.path.isfile(os.path.join(path, "dataset_dict.json")):
        return True
    if os.path.isfile(os.path.join(path, "dataset_info.json")):
        return True
    return False


def compute_file_hash(path: str) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def cache_is_ready(path: str, hashfile: str, expected_hash: str) -> bool:
    if not cached_dataset_ok(path):
        return False
    if not (os.path.isfile(hashfile) and os.path.isfile(os.path.join(path, ".ready"))):
        return False
    try:
        with open(hashfile, "r") as f:
            return f.read().strip() == expected_hash
    except Exception:
        return False


def wait_for_cache_ready(path: str, hashfile: str, expected_hash: str, timeout_s: int = 7200):
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        if cache_is_ready(path, hashfile, expected_hash):
            time.sleep(2.0)
            return True
        time.sleep(2.0)
    return False

# =============================================================
# Leitura e “explosão” em frases (vírgula/linha)
# =============================================================

def read_lines_fraction(path: str, frac: float) -> List[str]:
    assert 0 < frac <= 1.0, "train_frac deve estar entre (0, 1]"
    phrases: List[str] = []
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for raw in f:
            s = raw.strip()
            if not s:
                continue
            # Divide por vírgulas e quebras de linha (1+ vírgulas tratadas como um único separador)
            parts = [p.strip() for p in re.split(r"[,\n]+", s) if p.strip()]
            phrases.extend(parts)
    if frac < 1.0:
        n = max(1, int(len(phrases) * frac))
        phrases = phrases[:n]
    return phrases

# =============================================================
# Inicialização de dispositivo/tokenizer/modelo
# =============================================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Usando dispositivo: {device}")

_tokenizer = GPT2Tokenizer.from_pretrained(model_name)
_tokenizer.pad_token = _tokenizer.eos_token

# =============================================================
# Pipeline de tokenização com cache atômico
# =============================================================

def tokenize_function(examples):
    return _tokenizer(examples["text"], truncation=True, max_length=block_size)


def build_or_load_tokenized_dataset():
    os.makedirs(os.path.dirname(cached_dataset_path) or ".", exist_ok=True)

    # inclui assinatura do split para invalidar cache antigo
    split_signature = "split=comma+newline_v1"
    current_hash = compute_file_hash(input_txt_path) + f"_block{block_size}_frac{train_frac}_{split_signature}"

    tokenized = None
    if os.path.exists(hash_path) and cached_dataset_ok(cached_dataset_path) and os.path.isfile(ready_path):
        try:
            with open(hash_path, "r") as f:
                if f.read().strip() == current_hash:
                    print("Carregando dataset tokenizado do disco...")
                    tokenized = load_from_disk(cached_dataset_path)
        except Exception:
            tokenized = None

    if tokenized is not None:
        return tokenized

    if is_main_proc():
        print("Tokenizando dataset em chunks (rank 0)...")
        lines = read_lines_fraction(input_txt_path, train_frac)
        total = len(lines)
        print(f"Total de frases consideradas: {total}")

        datasets_parts = []
        for i in range(0, total, chunk_size):
            j = min(i + chunk_size, total)
            chunk_lines = lines[i:j]
            ds_chunk = Dataset.from_dict({"text": chunk_lines})
            tok_chunk = ds_chunk.map(
                tokenize_function,
                batched=True,
                remove_columns=["text"],
                num_proc=1,
                load_from_cache_file=False,
                writer_batch_size=10_000,
                desc=f"Tokenizing {i//chunk_size + 1}/{(total - 1)//chunk_size + 1}",
            )
            datasets_parts.append(tok_chunk)

        full_dataset = concatenate_datasets(datasets_parts)
        tokenized = full_dataset.train_test_split(test_size=eval_ratio, seed=42)

        if os.path.isdir(tmp_path):
            shutil.rmtree(tmp_path)
        os.makedirs(os.path.dirname(tmp_path) or ".", exist_ok=True)
        print(f"Salvando dataset tokenizado (tmp): {tmp_path}")
        tokenized.save_to_disk(tmp_path)

        if os.path.isdir(cached_dataset_path):
            shutil.rmtree(cached_dataset_path)
        os.rename(tmp_path, cached_dataset_path)

        with open(hash_path, "w") as f:
            f.write(current_hash)
        open(ready_path, "w").close()
        time.sleep(2.0)
    else:
        print("Aguardando o rank 0 preparar o cache...")
        ok = wait_for_cache_ready(cached_dataset_path, hash_path, current_hash, timeout_s=7200)
        if not ok:
            raise RuntimeError(
                f"Timeout esperando cache pronto em {cached_dataset_path}. Verifique se o rank 0 conseguiu salvar o dataset."
            )

    print("Carregando dataset tokenizado do disco (após preparo)...")
    return load_from_disk(cached_dataset_path)


# =============================================================
# Treinamento
# =============================================================

def _bf16_supported() -> bool:
    try:
        return torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    except Exception:
        return False

def main():
    tokenized = build_or_load_tokenized_dataset()

    model = GPT2LMHeadModel.from_pretrained(model_name)

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=_tokenizer,
        mlm=False,
    )

    # Força NCCL para evitar backend MPI
    # (evita o erro "IndexError: map::at" em alguns ambientes MPI)
    os.environ.setdefault("NCCL_ASYNC_ERROR_HANDLING", "1")

    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=num_train_epochs,
        evaluation_strategy="steps",   # mantido p/ compat; avisa deprecation apenas
        eval_steps=1000,
        logging_steps=200,
        save_steps=1000,
        save_total_limit=2,
        learning_rate=5e-5,
        weight_decay=0.01,
        warmup_steps=100,
        report_to=[],
        dataloader_pin_memory=True,
        gradient_accumulation_steps=1,
        bf16=_bf16_supported(),
        ddp_find_unused_parameters=False,   # <-- (novo) mais estável p/ GPT-2
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["test"],
        data_collator=data_collator,
        tokenizer=_tokenizer,
    )

    train_result = trainer.train()

    if is_main_proc():
        os.makedirs(output_dir, exist_ok=True)
        trainer.save_model(output_dir)
        _tokenizer.save_pretrained(output_dir)

        losses = [x["loss"] for x in trainer.state.log_history if "loss" in x]
        if losses:
            plt.figure()
            plt.plot(losses)
            plt.xlabel("step")
            plt.ylabel("loss")
            plt.title("Training loss")
            plt.savefig(os.path.join(output_dir, "loss.png"), bbox_inches="tight")
            plt.close()

    try:
        import torch.distributed as dist
        if dist.is_available() and dist.is_initialized():
            dist.barrier()
            dist.destroy_process_group()
    except Exception:
        pass


if __name__ == "__main__":
    main()
