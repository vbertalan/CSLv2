#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import argparse
from pathlib import Path
import regex as re
import sentencepiece as spm
from collections import Counter

from datasets import load_dataset, DatasetDict
from transformers import (
    T5Tokenizer,              # fluxo SentencePiece (original)
    GPTNeoXConfig,
    GPTNeoXForCausalLM,
    DataCollatorForLanguageModeling,
    TrainingArguments,
    Trainer,
    PreTrainedTokenizerFast,  # fluxo 1-linha=1-token
)
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Split
from tokenizers.normalizers import Sequence as NormalizerSequence, Strip

import torch


# ==========================
# Utilidades de hardware
# ==========================
def print_gpu_info():
    if torch.cuda.is_available():
        n = torch.cuda.device_count()
        print(f"[GPU] CUDA disponível: {n} GPU(s)")
        for i in range(n):
            print(f"  - GPU {i}: {torch.cuda.get_device_name(i)}")
    else:
        print("[GPU] CUDA NÃO disponível (rodando em CPU)")


# ==========================
# SentencePiece (fluxo original)
# ==========================
def train_sentencepiece(
    corpus_path: str,
    sp_prefix: str,
    vocab_size: int,
    character_coverage: float = 1.0,
    model_type: str = "unigram",
    input_sentence_size: int = 0,
    shuffle_input_sentence: bool = True,
):
    spm.SentencePieceTrainer.Train(
        input=corpus_path,
        model_prefix=sp_prefix,
        vocab_size=vocab_size,
        character_coverage=character_coverage,
        model_type=model_type,
        input_sentence_size=input_sentence_size,
        shuffle_input_sentence=shuffle_input_sentence,
        unk_id=0,
        bos_id=1,
        eos_id=2,
        pad_id=3,
        hard_vocab_limit=False,
    )
    print(f"[SPM] Treinado em {sp_prefix}.model / {sp_prefix}.vocab")


def build_sp_tokenizer(sp_model_path: str):
    assert os.path.exists(sp_model_path), f"SP model não encontrado: {sp_model_path}"
    tok = T5Tokenizer(
        vocab_file=sp_model_path,   # <<<<<< OBRIGATÓRIO
        bos_token="<s>",
        eos_token="</s>",
        unk_token="<unk>",
        pad_token="<pad>",
    )
    tok.add_special_tokens({"additional_special_tokens": ["<SENT_END>"]})
    print("[TOK] <SENT_END> id =", tok.convert_tokens_to_ids("<SENT_END>"))
    return tok


# ==========================
# Tokenizer 1-linha=1-token
# ==========================
def build_line_tokenizer(input_path: str, out_dir: Path, min_line_freq: int = 1):
    """
    Constrói um tokenizer WordLevel em que cada linha (inteira) do arquivo é um token.
    - Linhas com frequência < min_line_freq são mapeadas para <unk>.
    - Salva em out_dir / "line_tok".
    """
    # 1) Contar linhas
    counter = Counter()
    with open(input_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.rstrip("\n")
            if line:
                counter[line] += 1

    # 2) Montar vocabulário
    vocab = {}
    specials = ["<pad>", "<unk>", "<s>", "</s>"]
    for i, sp in enumerate(specials):
        vocab[sp] = i
    next_id = len(vocab)

    kept = 0
    for line, freq in counter.items():
        if freq >= min_line_freq and line not in vocab:
            vocab[line] = next_id
            next_id += 1
            kept += 1

    print(f"[TOK] Linhas totais: {len(counter)} | min_line_freq={min_line_freq} | mantidas no vocab: {kept}")

    # 3) Tokenizer WordLevel
    model = WordLevel(vocab=vocab, unk_token="<unk>")
    tok = Tokenizer(model)

    # Normalizador (apenas strip; NÃO use lowercase se suas linhas são case-sensitive)
    tok.normalizer = NormalizerSequence([Strip()])

    # PreTokenizer: dividir por quebras de linha; ao tokenizar strings com múltiplas linhas,
    # cada linha vira 1 token (desde que esteja no vocabulário; senão -> <unk>)
    tok.pre_tokenizer = Split("\n", "removed")

    # 4) Wrap HF
    fast_tok = PreTrainedTokenizerFast(
        tokenizer_object=tok,
        bos_token="<s>",
        eos_token="</s>",
        unk_token="<unk>",
        pad_token="<pad>",
    )

    # 5) Salvar
    tdir = out_dir / "line_tok"
    tdir.mkdir(parents=True, exist_ok=True)
    fast_tok.save_pretrained(str(tdir))
    print(f"[TOK] Tokenizer de linhas salvo em: {tdir}")
    print(f"[TOK] Vocab size (incl. especiais) = {len(fast_tok)}")
    return fast_tok


# ==========================
# Dataset
# ==========================

def prepare_stream_dataset(input_path: str, tokenizer, block_size: int, num_proc: int = 1, line_as_token: bool = False, eval_ratio: float = 0.1):
    raw = load_dataset("text", data_files={"train": input_path})
    # tokenização
    def tok_fn(batch):
        return tokenizer(
            batch["text"],
            add_special_tokens=False,
            return_attention_mask=True,
            return_token_type_ids=False,
        )

    if line_as_token:
        mapped = raw["train"].map(tok_fn, batched=True, remove_columns=["text"], num_proc=num_proc)
    else:
        SENT_END = "<SENT_END>"
        def add_delim(example):
            txt = (example["text"] or "").strip()
            return {"text": f"{txt} {SENT_END} "} if txt else {"text": ""}
        with_delim = raw.map(add_delim, num_proc=num_proc)
        mapped = with_delim["train"].map(tok_fn, batched=True, remove_columns=["text"], num_proc=num_proc)

    # split ANTES de agrupar
    spl = mapped.train_test_split(test_size=eval_ratio, shuffle=True, seed=42)
    train_mapped, eval_mapped = spl["train"], spl["test"]

    def group_texts(examples):
        concatenated = []
        for seq in examples["input_ids"]:
            concatenated.extend(seq)
        total_len = (len(concatenated) // block_size) * block_size
        concatenated = concatenated[:total_len]
        input_ids = [concatenated[i:i + block_size] for i in range(0, total_len, block_size)]
        attention_mask = [[1] * len(x) for x in input_ids]
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": [ids.copy() for ids in input_ids],
        }

    train_ds = train_mapped.map(group_texts, batched=True, batch_size=1000, num_proc=num_proc, remove_columns=train_mapped.column_names)
    eval_ds  = eval_mapped.map(group_texts,  batched=True, batch_size=1000, num_proc=num_proc, remove_columns=eval_mapped.column_names)

    # prints diagnósticos
    print(f"[DATA] train_batches={len(train_ds)}  eval_batches={len(eval_ds)}  (block_size={block_size})")
    if len(train_ds) == 0:
        raise ValueError("Treino vazio após agrupar. Diminua --block_size.")
    if len(eval_ds) == 0:
        print("[WARN] Validação vazia; aumente eval_ratio ou diminua block_size.")

    return DatasetDict({"train": train_ds, "eval": eval_ds})


    def group_texts(examples):
        concatenated = []
        for seq in examples["input_ids"]:
            concatenated.extend(seq)

        # block_size = "quantas sentenças/linhas por bloco" quando line_as_token=True
        total_len = (len(concatenated) // block_size) * block_size
        concatenated = concatenated[:total_len]

        input_ids = [concatenated[i:i + block_size] for i in range(0, total_len, block_size)]
        attention_mask = [[1] * len(x) for x in input_ids]

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": [ids.copy() for ids in input_ids],
        }

    lm_ds = mapped.map(
        group_texts,
        batched=True,
        batch_size=1000,
        num_proc=num_proc,
        remove_columns=mapped.column_names,  # <<< ponto-chave
    )
    return DatasetDict({"train": lm_ds})



# ==========================
# Modelo
# ==========================
def build_model(tokenizer, hidden_size: int, n_layers: int, n_heads: int,
                intermediate_size: int, max_position_embeddings: int):
    config = GPTNeoXConfig(
        vocab_size=len(tokenizer),
        hidden_size=hidden_size,
        num_hidden_layers=n_layers,
        num_attention_heads=n_heads,
        intermediate_size=intermediate_size,
        max_position_embeddings=max_position_embeddings,
    )
    model = GPTNeoXForCausalLM(config)
    return model


# ==========================
# AMP / TF32 / compile
# ==========================
def resolve_amp(amp_arg: str):
    """
    amp_arg in {"auto","bf16","fp16","off"}
    - "auto": usa bf16 se suportado, senão fp16 se CUDA, senão off.
    """
    amp_arg = amp_arg.lower()
    if amp_arg == "auto":
        if torch.cuda.is_available():
            if torch.cuda.is_bf16_supported():
                return {"bf16": True, "fp16": False}
            else:
                return {"bf16": False, "fp16": True}
        return {"bf16": False, "fp16": False}
    elif amp_arg == "bf16":
        return {"bf16": True, "fp16": False}
    elif amp_arg == "fp16":
        return {"bf16": False, "fp16": True}
    else:
        return {"bf16": False, "fp16": False}


def maybe_enable_tf32(enable: bool):
    if enable and torch.cuda.is_available():
        try:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            print("[TF32] Ativado")
        except Exception as e:
            print(f"[TF32] Falha ao ativar: {e}")


def maybe_compile(model, do_compile: bool):
    if do_compile and hasattr(torch, "compile"):
        try:
            model = torch.compile(model)
            print("[compile] torch.compile ativado")
        except Exception as e:
            print(f"[compile] Falha ao compilar: {e}")
    return model


# ==========================
# Main
# ==========================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_file", required=True, help="Arquivo com 1 frase por linha (logs etc.)")
    ap.add_argument("--out_dir", default="./neox_sent", help="Saída (modelo/tokenizer/resultados)")
    ap.add_argument("--vocab_size", type=int, default=8000)
    ap.add_argument("--block_size", type=int, default=1024)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--grad_accum", type=int, default=1)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--weight_decay", type=float, default=0.01)
    ap.add_argument("--hidden_size", type=int, default=768)
    ap.add_argument("--n_layers", type=int, default=12)
    ap.add_argument("--n_heads", type=int, default=12)
    ap.add_argument("--intermediate_size", type=int, default=3072)
    ap.add_argument("--max_pos", type=int, default=2048)

    # GPU-related
    ap.add_argument("--amp", default="auto", choices=["auto", "bf16", "fp16", "off"])
    ap.add_argument("--tf32", action="store_true")
    ap.add_argument("--gradient_checkpointing", action="store_true")
    ap.add_argument("--compile", action="store_true")
    ap.add_argument("--dataloader_workers", type=int, default=2)

    # NOVO: modo 1-linha=1-token
    ap.add_argument("--line_as_token", action="store_true",
                    help="Se definido, usa tokenizer WordLevel em que cada linha vira 1 token (substitui SentencePiece).")
    ap.add_argument("--min_line_freq", type=int, default=1,
                    help="Frequência mínima para incluir uma linha no vocabulário (apenas com --line_as_token).")

    args = ap.parse_args()

    print_gpu_info()
    maybe_enable_tf32(args.tf32)

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    # === 1) Tokenizer ===
    if args.line_as_token:
        tokenizer = build_line_tokenizer(args.input_file, out, min_line_freq=args.min_line_freq)
    else:
        # SentencePiece + T5Tokenizer (fluxo original)
        sp_prefix = str(out / "sp_logs")
        if not Path(sp_prefix + ".model").exists():
            train_sentencepiece(
                corpus_path=args.input_file,
                sp_prefix=sp_prefix,
                vocab_size=args.vocab_size,
                character_coverage=1.0,
                model_type="unigram",
                input_sentence_size=0,
                shuffle_input_sentence=True,
            )
        tokenizer = build_sp_tokenizer(sp_prefix + ".model")

    # === 2) Dataset ===
    num_proc = max(1, args.dataloader_workers)
    ds = prepare_stream_dataset(
        args.input_file,
        tokenizer,
        block_size=args.block_size,
        num_proc=num_proc,
        line_as_token=args.line_as_token,
        eval_ratio=0.2,  # 10% validação
    )

    # === 3) Modelo ===
    model = build_model(
        tokenizer,
        hidden_size=args.hidden_size,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        intermediate_size=args.intermediate_size,
        max_position_embeddings=args.max_pos,
    )

    # Opcional: gradient checkpointing (memória ↓, compute ↑)
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        print("[GC] Gradient checkpointing habilitado")

    model = maybe_compile(model, args.compile)

    # === 4) Treino ===
    amp_flags = resolve_amp(args.amp)
    print(f"[AMP] Config: {amp_flags}  (auto escolhe bf16 se suportado, senão fp16)")

    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    training_args = TrainingArguments(
        output_dir=str(out / "ckpts"),
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        num_train_epochs=args.epochs,
        weight_decay=args.weight_decay,

        # >>> mais granularidade
        logging_steps=1,
        logging_first_step=True,
        eval_strategy="steps",
        eval_steps=50,              # ajuste conforme o tamanho do dataset
        save_strategy="no",         # opcional: evitar save frequente
        report_to="none",

        dataloader_num_workers=args.dataloader_workers,
        bf16=amp_flags["bf16"],
        fp16=amp_flags["fp16"],
        tf32=args.tf32,
        ddp_find_unused_parameters=False,
        save_safetensors=True,
    )


    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=ds["train"],
        eval_dataset=ds["eval"],     # <<<<<< agora tem eval!
        data_collator=collator,
        tokenizer=tokenizer,
    )

    trainer.train()

    # === 5) Plot do erro (loss) ===
    import matplotlib.pyplot as plt

    def is_main_process():
        try:
            return trainer.is_world_process_zero()
        except Exception:
            return True

    if is_main_process():
        hist = trainer.state.log_history
        train_steps, train_losses = [], []
        eval_steps, eval_losses = [], []

        for i, rec in enumerate(hist):
            if "loss" in rec:
                train_steps.append(rec.get("step", i))
                train_losses.append(rec["loss"])
            if "eval_loss" in rec:
                eval_steps.append(rec.get("step", i))
                eval_losses.append(rec["eval_loss"])

        plt.figure(figsize=(7, 4))
        if train_losses:
            plt.plot(train_steps, train_losses, label="train_loss")
        if eval_losses:
            plt.plot(eval_steps, eval_losses, label="eval_loss", linestyle="--")
        plt.xlabel("step")
        plt.ylabel("loss")
        plt.yscale("log") 
        plt.title("Training / Eval Loss")
        plt.grid(True, alpha=0.3)
        plt.legend()
        out_plot = Path(args.out_dir) / "loss_curve.png"
        plt.tight_layout()
        plt.savefig(out_plot, dpi=150)
        print(f"[OK] Gráfico de loss salvo em: {out_plot}")

    # === 6) Salvar modelo e tokenizer final ===
    save_dir = out / "final"
    save_dir.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(save_dir))
    tokenizer.save_pretrained(str(save_dir))
    print(f"[OK] Modelo e tokenizer salvos em: {save_dir}")


if __name__ == "__main__":
    main()

# Exemplo:
# Modo 1 linha = 1 token
# python neoxv2.py --input_file "part_3.log" --out_dir "part_3_neoxv2" --block_size 512 --epochs 20 --batch_size 2 --line_as_token --min_line_freq 1
#
# Modo original (SentencePiece + <SENT_END>)
# python neox.py --input_file "part_3.log" --out_dir "part_3_neox_spm" --vocab_size 11000 --block_size 768 --epochs 3 --batch_size 2
