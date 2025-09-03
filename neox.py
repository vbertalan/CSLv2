#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import argparse
from pathlib import Path
import sentencepiece as spm
from datasets import load_dataset, DatasetDict
from transformers import (
    T5Tokenizer,   # ⬅️ adicione
    GPTNeoXConfig,
    GPTNeoXForCausalLM,
    DataCollatorForLanguageModeling,
    TrainingArguments,
    Trainer,
)
import torch


def print_gpu_info():
    if torch.cuda.is_available():
        n = torch.cuda.device_count()
        print(f"[GPU] CUDA disponível: {n} GPU(s)")
        for i in range(n):
            print(f"  - GPU {i}: {torch.cuda.get_device_name(i)}")
    else:
        print("[GPU] CUDA NÃO disponível (rodando em CPU)")


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

def build_tokenizer(sp_model_path: str):
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


def prepare_stream_dataset(input_path: str, tokenizer, block_size: int, num_proc: int = 1):
    raw = load_dataset("text", data_files={"train": input_path})

    SENT_END = "<SENT_END>"

    def add_delim(example):
        txt = (example["text"] or "").strip()
        if not txt:
            return {"text": ""}
        return {"text": f"{txt} {SENT_END} "}

    with_delim = raw.map(add_delim, num_proc=num_proc)

    def tok_fn(batch):
        return tokenizer(batch["text"])

    tok = with_delim["train"].map(tok_fn, batched=True, remove_columns=["text"], num_proc=num_proc)

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

    lm_ds = tok.map(group_texts, batched=True, batch_size=1000, num_proc=num_proc)
    return DatasetDict({"train": lm_ds})


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
    args = ap.parse_args()

    print_gpu_info()
    maybe_enable_tf32(args.tf32)

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    # 1) SentencePiece
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

    # 2) Tokenizer
    tokenizer = build_tokenizer(sp_prefix + ".model")

    # 3) Dataset (use num_proc para acelerar o preprocessing)
    num_proc = max(1, args.dataloader_workers)
    ds = prepare_stream_dataset(args.input_file, tokenizer, block_size=args.block_size, num_proc=num_proc)

    # 4) Modelo
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

    # 5) Treino
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
        logging_steps=50,
        save_steps=2000,
        save_total_limit=2,
        report_to="none",
        dataloader_num_workers=args.dataloader_workers,
        bf16=amp_flags["bf16"],
        fp16=amp_flags["fp16"],
        tf32=args.tf32,
        ddp_find_unused_parameters=False,  # útil em DDP
        save_safetensors=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=ds["train"],
        data_collator=collator,
        tokenizer=tokenizer,
    )

    trainer.train()

    # 5) Plota Erro

      # === Plot do erro (loss) ===
    import matplotlib.pyplot as plt

    # só no processo principal em DDP
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
        plt.title("Training / Eval Loss")
        plt.grid(True, alpha=0.3)
        plt.legend()
        out_plot = Path(args.out_dir) / "loss_curve.png"
        plt.tight_layout()
        plt.savefig(out_plot, dpi=150)
        print(f"[OK] Gráfico de loss salvo em: {out_plot}")


    # 6) Salvar
    save_dir = out / "final"
    save_dir.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(save_dir))
    tokenizer.save_pretrained(str(save_dir))
    print(f"[OK] Modelo e tokenizer salvos em: {save_dir}")


if __name__ == "__main__":
    main()

# python neox.py --input_file "part_3.log" --out_dir "part_3_neox" --vocab_size 11000 --block_size 768 --epochs 20 --batch_size 2 

# python neox.py --input_file "synthetic_sequences.txt" --out_dir "neoxv1-synthetic" --vocab_size 32000 --block_size 768 --epochs 20 --batch_size 2 
  
 # --grad_accum $GRAD_ACCUM \
 # --lr $LR \
 # --amp auto \
 #--tf32 \
 # --gradient_checkpointing \
 # --compile \
 # --dataloader_workers $SLURM_CPUS_PER_TASK

# 