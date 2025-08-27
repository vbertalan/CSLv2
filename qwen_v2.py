#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import argparse
from pathlib import Path
import sentencepiece as spm
from datasets import load_dataset, DatasetDict
from transformers import (
    T5Tokenizer,
    PreTrainedTokenizerFast,
    Qwen3Config,
    Qwen3ForCausalLM,
    DataCollatorForLanguageModeling,
    TrainingArguments,
    Trainer,
)
import torch
import torch.distributed as dist
import matplotlib.pyplot as plt


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

def train_sentencepiece_line_tokens(
    corpus_path: str,
    sp_prefix: str,
    vocab_size: int = None,
):
    """
    Treina um SentencePiece onde cada linha inteira é tratada como UM token atômico.
    Ideal para logs/templates muito repetitivos.

    - corpus_path: caminho do arquivo de texto (1 frase por linha)
    - sp_prefix: prefixo para salvar .model e .vocab
    - vocab_size: opcional, se None será calculado como nº de linhas únicas
    """
    # contar frases únicas
    with open(corpus_path, "r", encoding="utf-8") as f:
        unique_lines = {line.strip() for line in f if line.strip()}
    n_unique = len(unique_lines)
    if vocab_size is None:
        vocab_size = n_unique + 10  # margem para <unk>, <pad>, <bos>, <eos>, etc.

    print(f"[SPM] Treinando com {n_unique} frases únicas → vocab_size={vocab_size}")

    spm.SentencePieceTrainer.Train(
        input=corpus_path,
        model_prefix=sp_prefix,
        vocab_size=vocab_size,
        model_type="unigram",          # unigram é mais flexível
        character_coverage=1.0,        # cobre todos os caracteres
        split_by_whitespace=False,     # não dividir por espaço
        byte_fallback=False,           # não quebrar em bytes
        unk_id=0,
        bos_id=1,
        eos_id=2,
        pad_id=3,
        hard_vocab_limit=False,
    )
    print(f"[SPM] Modelo salvo: {sp_prefix}.model / {sp_prefix}.vocab")


def build_tokenizer(sp_model_path: str):
    """
    Constrói um tokenizer a partir do modelo SentencePiece treinado no modo
    'cada linha = 1 token'. Usa PreTrainedTokenizerFast para integração limpa
    com HuggingFace.
    """
    assert os.path.exists(sp_model_path), f"SP model não encontrado: {sp_model_path}"

    tokenizer = PreTrainedTokenizerFast(
        tokenizer_file=sp_model_path,
        bos_token="<s>",
        eos_token="</s>",
        unk_token="<unk>",
        pad_token="<pad>",
    )

    # Se quiser tokens adicionais específicos
    special_tokens = {"additional_special_tokens": ["<SENT_END>"]}
    tokenizer.add_special_tokens(special_tokens)

    print("[TOK] ids de controle:")
    print("  <s> =", tokenizer.convert_tokens_to_ids("<s>"))
    print("  </s> =", tokenizer.convert_tokens_to_ids("</s>"))
    print("  <unk> =", tokenizer.convert_tokens_to_ids("<unk>"))
    print("  <pad> =", tokenizer.convert_tokens_to_ids("<pad>"))
    print("  <SENT_END> =", tokenizer.convert_tokens_to_ids("<SENT_END>"))

    return tokenizer



from datasets import load_dataset, DatasetDict

def prepare_stream_dataset(input_path: str, tokenizer, block_size: int, num_proc: int = 1):
    """
    Prepara dataset em que cada linha do arquivo é 1 token (ID).
    Junta várias sentenças até atingir block_size.
    """

    raw = load_dataset("text", data_files={"train": input_path})

    def tok_fn(batch):
        # Cada sentença vira 1 ID
        enc = tokenizer(batch["text"])
        return {"input_ids": enc["input_ids"]}

    tok = raw["train"].map(
        tok_fn,
        batched=True,
        remove_columns=["text"],
        num_proc=num_proc
    )

    def group_sentences(examples):
        concatenated = []
        for seq in examples["input_ids"]:
            concatenated.extend(seq)  # cada seq aqui tem 1 único ID
        total_len = (len(concatenated) // block_size) * block_size
        concatenated = concatenated[:total_len]

        input_ids = [
            concatenated[i:i + block_size]
            for i in range(0, total_len, block_size)
        ]
        attention_mask = [[1] * len(x) for x in input_ids]

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": [ids.copy() for ids in input_ids],
        }

    lm_ds = tok.map(
        group_sentences,
        batched=True,
        batch_size=1000,
        num_proc=num_proc
    )

    return DatasetDict({"train": lm_ds})



def build_model(tokenizer, hidden_size: int, n_layers: int, n_heads: int,
                intermediate_size: int, max_position_embeddings: int):
    config = Qwen3Config(
        vocab_size=len(tokenizer),
        hidden_size=hidden_size,
        num_hidden_layers=n_layers,
        num_attention_heads=n_heads,
        intermediate_size=intermediate_size,
        max_position_embeddings=max_position_embeddings,
        num_key_value_heads=n_heads,  # ajuste aqui para igualar ou dividir num_attention_heads
    )
    model = Qwen3ForCausalLM(config)
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
        train_sentencepiece_line_tokens(
            corpus_path=args.input_file,
            sp_prefix=sp_prefix,
            vocab_size=args.vocab_size,  # pode deixar None que ele calcula
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

    # treino estável com Trainer/DDP
    model.config.use_cache = False

    # Opcional: gradient checkpointing (memória ↓, compute ↑)
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        print("[GC] Gradient checkpointing habilitado")

    model = maybe_compile(model, args.compile)

    # 5) Treino
    amp_flags = resolve_amp(args.amp)
    print(f"[AMP] Config: {amp_flags} (auto escolhe bf16 se suportado, senão fp16)")

    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    training_args = TrainingArguments(
        output_dir=str(out / "ckpts"),
        per_device_train_batch_size=args.batch_size,   # nº de blocos (cada bloco = block_size sentenças)
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        num_train_epochs=args.epochs,
        weight_decay=args.weight_decay,
        logging_strategy="steps",
        logging_steps=50,
        save_steps=2000,
        save_total_limit=2,
        report_to="none",
        dataloader_num_workers=args.dataloader_workers,
        bf16=amp_flags["bf16"],
        fp16=amp_flags["fp16"],
        tf32=args.tf32,
        ddp_find_unused_parameters=False,
        save_safetensors=True,
        remove_unused_columns=False,
    )

    print(f"[ARGS] batch_size={args.batch_size} × grad_accum={args.grad_accum} "
            f"→ batch efetivo = {args.batch_size * args.grad_accum * torch.cuda.device_count()} blocos "
            f"(cada bloco = {args.block_size} sentenças)")

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=ds["train"],
        data_collator=collator,
        processing_class=tokenizer,  # futuro substituto de `tokenizer=...`
    )

    trainer.train()

    # ===== Plot do loss (apenas no processo 0) =====
# ===== Plot do loss (apenas no processo 0) =====
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

        # === converter steps → nº de sentenças processadas ===
        world_size = torch.cuda.device_count() if torch.cuda.is_available() else 1
        eff_batch = args.batch_size * args.grad_accum * world_size  # blocos por update
        sentences_per_step = eff_batch * args.block_size            # sentenças por update
        train_sentences = [s * sentences_per_step for s in train_steps]
        eval_sentences = [s * sentences_per_step for s in eval_steps]

        # === plot ===
        plt.figure(figsize=(7, 4))
        if train_losses:
            plt.plot(train_sentences, train_losses, label="train_loss")
        if eval_losses:
            plt.plot(eval_sentences, eval_losses, label="eval_loss", linestyle="--")
        plt.xlabel("Sentenças processadas")
        plt.ylabel("loss")
        plt.title("Training (and Eval) Loss")
        plt.grid(True, alpha=0.3)
        plt.legend()
        out_plot = Path(args.out_dir) / "loss_curve.png"
        plt.tight_layout()
        plt.savefig(out_plot, dpi=150)
        print(f"[OK] Gráfico de loss salvo em: {out_plot}")


    # 6) Salvar (apenas no processo 0)
    if getattr(trainer, "is_world_process_zero", lambda: True)():
        save_dir = out / "final"
        save_dir.mkdir(parents=True, exist_ok=True)
        trainer.save_model(str(save_dir))
        tokenizer.save_pretrained(str(save_dir))
        print(f"[OK] Modelo e tokenizer salvos em: {save_dir}")

    # Encerrar DDP limpo (se aplicável)
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()

# Exemplo de execução:
# python qwen.py --input_file part_3.log --out_dir part3_qwen --vocab_size 11000 --block_size 1024 --epochs 20 --batch_size 8
