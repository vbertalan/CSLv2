#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse, os, random, tempfile, statistics as stats
from pathlib import Path

# Opcional: só importe sentencepiece se quiser estimar subwords
try:
    import sentencepiece as spm
    HAS_SPM = True
except Exception:
    HAS_SPM = False


def read_lines(path, max_lines=None):
    lines = []
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for i, line in enumerate(f):
            if max_lines and i >= max_lines:
                break
            s = line.strip()
            if s:
                lines.append(s)
    return lines


def basic_stats(lines):
    n = len(lines)
    uniq = len(set(lines))
    char_lens = [len(s) for s in lines]
    word_lens = [len(s.split()) for s in lines]
    return {
        "n_lines": n,
        "unique_lines": uniq,
        "avg_chars_per_line": sum(char_lens)/n if n else 0.0,
        "p50_chars": int(stats.median(char_lens)) if n else 0,
        "p95_chars": int(stats.quantiles(char_lens, n=100)[94]) if n >= 100 else (max(char_lens) if n else 0),
        "avg_words_per_line": sum(word_lens)/n if n else 0.0,
        "p50_words": int(stats.median(word_lens)) if n else 0,
        "p95_words": int(stats.quantiles(word_lens, n=100)[94]) if n >= 100 else (max(word_lens) if n else 0),
    }


def train_temp_sp(corpus_lines, vocab_size, model_type="unigram"):
    # Cria arquivo temporário com amostra do corpus
    with tempfile.TemporaryDirectory() as td:
        input_path = Path(td) / "sp_corpus.txt"
        model_prefix = str(Path(td) / "sp_model")
        with open(input_path, "w", encoding="utf-8") as w:
            w.write("\n".join(corpus_lines))
        spm.SentencePieceTrainer.Train(
            input=str(input_path),
            model_prefix=model_prefix,
            vocab_size=int(vocab_size),
            character_coverage=1.0,
            model_type=model_type,
            input_sentence_size=0,
            shuffle_input_sentence=True,
            unk_id=0, bos_id=1, eos_id=2, pad_id=3,
            hard_vocab_limit=False,
        )
        sp = spm.SentencePieceProcessor()
        sp.load(model_prefix + ".model")
        return sp


def estimate_subtokens_per_line(sp, lines, sample=20000):
    # Estima #subtokens por linha em uma amostra
    sample_lines = lines if len(lines) <= sample else random.sample(lines, sample)
    lens = [len(sp.encode(s)) for s in sample_lines]
    avg = sum(lens)/len(lens) if lens else 0.0
    p95 = int(stats.quantiles(lens, n=100)[94]) if len(lens) >= 100 else (max(lens) if lens else 0)
    return avg, p95


def suggest_vocab_size(unique_words_estimate, unique_lines, default_candidates):
    # Heurística simples: quanto mais restrito o domínio, menor vocab
    # Use metade de unique_words como base, mas clip nos candidatos
    base = max(2000, int(0.5 * unique_words_estimate))
    # ajuste pelas linhas únicas: se poucas linhas únicas, mantenha mais baixo
    if unique_lines < 100_000:
        base = min(base, 8000)
    # escolha candidato mais próximo para baixo
    candidates = sorted(default_candidates)
    choice = candidates[0]
    for c in candidates:
        if c <= base:
            choice = c
    return choice


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", required=True, help="Arquivo com 1 frase por linha")
    ap.add_argument("--sample_for_vocab", type=int, default=500_000,
                    help="Amostra de linhas para estimar vocabulário/treinar SP temp")
    ap.add_argument("--sp_candidates", default="8000,16000,32000",
                    help="Candidatos de vocab_size para estimar (coma-separado)")
    ap.add_argument("--target_sentences", type=int, default=50,
                    help="Contexto desejado em nº de frases para projetar BLOCK_SIZE")
    ap.add_argument("--model_hidden_size", type=int, default=768,
                    help="Para estimar custo de embeddings (vocab_size * hidden_size)")
    ap.add_argument("--no_spm", action="store_true", help="Não usar SentencePiece temporário (só heurísticas)")
    args = ap.parse_args()

    random.seed(42)

    lines = read_lines(args.file)
    if not lines:
        print("Arquivo vazio.")
        return

    bs = basic_stats(lines)
    print("=== Estatísticas básicas ===")
    for k, v in bs.items():
        print(f"{k}: {v}")

    # Estima unique words por split simples (rápido)
    uniq_words = len(set(w for s in lines for w in s.split()))
    print(f"unique_words_estimate (split em espaço): {uniq_words}")

    candidates = [int(x) for x in args.sp_candidates.split(",") if x.strip().isdigit()]

    # Sugestão inicial de vocab (sem SP)
    rough_vocab = suggest_vocab_size(uniq_words, bs["unique_lines"], candidates)
    print(f"\nSugestão inicial de VOCAB_SIZE (heurística): {rough_vocab}")

    # Se solicitado, treine SP temporário para refinar estimativas
    sp_results = {}
    if not args.no_spm:
        if not HAS_SPM:
            print("\n[AVISO] sentencepiece não instalado; pulei a estimativa de subtokens.")
        else:
            sample_lines = lines if len(lines) <= args.sample_for_vocab else random.sample(lines, args.sample_for_vocab)
            print(f"\nTreinando SP temporário em {len(sample_lines)} linhas (candidatos: {candidates})...")
            for v in candidates:
                sp = train_temp_sp(sample_lines, vocab_size=v, model_type="unigram")
                avg, p95 = estimate_subtokens_per_line(sp, sample_lines)
                emb_params = v * args.model_hidden_size
                emb_mem_mb = emb_params * 2 / (1024**2)  # ~fp16/bf16 2 bytes/param
                sp_results[v] = {"avg_subtokens_per_line": avg, "p95_subtokens_per_line": p95,
                                 "embedding_params": emb_params, "embedding_mem_mb_fp16": emb_mem_mb}
                print(f"- vocab={v}: avg_subtokens/linha≈{avg:.2f}, p95≈{p95}, "
                      f"emb_params={emb_params/1e6:.1f}M (~{emb_mem_mb:.0f} MB fp16)")

    # Projetar BLOCK_SIZE
    # Se temos SP, escolha vocab que produza avg_subtokens/linha razoável (ex.: <= 12)
    chosen_vocab = rough_vocab
    avg_tokens_per_sentence = max(1.0, bs["avg_words_per_line"])  # fallback
    if sp_results:
        # escolhe o menor vocab cuja média de subtokens/linha <= 12 (ajuste se quiser)
        target_avg_limit = 12
        for v in sorted(sp_results):
            if sp_results[v]["avg_subtokens_per_line"] <= target_avg_limit:
                chosen_vocab = v
                avg_tokens_per_sentence = sp_results[v]["avg_subtokens_per_line"]
                break
        else:
            # senão, pegue o maior candidato e use sua média
            vmax = max(sp_results)
            chosen_vocab = vmax
            avg_tokens_per_sentence = sp_results[vmax]["avg_subtokens_per_line"]
    else:
        # heurística: subtokens ~ palavras (aproximação)
        pass

    # BLOCK_SIZE = target_sentences * avg_tokens_per_sentence * 1.1 (folga)
    block_size = int(args.target_sentences * avg_tokens_per_sentence * 1.1)
    # Arredonde para múltiplos típicos
    for cand in [256, 512, 768, 1024, 1536, 2048, 3072, 4096]:
        if block_size <= cand:
            block_size = cand
            break

    print("\n=== Sugestões ===")
    print(f"VOCAB_SIZE sugerido: {chosen_vocab}")
    print(f"BLOCK_SIZE sugerido (para ~{args.target_sentences} frases de contexto): {block_size}")

    if sp_results and chosen_vocab in sp_results:
        r = sp_results[chosen_vocab]
        print(f"(com vocab={chosen_vocab}: avg_subtokens/linha≈{r['avg_subtokens_per_line']:.2f}, "
              f"p95≈{r['p95_subtokens_per_line']})")
        print(f"Embedding params≈{r['embedding_params']/1e6:.1f}M  "
              f"(~{r['embedding_mem_mb_fp16']:.0f} MB fp16)")

    print("\nDicas:")
    print("- Se estourar VRAM, reduza BLOCK_SIZE um nível (ex.: 1024 → 512) ou aumente grad_accum.")
    print("- Se as frases ficarem muito fragmentadas, aumente VOCAB_SIZE (8k → 16k).")
    print("- Se o domínio for super restrito (muitas repetições), VOCAB_SIZE menor tende a funcionar bem.")


if __name__ == "__main__":
    main()
'''
python estimate-hps.py \
  --file /home/vbertalan/Downloads/Projetos/CSL/CSLv2/logs/part_3.log \
  --sp_candidates 8000,16000,32000 \
  --target_sentences 50 \
  --model_hidden_size 768
'''