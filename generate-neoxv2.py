#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, StoppingCriteria, StoppingCriteriaList

class StopAfterNSentences(StoppingCriteria):
    def __init__(self, tokenizer, n_sentences=1, sent_token="<SENT_END>"):
        super().__init__()
        self.tokenizer = tokenizer
        self.n_sentences = n_sentences
        self.sent_id = tokenizer.convert_tokens_to_ids(sent_token)

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs):
        count = (input_ids == self.sent_id).sum().item()
        return count >= self.n_sentences

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", required=True, help="Diretório do modelo salvo (ex.: ./neox_sent/final)")
    ap.add_argument("--prompt", required=True, help="Frase inicial (sem <SENT_END>)")
    ap.add_argument("--num_sentences", type=int, default=1, help="Qtd. de frases/linhas novas para gerar")
    ap.add_argument("--max_new_tokens", type=int, default=256, help="Limite superior de tokens novos")
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top_p", type=float, default=0.9)
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model_dir, use_fast=True)
    mdl = AutoModelForCausalLM.from_pretrained(args.model_dir).eval()

    # 1) GARANTIR que o tokenizer não usa token_type_ids
    # (alguns tokenizers salvos podem vir com isso ativado)
    try:
        tok.model_input_names = ["input_ids", "attention_mask"]
    except Exception:
        pass

    # 2) Codificar SEM token_type_ids e SEM adicionar especiais
    enc = tok(
        args.prompt,
        return_tensors="pt",
        add_special_tokens=False
    )
    # Remover se por acaso veio:
    enc.pop("token_type_ids", None)

    input_ids = enc["input_ids"]
    attention_mask = enc.get("attention_mask", torch.ones_like(input_ids))

    # Detectar modo
    has_sent_end = ("<SENT_END>" in tok.get_vocab())
    stopping = None
    max_new_tokens = args.max_new_tokens

    if has_sent_end:
        # Modo SPM + <SENT_END>: parar após N frases (<SENT_END>)
        stopping = StoppingCriteriaList([StopAfterNSentences(tok, n_sentences=args.num_sentences)])
    else:
        # Modo 1-linha=1-token: pare após N tokens (cada token = 1 linha)
        max_new_tokens = min(args.num_sentences, args.max_new_tokens)

    with torch.no_grad():
        out = mdl.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            do_sample=True,
            top_p=args.top_p,
            temperature=args.temperature,
            max_new_tokens=max_new_tokens,
            pad_token_id=tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id,
            stopping_criteria=stopping
        )

    gen_ids = out[0][input_ids.shape[1]:]

    if has_sent_end:
        decoded = tok.decode(gen_ids, skip_special_tokens=False)
        pieces = [s.strip() for s in decoded.split("<SENT_END>") if s.strip()]
        print("\n".join(pieces[:args.num_sentences]))
    else:
        # cada token == uma linha (WordLevel)
        line_ids = gen_ids.tolist()[:args.num_sentences]
        lines = [tok.convert_ids_to_tokens(i) for i in line_ids]
        lines = [l.strip() for l in lines if l is not None and l.strip()]
        print("\n".join(lines))

if __name__ == "__main__":
    main()

# python generate-neoxv2.py   --model_dir ./part_3_neoxv2/final   --prompt "mv cannot stat no such file or directory"   --num_sentences 5 --max_new_tokens 100 --temperature 1 --top_p 0.9