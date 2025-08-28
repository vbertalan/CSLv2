#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import StoppingCriteria, StoppingCriteriaList
import torch


class StopAfterNSentences(StoppingCriteria):
    def __init__(self, tokenizer, n_sentences=1):
        super().__init__()
        self.tokenizer = tokenizer
        self.n_sentences = n_sentences
        self.sent_end_id = tokenizer.convert_tokens_to_ids("<SENT_END>")

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs):
        # conta quantos <SENT_END> existem na sequência inteira
        count = (input_ids == self.sent_end_id).sum().item()
        return count >= self.n_sentences


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", required=True, help="Diretório salvo (ex.: ./neox_sent/final)")
    ap.add_argument("--prompt", required=True, help="Frase inicial (sem <SENT_END>)")
    ap.add_argument("--num_sentences", type=int, default=1, help="Quantas frases gerar")
    ap.add_argument("--max_new_tokens", type=int, default=256)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top_p", type=float, default=0.9)
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model_dir, use_fast=True)
    mdl = AutoModelForCausalLM.from_pretrained(args.model_dir).eval()

    inp = tok(args.prompt, return_tensors="pt")
    stopper = StoppingCriteriaList([StopAfterNSentences(tok, n_sentences=args.num_sentences)])

    with torch.no_grad():
        out = mdl.generate(
            **inp,
            max_new_tokens=args.max_new_tokens,
            do_sample=True,
            top_p=args.top_p,
            temperature=args.temperature,
            pad_token_id=tok.pad_token_id,
            stopping_criteria=stopper
        )

    gen_full = tok.decode(out[0][inp["input_ids"].shape[1]:], skip_special_tokens=False)

    # divide por <SENT_END> e pega as N sentenças geradas
    pieces = [s.strip() for s in gen_full.split("<SENT_END>") if s.strip()]
    sentences = pieces[:args.num_sentences]
    print("\n".join(sentences))


if __name__ == "__main__":
    main()


# python generate-neox.py \
#   --model_dir ./Qwen3 \                                                 ## o diretório onde o modelo treinado foi salvo (pode ser Hugging Face ou seu fine-tune).
#   --prompt "tput no value for term and no t specified" \                ## → o texto inicial usado como prompt.
#   --num_sentences 2 \                                                   ## → vai gerar duas frases (cada uma delimitada por <SENT_END>).
#   --max_new_tokens 100 \                                                ## → no máximo 100 tokens novos gerados.
#   --temperature 0.7 \                                                   ## → controla a aleatoriedade (mais baixo = mais determinístico).
#   --top_p 0.9                                                           ## → usa nucleus sampling para limitar as probabilidades acumuladas.


#python generate-neox.py --model_dir ./Qwen3 --prompt "tput no value for term and no t specified" --num_sentences 2 --max_new_tokens 100 --temperature 0.7 --top_p 0.9     