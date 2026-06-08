from __future__ import annotations

import argparse
import sys

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from .data import clean_text
from .utils import configure_logging


def generate_summary(args: argparse.Namespace) -> str:
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, use_fast=True)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model_path)
    device = "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    model.to(device)
    model.eval()

    if args.file:
        with open(args.file, "r", encoding="utf-8") as f:
            text = f.read()
    elif args.text:
        text = args.text
    else:
        text = sys.stdin.read()

    text = clean_text(args.prefix + text)
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=args.max_source_length,
    ).to(device)
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_length=args.max_length,
            min_length=args.min_length,
            num_beams=args.num_beams,
            no_repeat_ngram_size=args.no_repeat_ngram_size,
            repetition_penalty=args.repetition_penalty,
            length_penalty=args.length_penalty,
            early_stopping=True,
        )
    return tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate one Vietnamese summary.")
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--text")
    parser.add_argument("--file")
    parser.add_argument("--prefix", default="summarize: ")
    parser.add_argument("--max_source_length", type=int, default=768)
    parser.add_argument("--max_length", type=int, default=160)
    parser.add_argument("--min_length", type=int, default=50)
    parser.add_argument("--num_beams", type=int, default=4)
    parser.add_argument("--no_repeat_ngram_size", type=int, default=3)
    parser.add_argument("--repetition_penalty", type=float, default=1.05)
    parser.add_argument("--length_penalty", type=float, default=1.0)
    parser.add_argument("--cpu", action="store_true")
    return parser.parse_args()


def main() -> None:
    configure_logging()
    print(generate_summary(parse_args()))


if __name__ == "__main__":
    main()

