#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Minimal inference script using FactoredTransformerModelALiBi.load_from_checkpoint.
"""

import argparse
import torch
from transformers import AutoTokenizer
from model.model_token_factored_alibi import FactoredTransformerModelALiBi

def parse_args():
    parser = argparse.ArgumentParser(description="Run inference with FactoredTransformerModelALiBi.")
    parser.add_argument('--model_path', type=str, required=True, help='Path to model checkpoint (.pt)')
    parser.add_argument('--prompt', type=str, required=True, help='Prompt text for generation')
    parser.add_argument('--device', type=str, default='cuda', choices=['cpu', 'cuda'], help='Device to use')
    parser.add_argument('--max_new_tokens', type=int, default=50, help='Number of tokens to generate')
    parser.add_argument('--temperature', type=float, default=0.8, help='Sampling temperature')
    parser.add_argument('--top_k', type=int, default=40, help='Top-k filtering')
    return parser.parse_args()

def main():
    args = parse_args()
    device = args.device

    # Load model and tokenizer from checkpoint
    model, tokenizer = FactoredTransformerModelALiBi.load_from_checkpoint(args.model_path, device)
    if model is None or tokenizer is None:
        print("Failed to load model or tokenizer.")
        return

    model.eval()
    tokenizer.pad_token = tokenizer.eos_token
    input_ids = tokenizer.encode(args.prompt, return_tensors='pt').to(device)

    with torch.no_grad():
        output_ids = model.generate(
            input_ids,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_k=args.top_k
        )

    output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    print("\nPrompt:")
    print(args.prompt)
    print("\nGenerated:")
    print(output_text)

if __name__ == "__main__":
    main()
