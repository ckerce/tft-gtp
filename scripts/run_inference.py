#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Minimal inference script using FactoredTransformerModelALiBi.load_from_checkpoint.
Supports interactive and one-shot prompt modes.
"""

import argparse
import torch
from transformers import AutoTokenizer
from model.model_token_factored_alibi import FactoredTransformerModelALiBi

def parse_args():
    parser = argparse.ArgumentParser(description="Run inference with FactoredTransformerModelALiBi.")
    parser.add_argument('--model_path', type=str, required=True, help='Path to model checkpoint (.pt)')
    parser.add_argument('--prompt', type=str, help='Text prompt for generation (optional in interactive mode)')
    parser.add_argument('--interactive', action='store_true', help='Enter interactive mode for multiple prompts')
    parser.add_argument('--device', type=str, default='cuda', choices=['cpu', 'cuda'], help='Device to use')
    parser.add_argument('--max_new_tokens', type=int, default=50, help='Number of tokens to generate')
    parser.add_argument('--temperature', type=float, default=0.8, help='Sampling temperature')
    parser.add_argument('--top_k', type=int, default=40, help='Top-k filtering')
    return parser.parse_args()

def generate_response(prompt, model, tokenizer, max_new_tokens, temperature, top_k):
    device = next(model.parameters()).device
    input_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)

    with torch.no_grad():
        output_ids = model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k
        )

    return tokenizer.decode(output_ids[0], skip_special_tokens=True)

def main():
    args = parse_args()

    model, tokenizer = FactoredTransformerModelALiBi.load_from_checkpoint(args.model_path, args.device)
    if model is None or tokenizer is None:
        print("Failed to load model or tokenizer.")
        return
    model.tokenizer = tokenizer

    model.eval()
    tokenizer.pad_token = tokenizer.eos_token

    if args.interactive:
        print("\n=== Interactive Mode ===")
        print("Type your prompt and press Enter. Type 'exit' or 'quit' to leave.")
        while True:
            try:
                prompt = input("\nPrompt > ")
                if prompt.strip().lower() in ("exit", "quit"):
                    print("Exiting interactive mode.")
                    break
                if not prompt.strip():
                    print("Please enter a non-empty prompt.")
                    continue

                output = generate_response(
                    prompt, model, tokenizer,
                    max_new_tokens=args.max_new_tokens,
                    temperature=args.temperature,
                    top_k=args.top_k
                )
                print("\nGenerated:")
                print(output)

            except KeyboardInterrupt:
                print("\nInterrupted. Exiting.")
                break
    else:
        if not args.prompt:
            print("Error: --prompt is required when not in interactive mode.")
            return

        output = generate_response(
            args.prompt, model, tokenizer,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_k=args.top_k
        )
        print("\nPrompt:")
        print(args.prompt)
        print("\nGenerated:")
        print(output)

if __name__ == "__main__":
    main()
