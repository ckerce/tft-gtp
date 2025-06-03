#!/usr/bin/env python
# train_simple.py
"""
Simplified training script for Token-Factored Transformer with ALiBi.
This demonstrates the core TFT concept without unnecessary complexity.
"""

import argparse
import os
import sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import logging
from datetime import datetime
from datasets import load_dataset
from transformers import AutoTokenizer

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.config.config_alibi import GPTConfigALiBi, print_config_alibi, load_alibi_config_preset
from model.model_token_factored_alibi import FactoredTransformerModelALiBi

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def simple_collate_fn(batch, tokenizer, max_length=128):
    """Simple collate function for language modeling."""
    texts = [item['text'] if 'text' in item else str(item) for item in batch]
    
    # Tokenize all texts
    encoded = tokenizer(
        texts,
        truncation=True,
        padding=True,
        max_length=max_length,
        return_tensors='pt'
    )
    
    # For causal LM, labels are the same as input_ids but shifted
    input_ids = encoded['input_ids']
    labels = input_ids.clone()
    
    return {
        'input_ids': input_ids,
        'attention_mask': encoded['attention_mask'],
        'labels': labels
    }

def load_simple_dataset(dataset_name, dataset_config, tokenizer, max_samples=1000, max_length=128):
    """Load and prepare a simple dataset - supports TinyStories, Wikipedia, and code datasets."""
    logger.info(f"Loading dataset: {dataset_name}")
    if dataset_config:
        logger.info(f"Dataset config: {dataset_config}")
    
    # Load dataset based on name
    try:
        split_str = f"train[:{max_samples}]"
        
        if dataset_name == "tinystories":
            dataset = load_dataset("roneneldan/TinyStories", split=split_str)
        elif dataset_name == "wikipedia":
            config = dataset_config or "20231101.en"
            dataset = load_dataset("wikimedia/wikipedia", config, split=split_str)
        elif dataset_name == "code":
            config = dataset_config or "python" 
            dataset = load_dataset("code_search_net", config, split=split_str)
        else:
            # Try direct loading with the provided name
            if dataset_config:
                dataset = load_dataset(dataset_name, dataset_config, split=split_str, trust_remote_code=True)
            else:
                dataset = load_dataset(dataset_name, split=split_str, trust_remote_code=True)
    except Exception as e:
        logger.error(f"Failed to load dataset {dataset_name}: {e}")
        raise ValueError(f"Unknown or unavailable dataset: {dataset_name}")
    
    # Dataset-specific filtering
    def is_valid_sample(x):
        # Get text field based on dataset
        text = None
        if 'text' in x:
            text = x['text']
        elif 'func_code_string' in x:  # CodeSearchNet
            text = x['func_code_string']
        elif 'code' in x:
            text = x['code']
        
        if not text or not isinstance(text, str):
            return False
            
        # Basic length filter
        if len(text.strip()) < 20:
            return False
            
        # Dataset-specific quality filters
        if dataset_name == "tinystories":
            return any(word in text.lower() for word in ['once', 'there', 'was', 'and'])
        elif dataset_name == "wikipedia":
            return len(text.strip()) > 100 and not text.startswith('#REDIRECT')
        elif dataset_name == "code":
            return any(char in text for char in ['{', '(', 'def ', 'function', 'class'])
            
        return True
    
    # Filter dataset
    original_size = len(dataset)
    dataset = dataset.filter(is_valid_sample)
    logger.info(f"Filtered dataset: {original_size} -> {len(dataset)} samples")
    
    # Create DataLoader
    def collate_wrapper(batch):
        return simple_collate_fn(batch, tokenizer, max_length)
    
    dataloader = DataLoader(
        dataset,
        batch_size=8,  # Small batch size for demo
        shuffle=True,
        collate_fn=collate_wrapper
    )
    
    logger.info(f"Dataset loaded: {len(dataset)} samples")
    return dataloader

def train_model(model, dataloader, optimizer, device, num_epochs=2):
    """Simple training loop."""
    model.train()
    model.to(device)
    
    for epoch in range(num_epochs):
        total_loss = 0
        num_batches = 0
        
        logger.info(f"Starting epoch {epoch + 1}/{num_epochs}")
        
        for batch_idx, batch in enumerate(dataloader):
            # Move to device
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            # Forward pass
            optimizer.zero_grad()
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs['loss']
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            
            if batch_idx % 10 == 0:
                logger.info(f"Epoch {epoch+1}, Batch {batch_idx}, Loss: {loss.item():.4f}")
        
        avg_loss = total_loss / num_batches
        logger.info(f"Epoch {epoch+1} completed. Average Loss: {avg_loss:.4f}")

def generate_text(model, tokenizer, prompt, max_new_tokens=50, device='cpu'):
    """Simple text generation."""
    model.eval()
    
    # Encode prompt
    input_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)
    
    with torch.no_grad():
        generated = model.generate(
            input_ids, 
            max_new_tokens=max_new_tokens,
            temperature=0.8,
            top_k=40
        )
    
    # Decode generated text
    generated_text = tokenizer.decode(generated[0], skip_special_tokens=True)
    return generated_text

def main():
    parser = argparse.ArgumentParser(description='Simple TFT Training')
    parser.add_argument('--dataset', choices=['tinystories', 'wikipedia', 'code'], default='tinystories')
    parser.add_argument('--dataset_config', type=str, default=None, 
                       help='Dataset config: for wikipedia use language like "20231101.en", for code use language like "python"')
    parser.add_argument('--max_samples', type=int, default=1000)
    parser.add_argument('--num_epochs', type=int, default=2)
    parser.add_argument('--device', default='auto')
    parser.add_argument('--preset', choices=['tiny', 'small'], default='tiny')
    args = parser.parse_args()
    
    # Setup device
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    
    logger.info(f"Using device: {device}")
    
    # Load configuration
    config = load_alibi_config_preset(args.preset)
    config.use_v = True  # Enable value factorization
    config.use_proj = True  # Enable projection
    
    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained('gpt2')
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Update config with tokenizer info
    config.update_from_tokenizer(tokenizer)
    
    # Print configuration
    print_config_alibi(config, dataset_name=args.dataset, dataset_config=args.dataset_config, max_samples=args.max_samples)
    
    # Load data
    dataloader = load_simple_dataset(args.dataset, args.dataset_config, tokenizer, args.max_samples, config.block_size)
    
    # Initialize model
    model = FactoredTransformerModelALiBi(config)
    logger.info(f"Model initialized with {model.get_num_params()/1e6:.2f}M parameters")
    
    # Setup optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    
    # Train model
    train_model(model, dataloader, optimizer, device, args.num_epochs)
    
    # Test generation with dataset-appropriate prompts
    if args.dataset == "tinystories":
        test_prompts = [
            "Once upon a time",
            "There was a little cat",
            "In a big forest"
        ]
    elif args.dataset == "wikipedia":
        test_prompts = [
            "The history of",
            "Climate change is",
            "Artificial intelligence"
        ]
    elif args.dataset == "code":
        test_prompts = [
            "def fibonacci(",
            "class DataProcessor:",
            "import numpy as np"
        ]
    
    logger.info("Testing text generation:")
    for prompt in test_prompts:
        generated = generate_text(model, tokenizer, prompt, device=device)
        print(f"\nPrompt: {prompt}")
        print(f"Generated: {generated}")
    
    # Save model
    output_path = f"tft_model_{args.preset}_{args.dataset}.pt"
    if args.dataset_config:
        output_path = f"tft_model_{args.preset}_{args.dataset}_{args.dataset_config}.pt"
    
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': config,
        'tokenizer': tokenizer
    }, output_path)
    logger.info(f"Model saved to {output_path}")

if __name__ == "__main__":
    main()
