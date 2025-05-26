# utils/data_utils.py
"""
Simple data utilities for TFT-GPT.
Provides a simplified dataloader alongside your existing complex system.
Handles TinyStories, Wikipedia, and code datasets.
"""

import torch
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import AutoTokenizer
import logging

logger = logging.getLogger(__name__)

def simple_collate_fn(batch, tokenizer, max_length=128):
    """Simple collate function for language modeling with dataset-aware text extraction."""
    texts = []
    for item in batch:
        if isinstance(item, dict):
            # Handle different dataset text fields
            text = None
            if 'text' in item:
                text = item['text']  # TinyStories, Wikipedia
            elif 'func_code_string' in item:
                text = item['func_code_string']  # CodeSearchNet
            elif 'code' in item:
                text = item['code']  # Other code datasets
            elif 'content' in item:
                text = item['content']  # Some Wikipedia formats
            else:
                # Fallback to first string field
                for key, value in item.items():
                    if isinstance(value, str) and len(value.strip()) > 10:
                        text = value
                        break
            
            if text:
                texts.append(text)
        else:
            texts.append(str(item))
    
    # Filter out empty/short texts
    texts = [t for t in texts if t and len(t.strip()) > 10]
    
    if not texts:
        # Return dummy batch if all texts are empty
        return {
            'input_ids': torch.tensor([[tokenizer.eos_token_id]]),
            'attention_mask': torch.tensor([[1]]),
            'labels': torch.tensor([[tokenizer.eos_token_id]])
        }
    
    # Tokenize
    encoded = tokenizer(
        texts,
        truncation=True,
        padding=True,
        max_length=max_length,
        return_tensors='pt'
    )
    
    # Labels are input_ids for causal language modeling
    return {
        'input_ids': encoded['input_ids'],
        'attention_mask': encoded['attention_mask'],
        'labels': encoded['input_ids'].clone()
    }

def load_and_prepare_data(dataset_name, dataset_config, tokenizer, max_samples, 
                         max_seq_length, batch_size, mlm=False, split='train', shuffle=True):
    """
    Simple data loading for TinyStories, Wikipedia, and code datasets.
    Maintains compatibility with your existing complex dataloader interface.
    
    Supported datasets:
    - roneneldan/TinyStories
    - wikimedia/wikipedia (with config like "20231101.en")  
    - code_search_net (with config like "python")
    
    Returns:
        tuple: (dataloader, tokenizer) - maintaining compatibility with existing code
    """
    logger.info(f"Loading dataset: {dataset_name}")
    if dataset_config:
        logger.info(f"Dataset config: {dataset_config}")
    
    # Handle different dataset loading patterns
    try:
        split_str = f"{split}[:{max_samples}]" if max_samples else split
        
        if dataset_config:
            dataset = load_dataset(dataset_name, dataset_config, split=split_str, trust_remote_code=True)
        else:
            dataset = load_dataset(dataset_name, split=split_str, trust_remote_code=True)
            
        logger.info(f"Raw dataset loaded: {len(dataset)} samples")
        
    except Exception as e:
        logger.error(f"Failed to load dataset {dataset_name}: {e}")
        # Try alternative loading for specific datasets
        if "tinystories" in dataset_name.lower():
            try:
                dataset = load_dataset("roneneldan/TinyStories", split=split_str)
                logger.info("Loaded TinyStories with fallback method")
            except Exception:
                raise e
        elif "wikipedia" in dataset_name.lower():
            try:
                config = dataset_config or "20231101.en"
                dataset = load_dataset("wikimedia/wikipedia", config, split=split_str)
                logger.info(f"Loaded Wikipedia ({config}) with fallback method")
            except Exception:
                raise e
        else:
            raise e
    
    # Dataset-specific filtering
    def is_valid_text(example):
        # Get text based on dataset type
        text = None
        
        if 'text' in example:
            text = example['text']
        elif 'func_code_string' in example:  # CodeSearchNet
            text = example['func_code_string']
        elif 'code' in example:
            text = example['code']
        elif 'content' in example:
            text = example['content']
        
        if not text or not isinstance(text, str):
            return False
            
        # Minimum length filter
        if len(text.strip()) < 20:
            return False
            
        # Dataset-specific filters
        if dataset_name == "roneneldan/TinyStories":
            # Stories should have some narrative structure
            return any(word in text.lower() for word in ['once', 'there', 'was', 'and', 'then'])
        elif "wikipedia" in dataset_name.lower():
            # Wikipedia articles should be substantial
            return len(text.strip()) > 100 and not text.startswith('#REDIRECT')
        elif "code" in dataset_name.lower():
            # Code should have basic structure
            return any(char in text for char in ['{', '(', 'def ', 'function', 'class'])
            
        return True
    
    original_size = len(dataset)
    dataset = dataset.filter(is_valid_text)
    filtered_size = len(dataset)
    logger.info(f"Filtered dataset: {original_size} -> {filtered_size} samples")
    
    if filtered_size == 0:
        raise ValueError(f"No valid samples found in dataset {dataset_name}")
    
    # Create collate function
    def collate_wrapper(batch):
        return simple_collate_fn(batch, tokenizer, max_seq_length)
    
    # Create DataLoader
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_wrapper,
        drop_last=True,  # Avoid issues with batch size variations
        num_workers=0  # Keep simple for compatibility
    )
    
    logger.info(f"Created DataLoader with {len(dataloader)} batches of size {batch_size}")
    return dataloader, tokenizer

# Additional utility for dataset info
def get_dataset_info(dataset_name):
    """Get information about supported datasets."""
    dataset_info = {
        "roneneldan/TinyStories": {
            "description": "Short children's stories dataset",
            "text_field": "text",
            "config": None,
            "typical_length": "100-500 tokens"
        },
        "wikimedia/wikipedia": {
            "description": "Wikipedia articles", 
            "text_field": "text",
            "config": "20231101.en (or other language/date)",
            "typical_length": "500-2000 tokens"
        },
        "code_search_net": {
            "description": "Code functions with documentation",
            "text_field": "func_code_string", 
            "config": "python (or other languages)",
            "typical_length": "100-1000 tokens"
        }
    }
    return dataset_info.get(dataset_name, {"description": "Unknown dataset"})
