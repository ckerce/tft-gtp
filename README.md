# TFT-GPT: Token-Factored Transformer with ALiBi

This repository introduces a representational refactoring of the standard transformer architecture, factoring internal representations into separate token-like and embedding-like streams. The purpose of this representation is to provide tracable symbolic manipulation from the attention mechanisms and inspectable abductive inference of context through the feed-forward network. Token positional information is encoded through ALiBi operators; this is done to maintain strict interpretability of the token-like embeddings (dentoted `xt`) and the non-token-like embeddings (`xe`).  


## Core Concept

Traditional transformers use a single hidden state vector `x` to represent all internal information. This implementation separates that state into two complementary streams:

- **`xt` (token-like states)**: Updated by attention mechanisms, handling symbolic manipulation and token-to-token relationships
- **`xe` (embedding-like states)**: Updated by MLPs, capturing contextual and semantic information

The key insight is maintaining the **dimensional integrity** of these separated representations throughout processing. These representations retain token-like interpretability throughout the inference process.  Manipulations and token associations can be tracked successively through the tft-gpt transformer blocks.

## Why ALiBi Positional Encoding

ALiBi is essential to this architecture -- not for parameter efficiency, but for **preserving represenational structure**. This is similar philosophically similar to the role that dimensional anlysis plays in physics; we only add like quantities. Traditional positional embeddings would contaminate the factored streams:

- Adding positional embeddings to `xt` corrupts token-like states with non-token information
- Adding them to `xe` pollutes embedding-like states with positional artifacts  
- Adding them to the combined state destroys the dimensional separation entirely

ALiBi solves this by encoding position as an **operator** (attention bias) rather than an **activation state**, keeping positional information orthogonal to the content representations in both streams.

## Dimensional Analysis Perspective

This factored representation provides a "dimensional analysis" accounting of transformer internals - each component of the hidden state has a clear informational role. The separation allows us to study how different types of processing (symbolic vs. contextual) contribute to the model's behavior, potentially improving interpretability and architectural understanding.

## Core Innovation

### Token-Factored Architecture

Traditional transformers maintain a single internal state vector `x` that gets updated by both attention and MLP layers:
```python
x = x + attention(x) + mlp(x)
```

TFT-GPT separates this into two specialized streams:
```python
xt = xt + attention(xt + xe)  # Token-like states (symbolic manipulation)
xe = xe + mlp(xt + xe)        # Embedding-like states (contextual information)
```

This factorization provides:
- An analog of **Dimensional analysis** from physics applied to transformer representations
- Inspectable **Symbolic reasoning** via attention operations on token-like states
- Traceable **Context injection** via MLP updates to embedding-like states  
- **Information structure preservation** by avoiding cross-contamination between these two mechanisms as informtion flows from one transformer block to the next

### ALiBi Positional Encoding

Mainly serves to move positional information from the internal state (`x`, `xt`, or `xe`) to the operators on internal states.

As in the original ALiBi, it replaces learned positional embeddings with linear biases in attention scores, so enabling:
- **Parameter efficiency** (no positional embedding table)
- **Length extrapolation** beyond training sequence lengths
- **Preserved factorization** (no positional modulation of internal states)

## Quick Start

### Installation
```bash
pip install torch transformers datasets tqdm
```

### Simple Training
```bash
# Train on TinyStories
python train_simple.py --dataset tinystories --max_samples 1000 --preset tiny

# Train on Wikipedia
python train_simple.py --dataset wikipedia --dataset_config "20231101.en" --max_samples 500

# Train on Python code
python train_simple.py --dataset code --dataset_config "python" --max_samples 800
```

### Advanced Training with Curriculum Learning
```bash
python examples/train_factored_alibi_example.py \
  --dataset "roneneldan/TinyStories" \
  --preset medium \
  --block_size 128 \
  --max_position_embeddings 256 \
  --use_v --use_proj \
  --num_epochs 10
```

## Repository Structure

```
tft-gpt/
├── train_simple.py                     # Simple training script
├── config_alibi.py                     # ALiBi configuration system
├── model/
│   ├── __init__.py                     # Model registry
│   └── model_token_factored_alibi.py   # Core TFT implementation
├── utils/
│   └── data_utils.py                   # Simplified data utilities
├── examples/
│   └── train_factored_alibi_example.py # Advanced training with curriculum
├── dataloader/                         # Curriculum learning system
│   ├── __init__.py
│   ├── base.py                         # Base curriculum classes
│   ├── strategies.py                   # Learning strategies
│   └── factory.py                      # Dataloader factory
├── mytokenizers/                       # Tokenization system
│   ├── __init__.py
│   ├── base_tokenizer.py               # Base tokenizer interface
│   └── gpt2_tokenizer.py               # GPT-2 tokenizer wrapper
├── trainers/                           # Training loop implementations
│   ├── __init__.py
│   ├── base_trainer.py                 # Base trainer with callbacks
│   └── simple_trainer.py               # Simple training implementation
└── inference/
    └── generation.py                   # Text generation utilities
```

## Key Features

### Model Architecture
- **Factored attention**: Separate Q,K computation from V processing
- **Configurable factorization**: `use_v` and `use_proj` flags for experimentation
- **ALiBi slopes**: Automatic computation of attention biases
- **Length extrapolation**: Train on 128 tokens, generate 512+ tokens

### Supported Datasets
- **TinyStories**: Children's stories for narrative learning
- **Wikipedia**: Encyclopedia articles for factual knowledge  
- **Code datasets**: Programming languages for structured reasoning

### Training Systems
- **Simple trainer**: Basic training loop for experimentation
- **Curriculum learning**: Advanced multi-dataset blending strategies
- **Callback system**: Extensible training hooks for monitoring

## Configuration Presets

Choose model size with `--preset`:

| Preset | Layers | Heads | Embedding | Parameters | Use Case |
|--------|--------|-------|-----------|------------|----------|
| `tiny` | 2 | 2 | 128 | ~0.5M | Fast experimentation |
| `small` | 6 | 6 | 192 | ~2M | Development |
| `medium` | 6 | 6 | 768 | ~12M | Research |
| `large` | 12 | 12 | 768 | ~24M | Production |

## Understanding the Theory

### Dimensional Analysis Perspective
The factorization provides a "dimensional analysis" of transformer states:
- `xt`: Token-like quantities (symbolic, discrete)
- `xe`: Embedding-like quantities (continuous, contextual)

This separation mirrors physics where different types of quantities (length, mass, time) are kept distinct to maintain mathematical consistency.

### Symbolic vs. Contextual Processing
- **Attention on `xt`**: Performs symbolic manipulation without contamination from continuous embeddings
- **MLP on `xe`**: Injects contextual information without disrupting symbolic structure
- **Combined input**: Both streams inform each operation while maintaining their distinct natures

### ALiBi Benefits
1. **No learned positions**: Saves parameters and enables extrapolation
2. **Linear attention bias**: Simple, interpretable positional encoding
3. **Factorization preservation**: Positions don't modulate internal representations

## Experimentation Guide

### Core Factorization Settings
```python
config.use_v = True/False      # Enable separate value projection
config.use_proj = True/False   # Enable output projection
```

### Length Extrapolation Testing
```python
config.block_size = 128                    # Training length
config.max_position_embeddings = 512      # Inference length (4x extrapolation)
```

### Comparing Configurations
- **No factorization**: `use_v=False, use_proj=False` (baseline)
- **Value factorization**: `use_v=True, use_proj=False`
- **Full factorization**: `use_v=True, use_proj=True` (recommended)

## Advanced Features

### Curriculum Learning
Train on multiple datasets with evolving mixture weights:
```python
# Linear transition from code-heavy to text-heavy
--curriculum_datasets "code_search_net" "roneneldan/TinyStories"
--curriculum_start_weights 0.8 0.2
--curriculum_end_weights 0.3 0.7
```

### Custom Tokenizers
- **GPT-2 tokenizer**: Standard BPE tokenization
- **Character tokenizer**: Character-level for analysis
- **Extensible system**: Add custom tokenizers via factory pattern

### Generation Testing
Automatic length extrapolation testing after training shows ALiBi's ability to generate coherent text beyond training length.

## Research Applications

This architecture is particularly suited for studying:
- **Symbolic reasoning** in language models
- **Length generalization** capabilities
- **Representational structure** in transformers
- **Multi-modal extensions** (separate streams for different modalities)

## Citation

If you use this work, please cite:
```bibtex
@article{tft-gpt,
  title={Token-Factored Transformers with ALiBi: Block Invariant Representations in Neural Language Models},
  author={Clayton Kerce},
  year={2025},
  note={Implementation of factored transformer architecture with linear attention biases}
}
```

## Contributing

This repository demonstrates a novel architectural approach. Contributions welcome for:
- Additional datasets and tokenizers
- Comparative analysis with standard transformers  
- Extensions to other domains (vision, multimodal)
- Theoretical analysis of factorization benefits

## License

[Your chosen license]
