# TFT-GPT: Token-Factored Transformer with ALiBi

This repository introduces a representational refactoring of the standard transformer architecture. It factors internal representations into separate **token-like (`xt`)** and **embedding-like (`xe`)** streams. The goal is to provide traceable symbolic manipulation through attention mechanisms and inspectable context inference via feed-forward networks. Positional information is encoded using **ALiBi** to maintain the interpretability of these distinct streams.

## Core Idea: Factored Representation

Traditional transformers use a single hidden state vector. This project separates that state:

- **`xt` (Token-like stream):** Updated by attention, handles symbolic manipulation and token relationships.
- **`xe` (Embedding-like stream):** Updated by MLPs, captures contextual and semantic information.

This factorization aims for "dimensional integrity," where each component of the hidden state has a clear informational role, enhancing interpretability.

The update mechanism is conceptually:

```python
xt = xt + attention(xt + xe)  # Symbolic manipulation
xe = xe + mlp(xt + xe)        # Contextual information
```

## Role of ALiBi

ALiBi (Attention with Linear Biases) is crucial for preserving the representational structure of the `xt` and `xe` streams. Instead of adding positional embeddings directly to the states (which would "contaminate" their distinct natures), ALiBi encodes position as an operator (an attention bias). This keeps positional information orthogonal to the content representations.

## Overarching Repository Structure

The repository is organized into several key directories:

- `README.md`: This file, providing an overview of the project.
- `config_alibi.py`: Handles configuration for the ALiBi model.
- `requirements.txt`: Lists project dependencies.
- `setup.py`: Script for packaging and distributing the project.

Directories:

- `examples/`: Example scripts for training and analysis.
  - `main_tft_gpt.py`: Simple training.
  - `train_factored_alibi_example.py`: More advanced setup.
- `inference/`: Modules for text generation and sampling strategies.
- `model/`: Core model definition.
  - `model_token_factored_alibi.py`: Implements the Token-Factored Transformer with ALiBi.
- `mytokenizers/`: Tokenizer implementations (base and GPT-2 wrapper).
- `outputs/`: Likely used for saving outputs, logs, and results.
- `scripts/`: Shell scripts for common tasks.
- `trainers/`: Training loop implementations.
- `utils/`: Utilities such as data loading (`data_utils.py`).

## Citation

If you use this work, please refer to the associated paper:

```bibtex
@article{kerce2025tokenfactored,
  title={Token-Factored Transformers: Architectural Separation of Symbolic and Contextual Reasoning for Mechanistic Interpretability},
  author={Kerce, Clayton and Fox, Alexis},
  year={2025}
}
```

## License

MIT
