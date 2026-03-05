# PLM for Antibody Comprehension

This repository implements Protein Language Models (PLMs) for antibody sequence analysis, specifically focused on B-cell receptor (BCR) classification.

## Overview

Antibodies are versatile proteins with the capacity to bind a broad range of targets. This project explores deep representation learning models (like AntiBERTa, ESM-2, and BioBERT) to comprehend antibody sequences and predict their binding properties.

### Key Features

- **Multiple Model Backbones:** Support for AntiBERTa2, ESM-2, and BioBERT.
- **Flexible Classification Heads:** Choose between simple Linear (FC), MLP, or Transformer Encoder heads.
- **Advanced Loss Functions:** Includes LogitNorm, Focal Loss, and Expected Calibration Error (ECE) for better model calibration.
- **Data Balancing:** Built-in utilities to handle imbalanced antibody datasets.
- **Analysis Tools:** Attention attribution and UMAP visualization for model interpretability.

## Installation

```bash
pip install .
```

For development:
```bash
pip install -e ".[dev]"
```

## Usage

### Data Preparation

Place your BCR sequence data in CSV format in the `data/` directory. The CSV should contain columns for sequences (e.g., `SEQUENCE_AA`) and labels (e.g., `BINDING`).

### Training

You can use the provided training logic to train a model:

```python
from src.plm import PLMClassifier, build_balanced_dataset, get_dataloaders, train
from transformers import RoFormerTokenizer

# Configuration
config = {
    'model_type': 'antiberta',
    'classifier_type': 'mlp',
    'device': 'cuda',
    'learning_rate': 1e-5,
    'num_epochs': 50,
    'batch_size': 32,
    'save_path': 'checkpoints/best_model.ckpt',
    'log_interval': 10,
    'require_improvement': 1000,
}

# Data
tokenizer = RoFormerTokenizer.from_pretrained("alchemab/antiberta2")
balanced_data = build_balanced_dataset('data/your_data.csv')
train_loader, test_loader, label_map = get_dataloaders(balanced_data, tokenizer, max_length=128)

# Model
model = PLMClassifier(model_type=config['model_type'], classifier_type=config['classifier_type'])

# Run Training
train(model, train_loader, test_loader, config)
```

## Critical Review and Architecture

This repository has been refactored from research notebooks into a professional Python package.

### Architecture
- **Backbone Agnostic:** The `PLMClassifier` uses a unified interface for AntiBERTa, ESM-2, and BioBERT.
- **Modularity:** Models, data processing, and training logic are decoupled, allowing for easy experimentation with new heads or loss functions.
- **Calibration Focused:** Includes implementations for LogitNorm and ECE to address common overconfidence issues in deep protein models.
- **CLI-Ready:** A dedicated entry point (`plm-train`) simplifies the training pipeline for reproducible experiments.

### Improvements made during refactoring
- **Dynamic Configuration:** Hidden dimensions are now inferred directly from the transformer backbones.
- **Standardized Training:** Centralized the training loop with integrated validation, early stopping, and checkpointing.
- **Package Management:** Adopted `pyproject.toml` and standard `src/` layout for better maintainability and reproducibility.

## References
Liu, M., Zhang, Y., & Zhang, Y. (2025). Exploring protein language model architecture-induced biases for antibody comprehension. arXiv preprint. https://arxiv.org/abs/2512.09894v1