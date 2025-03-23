# ABL: A Naive Implementation of Auto Labelling System
This is my graduation project v1 (metaphase).
## Overview

ABL combines neural networks and symbolic reasoning to create an interpretable text classification system. The model works by:

1. Encoding text using a pre-trained language model (BERT)
2. Detecting concepts present in the text
3. Applying symbolic reasoning over detected concepts to make predictions
4. Learning interpretable rules that connect concepts to labels

The key advantage of this approach is that the model's decisions can be explained in terms of the concepts it identifies in the text and the rules it learns to connect these concepts to labels.

## Project Structure

```
abl-text/
├── abl/                    # Core framework
│   ├── datasets/            # Dataset definitions and loading
│   ├── models/              # Model components
│   ├── nn/                  # Text perception modules
│   └── utils/               # Utility functions
├── scripts/                 # Training and evaluation scripts
└── requirements.txt         # Package dependencies
```

## Data Format

This framework expects data in the following format:

- **train.jsonl**, **val.jsonl**, **test.jsonl**: Each line contains a JSON object with text, label, and optional concept annotations
- **meta/labels.json**: Mapping of label names to indices
- **meta/concepts.json**: Mapping of concept names to indices

## Usage

### Training

To train a model(By myself simple test):

```bash
python scripts/text_reasoning_train.py \     
    --config configs/config.json \     
    --dataset-config configs/dataset_config.json \     
    --output-dir experiments/text_reasoning \     
    --epochs 50     --batch-size 32     --lr 1e-4     --device cuda
```
### Visualization training process
To show the training process:

```bash
tensorboard --logdir experiments/text_reasoning/tensorboard
```

### Evaluation

To evaluate a trained model:

```bash
python scripts/text_reasoning_test.py \     
    --config configs/config.json \     
    --dataset-config configs/dataset_config.json \     
    --checkpoint experiments/text_reasoning/checkpoints/model_best.pth \     
    --output-dir experiments/text_reasoning/results \     
    --batch-size 32     --device cuda     --visualize     --interpret-rules
```

## Key Components

- **Text Encoder**: Uses a pre-trained BERT model to encode text.
- **Concept Embedding**: Detects concepts in text and represents them as embeddings.
- **Quasi-Symbolic Reasoning**: Bridges neural perception with symbolic reasoning.
- **NSCLReasoning**: Applies rule-based reasoning to connect concepts to labels.
- **Program Executor**: Executes symbolic programs for interpretable predictions.
