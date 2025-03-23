#! /usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import torch
import numpy as np
import random
import collections


def set_random_seed(seed):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def ensure_dir(path):
    """Make sure the directory exists."""
    if not os.path.exists(path):
        os.makedirs(path)
    return path


def load_json(filename):
    """Load JSON file."""
    with open(filename, 'r') as f:
        return json.load(f)


def save_json(data, filename):
    """Save data to JSON file."""
    with open(filename, 'w') as f:
        json.dump(data, f, indent=2)


def compute_metrics(predictions, targets, average='macro'):
    """Compute evaluation metrics."""
    predictions = np.array(predictions)
    targets = np.array(targets)
    
    # Compute accuracy
    accuracy = np.mean(predictions == targets)
    
    # Compute precision, recall, F1 score
    unique_labels = np.unique(np.concatenate([predictions, targets]))
    precisions = []
    recalls = []
    f1_scores = []
    
    for label in unique_labels:
        true_positives = np.sum((predictions == label) & (targets == label))
        false_positives = np.sum((predictions == label) & (targets != label))
        false_negatives = np.sum((predictions != label) & (targets == label))
        
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
        
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        precisions.append(precision)
        recalls.append(recall)
        f1_scores.append(f1)
    
    if average == 'macro':
        precision = np.mean(precisions)
        recall = np.mean(recalls)
        f1_score = np.mean(f1_scores)
    else:  # micro average
        true_positives = 0
        false_positives = 0
        false_negatives = 0
        
        for label in unique_labels:
            true_positives += np.sum((predictions == label) & (targets == label))
            false_positives += np.sum((predictions == label) & (targets != label))
            false_negatives += np.sum((predictions != label) & (targets == label))
        
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
        
        f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1_score
    }


def extract_rules_from_model(reasoning_model, concept_names=None, threshold=0.1):
    """Extract interpretable rules from a reasoning model."""
    if not hasattr(reasoning_model, 'get_rules'):
        return None
    
    # Get rule weights and biases
    rule_weights, rule_biases = reasoning_model.get_rules()
    
    # Convert to numpy arrays
    if isinstance(rule_weights, torch.Tensor):
        rule_weights = rule_weights.detach().cpu().numpy()
    if isinstance(rule_biases, torch.Tensor) and rule_biases is not None:
        rule_biases = rule_biases.detach().cpu().numpy()
    
    # Create rules dictionary
    rules = {}
    
    for i in range(rule_weights.shape[0]):
        # Get rule weights
        weights = rule_weights[i]
        bias = rule_biases[i] if rule_biases is not None else 0.0
        
        # Find concepts with significant weights
        significant_indices = np.where(np.abs(weights) > threshold)[0]
        significant_weights = weights[significant_indices]
        
        # Skip rules with no significant concepts
        if len(significant_indices) == 0:
            continue
        
        # Create rule
        rule = []
        for idx, weight in zip(significant_indices, significant_weights):
            concept_name = concept_names[idx] if concept_names and idx < len(concept_names) else f'Concept-{idx}'
            rule.append({
                'concept': concept_name,
                'weight': float(weight)
            })
        
        # Sort concepts by absolute weight
        rule = sorted(rule, key=lambda x: abs(x['weight']), reverse=True)
        
        rules[f'Rule-{i}'] = {
            'concepts': rule,
            'bias': float(bias)
        }
    
    return rules

# Dataset
def validate_dataset_config(config, used_keys, config_name="dataset"):
    """Validate dataset configuration and add default values."""
    for key in used_keys:
        if key not in config:
            default = get_default_dataset_configs()[config_name][key]
            print(f'[WARNING] {config_name}.{key} not found; using default value: {default}')
            config[key] = default
    return config


def get_default_dataset_configs():
    """Get default configurations for datasets."""
    return {
        'dataset': {
            'name': 'text',
            'max_length': 128,
            'root_dir': 'data/text',
            'test_split': 'val',
            'concept_threshold': 0.5
        }
    }

def as_tensor(obj):
    """Convert object to tensor."""
    if isinstance(obj, torch.Tensor):
        return obj
    elif isinstance(obj, (list, tuple)):
        return torch.tensor(obj)
    elif isinstance(obj, collections.Mapping):
        return {k: as_tensor(v) for k, v in obj.items()}
    elif isinstance(obj, collections.Sequence):
        return [as_tensor(v) for v in obj]
    else:
        return obj


def as_float(obj):
    """Convert tensor or numpy array to float."""
    if isinstance(obj, torch.Tensor):
        return obj.item() if obj.numel() == 1 else obj.cpu().numpy().tolist()
    elif isinstance(obj, (list, tuple)):
        return [as_float(v) for v in obj]
    elif isinstance(obj, collections.Mapping):
        return {k: as_float(v) for k, v in obj.items()}
    elif isinstance(obj, collections.Sequence):
        return [as_float(v) for v in obj]
    else:
        return obj


def as_cpu(obj):
    """Move tensor to CPU."""
    if isinstance(obj, torch.Tensor):
        return obj.cpu()
    elif isinstance(obj, (list, tuple)):
        return [as_cpu(v) for v in obj]
    elif isinstance(obj, collections.Mapping):
        return {k: as_cpu(v) for k, v in obj.items()}
    elif isinstance(obj, collections.Sequence):
        return [as_cpu(v) for v in obj]
    else:
        return obj


def as_numpy(obj):
    """Convert tensor to numpy array."""
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu().numpy()
    elif isinstance(obj, (list, tuple)):
        return [as_numpy(v) for v in obj]
    elif isinstance(obj, collections.Mapping):
        return {k: as_numpy(v) for k, v in obj.items()}
    elif isinstance(obj, collections.Sequence):
        return [as_numpy(v) for v in obj]
    else:
        return obj


def as_device(obj, device):
    """Move tensor to specified device."""
    if isinstance(obj, torch.Tensor):
        return obj.to(device)
    elif isinstance(obj, (list, tuple)):
        return [as_device(v, device) for v in obj]
    elif isinstance(obj, collections.Mapping):
        return {k: as_device(v, device) for k, v in obj.items()}
    elif isinstance(obj, collections.Sequence):
        return [as_device(v, device) for v in obj]
    else:
        return obj
