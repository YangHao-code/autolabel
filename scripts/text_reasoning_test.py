#! /usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import os.path as osp
import sys
import json
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

from abl.datasets.factory import make_dataset_split
from abl.nn.text_encoder import TextEncoder
from abl.models.concept_embedding import ConceptEmbedding
from abl.models.quasi_symbolic import TextEncoder as QSTextEncoder
from abl.models.quasi_symbolic import ConceptDetector, QuasiSymbolicReasoning
from abl.models.reasoning_v1 import ABLReasoning
from abl.datasets.program_executor import TextProgramExecutor
from abl.datasets.program_translator import ProgramTranslator
from abl.utils.html_table import HTMLTableVisualizer
from abl.utils.misc import as_numpy, as_float


def main():
    parser = argparse.ArgumentParser('ABL Text Reasoning Testing')
    parser.add_argument('--config', required=True, help='configuration file')
    parser.add_argument('--dataset-config', required=True, help='dataset configuration file')
    parser.add_argument('--checkpoint', required=True, help='checkpoint file')
    parser.add_argument('--output-dir', required=True, help='output directory')
    parser.add_argument('--batch-size', type=int, default=16, help='batch size')
    parser.add_argument('--device', default='cuda', help='device')
    parser.add_argument('--split', default='test', help='dataset split to evaluate')
    parser.add_argument('--visualize', action='store_true', help='generate visualization')
    parser.add_argument('--interpret-rules', action='store_true', help='interpret learned rules')
    args = parser.parse_args()
    
    # Load configurations
    with open(args.config, 'r') as f:
        config = json.load(f)
    
    with open(args.dataset_config, 'r') as f:
        dataset_config = json.load(f)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Create dataset
    test_dataset = make_dataset_split(
        dataset_config['name'],
        dataset_config,
        args.split
    )
    
    # Create data loader
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4
    )
    
    # Get dataset statistics
    test_stats = test_dataset.get_stats()
    num_concepts = test_stats['num_concepts'] if test_stats['num_concepts'] > 0 else config['model']['num_concepts']
    num_labels = test_stats['num_labels']
    
    # Create model
    device = torch.device(args.device if torch.cuda.is_available() and args.device == 'cuda' else 'cpu')
    
    # Text encoder
    text_encoder = TextEncoder(
        pretrained_model=config['model']['text_encoder']['pretrained_model'],
        output_dim=config['model']['text_encoder']['output_dim'],
        use_pooler=config['model']['text_encoder'].get('use_pooler', False),
        freeze_bert=config['model']['text_encoder'].get('freeze_bert', False)
    )
    
    # Concept embedding
    concept_embedding = ConceptEmbedding(
        text_feature_dim=config['model']['text_encoder']['output_dim'],
        embedding_dim=config['model']['concept_embedding']['embedding_dim'],
        nr_concepts=num_concepts
    )
    
    # Create quasi-symbolic model
    qs_text_encoder = QSTextEncoder(text_encoder)
    concept_detector = ConceptDetector(concept_embedding)
    quasi_symbolic = QuasiSymbolicReasoning(qs_text_encoder, concept_detector)
    
    # Create reasoning model
    reasoning = ABLReasoning(
        concept_embedding_dim=config['model']['concept_embedding']['embedding_dim'],
        hidden_dim=config['model']['reasoning']['hidden_dim'],
        num_rules=config['model']['reasoning']['num_rules'],
        num_concepts=num_concepts,
        num_labels=num_labels
    )
    
    # Move models to device
    quasi_symbolic = quasi_symbolic.to(device)
    reasoning = reasoning.to(device)
    
    # Load checkpoint
    checkpoint = torch.load(args.checkpoint, map_location=device)
    quasi_symbolic.load_state_dict(checkpoint['quasi_symbolic'])
    reasoning.load_state_dict(checkpoint['reasoning'])
    
    print(f"Loaded checkpoint '{args.checkpoint}' (epoch {checkpoint['epoch']})")
    
    # Get concept and label mappings
    concept_names = test_dataset.get_concept_names() if hasattr(test_dataset, 'get_concept_names') else None
    label_names = test_dataset.get_label_names() if hasattr(test_dataset, 'get_label_names') else None
    # print("here 1")
    # Create program translator and executor
    translator = ProgramTranslator(
        concept_mapping=test_dataset.concept_mapping if hasattr(test_dataset, 'concept_mapping') else None,
        label_mapping=test_dataset.label_mapping if hasattr(test_dataset, 'label_mapping') else None
    )
    # print("here 2")
    executor = TextProgramExecutor(
        concept_names=concept_names
    )
    # Evaluate model
    # print(f"Evaluating model on '{args.split}' split...")
    metrics = evaluate(
        quasi_symbolic,
        reasoning,
        test_loader,
        device,
        translator,
        executor,
        test_dataset.inv_label_mapping if hasattr(test_dataset, 'inv_label_mapping') else None,
        concept_names
    )
    # print("here")
    # Save metrics
    with open(osp.join(args.output_dir, 'metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=2)
    
    print(f"Test accuracy: {metrics['accuracy']:.4f}")
    print(f"Test F1 score: {metrics['f1_score']:.4f}")
    
    # Generate visualization if requested
    if args.visualize:
        visualize(
            quasi_symbolic,
            reasoning,
            test_loader,
            device,
            osp.join(args.output_dir, 'visualization.html'),
            test_dataset.inv_label_mapping if hasattr(test_dataset, 'inv_label_mapping') else None,
            concept_names
        )
    
    # Interpret rules if requested
    if args.interpret_rules:
        interpret_rules(
            reasoning,
            concept_names,
            label_names,
            osp.join(args.output_dir, 'rules.json')
        )


def evaluate(quasi_symbolic, reasoning, data_loader, device, translator, executor, label_mapping=None, concept_names=None):
    """Evaluate the model."""
    quasi_symbolic.eval()
    reasoning.eval()
    
    all_predictions = []
    all_targets = []
    all_concept_probs = []
    all_rule_activations = []
    
    with torch.no_grad():
        for batch in data_loader:
            # Get data
            texts = batch['text']
            labels = batch['label_onehot'].to(device)
            label_indices = batch['label'].to(device)
            
            # Forward pass through quasi-symbolic model
            quasi_outputs = quasi_symbolic(texts)
            
            # Forward pass through reasoning model
            reasoning_outputs = reasoning(
                quasi_outputs['concept_logits'],
                quasi_outputs['concept_embeddings']
            )
            
            # Get predictions
            _, predicted = torch.max(reasoning_outputs['label_logits'], 1)
            
            # Store results
            all_predictions.extend(predicted.cpu().numpy())
            all_targets.extend(label_indices.cpu().numpy())
            all_concept_probs.append(reasoning_outputs['concept_probs'].cpu().numpy())
            all_rule_activations.append(reasoning_outputs['rule_activations'].cpu().numpy())
    
    # Concatenate results
    all_concept_probs = np.concatenate(all_concept_probs, axis=0)
    all_rule_activations = np.concatenate(all_rule_activations, axis=0)
    
    # Compute metrics
    accuracy = np.mean(np.array(all_predictions) == np.array(all_targets))
    
    # Compute F1 score (macro-averaged)
    f1_scores = []
    unique_labels = np.unique(all_targets)
    for label in unique_labels:
        true_positives = np.sum((np.array(all_predictions) == label) & (np.array(all_targets) == label))
        false_positives = np.sum((np.array(all_predictions) == label) & (np.array(all_targets) != label))
        false_negatives = np.sum((np.array(all_predictions) != label) & (np.array(all_targets) == label))
        
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
        
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        f1_scores.append(f1)
    
    f1_score = np.mean(f1_scores)
    
    # Compute confusion matrix
    num_classes = len(unique_labels)
    confusion_matrix = np.zeros((num_classes, num_classes), dtype=np.int64)
    for pred, target in zip(all_predictions, all_targets):
        confusion_matrix[target, pred] += 1
    
    # Compute concept activation statistics
    concept_stats = {}
    for i in range(all_concept_probs.shape[1]):
        concept_name = f'Concept-{i}' if concept_names is None or i >= len(concept_names) else concept_names[i]
        activations = all_concept_probs[:, i]
        concept_stats[concept_name] = {
            'mean': float(np.mean(activations)),
            'std': float(np.std(activations)),
            'min': float(np.min(activations)),
            'max': float(np.max(activations)),
            'median': float(np.median(activations))
        }
    
    # Compute rule activation statistics
    rule_stats = {}
    for i in range(all_rule_activations.shape[1]):
        activations = all_rule_activations[:, i]
        rule_stats[f'Rule-{i}'] = {
            'mean': float(np.mean(activations)),
            'std': float(np.std(activations)),
            'min': float(np.min(activations)),
            'max': float(np.max(activations)),
            'median': float(np.median(activations))
        }
    
    # Create metrics dictionary
    metrics = {
        'accuracy': float(accuracy),
        'f1_score': float(f1_score),
        'confusion_matrix': confusion_matrix.tolist(),
        'concept_stats': concept_stats,
        'rule_stats': rule_stats
    }
    
    # Convert label indices to names if mapping is available
    if label_mapping is not None:
        predictions_named = [label_mapping.get(p, f'Class-{p}') for p in all_predictions]
        targets_named = [label_mapping.get(t, f'Class-{t}') for t in all_targets]
        metrics['predictions_named'] = predictions_named
        metrics['targets_named'] = targets_named
    
    return metrics

def visualize(quasi_symbolic, reasoning, data_loader, device, output_file, label_mapping=None, concept_names=None):
    """Generate visualization."""
    quasi_symbolic.eval()
    reasoning.eval()
    
    # Create visualizer
    visualizer = HTMLTableVisualizer('ABL Text Reasoning Visualization')
    
    # Get a sample of texts for visualization
    sample_texts = []
    sample_outputs = []
    sample_labels = []
    
    with torch.no_grad():
        for batch in data_loader:
            # Get data
            texts = batch['text']
            labels = batch['label_onehot'].to(device)
            label_indices = batch['label'].to(device)
            
            # Forward pass through quasi-symbolic model
            quasi_outputs = quasi_symbolic(texts)
            
            # Forward pass through reasoning model
            reasoning_outputs = reasoning(
                quasi_outputs['concept_logits'],
                quasi_outputs['concept_embeddings']
            )
            
            # Store results
            sample_texts.extend(texts)
            sample_outputs.append({
                'concept_probs': reasoning_outputs['concept_probs'].cpu().numpy(),
                'rule_activations': reasoning_outputs['rule_activations'].cpu().numpy(),
                'label_logits': reasoning_outputs['label_logits'].cpu().numpy()
            })
            sample_labels.extend(label_indices.cpu().numpy())
            
            # Limit to a reasonable number of samples
            if len(sample_texts) >= 20:
                break
    
    # Concatenate outputs
    concatenated_outputs = {}
    for key in sample_outputs[0].keys():
        concatenated_outputs[key] = np.concatenate([o[key] for o in sample_outputs], axis=0)
    
    # Visualize samples
    for i, text in enumerate(sample_texts[:20]):  # Limit to 20 samples
        # Get prediction
        pred_idx = np.argmax(concatenated_outputs['label_logits'][i])
        pred_label = label_mapping.get(pred_idx, f'Class-{pred_idx}') if label_mapping else f'Class-{pred_idx}'
        
        # Get concept probabilities and rule activations
        concept_probs = concatenated_outputs['concept_probs'][i]
        rule_activations = concatenated_outputs['rule_activations'][i]
        
        # Add sample to visualizer
        visualizer.add_text_sample(
            text=text,
            label=pred_label,
            concept_probs=concept_probs,
            concept_names=concept_names,
            rule_activations=rule_activations
        )
    
    # Visualize concept embeddings
    if hasattr(reasoning, 'get_rules'):
        rule_weights, _ = reasoning.get_rules()
        rule_weights = as_numpy(rule_weights)
        
        visualizer.add_rule_analysis(rule_weights, concept_names)
    
    # Save visualization
    visualizer.save(output_file)
    print(f'Visualization saved to {output_file}')


def interpret_rules(reasoning, concept_names, label_names, output_file):
    """Interpret learned rules."""
    if not hasattr(reasoning, 'get_rules'):
        print("Reasoning model does not support rule extraction.")
        return
    
    # Get rule weights and biases
    rule_weights, rule_biases = reasoning.get_rules()
    rule_weights = as_numpy(rule_weights)
    rule_biases = as_numpy(rule_biases)
    
    # Create rule interpretation dictionary
    rules = {}
    
    for i in range(rule_weights.shape[0]):
        # Get rule weights
        weights = rule_weights[i]
        bias = rule_biases[i] if rule_biases is not None else 0.0
        
        # Find concepts with significant weights
        significant_indices = np.where(np.abs(weights) > 0.1)[0]
        significant_weights = weights[significant_indices]
        
        # Create rule description
        rule = {
            'id': f'Rule-{i}',
            'bias': float(bias),
            'concepts': []
        }
        
        for idx, weight in zip(significant_indices, significant_weights):
            concept_name = concept_names[idx] if concept_names and idx < len(concept_names) else f'Concept-{idx}'
            rule['concepts'].append({
                'concept': concept_name,
                'weight': float(weight),
                'contribution': 'positive' if weight > 0 else 'negative'
            })
        
        # Sort concepts by absolute weight
        rule['concepts'] = sorted(rule['concepts'], key=lambda x: abs(x['weight']), reverse=True)
        
        # Generate human-readable description
        description = f"Rule {i}: "
        if rule['concepts']:
            for j, concept in enumerate(rule['concepts']):
                if j > 0:
                    description += " AND "
                if concept['weight'] < 0:
                    description += f"NOT {concept['concept']}"
                else:
                    description += concept['concept']
        else:
            description += "No significant concepts found."
        
        rule['description'] = description
        rules[f'Rule-{i}'] = rule
    
    # Save rule interpretations
    with open(output_file, 'w') as f:
        json.dump(rules, f, indent=2)
    
    print(f'Rule interpretations saved to {output_file}')


if __name__ == '__main__':
    main()