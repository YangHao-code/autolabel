#! /usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import os.path as osp
import sys
import json
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

from abl.datasets.factory import make_dataset_split
from abl.nn.text_encoder import TextEncoder
from abl.models.concept_embedding import ConceptEmbedding
from abl.models.quasi_symbolic import TextEncoder as QSTextEncoder
from abl.models.quasi_symbolic import ConceptDetector, QuasiSymbolicReasoning
from abl.models.reasoning_v1 import ABLReasoning
import shutil

def main():
    parser = argparse.ArgumentParser('ABL Text Reasoning')
    parser.add_argument('--config', required=True, help='configuration file')
    parser.add_argument('--dataset-config', required=True, help='dataset configuration file')
    parser.add_argument('--output-dir', required=True, help='output directory')
    parser.add_argument('--epochs', type=int, default=50, help='number of training epochs')
    parser.add_argument('--batch-size', type=int, default=32, help='batch size')
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
    parser.add_argument('--weight-decay', type=float, default=1e-4, help='weight decay')
    parser.add_argument('--device', default='cuda', help='device')
    parser.add_argument('--resume', default='', help='checkpoint to resume from')
    parser.add_argument('--seed', type=int, default=42, help='random seed')
    parser.add_argument('--accum-steps', type=int, default=4, help='gradient accumulation steps')
    parser.add_argument('--concept-loss-weight', type=float, default=0.5, help='weight for concept loss')
    args = parser.parse_args()
    
    # Set random seed
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    
    # Load configurations
    with open(args.config, 'r') as f:
        config = json.load(f)
    
    with open(args.dataset_config, 'r') as f:
        dataset_config = json.load(f)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    # Save configurations
    with open(osp.join(args.output_dir, 'config.json'), 'w') as f:
        json.dump(config, f, indent=2)
    with open(osp.join(args.output_dir, 'dataset_config.json'), 'w') as f:
        json.dump(dataset_config, f, indent=2)
    
    # Create datasets
    train_dataset = make_dataset_split(
        dataset_config['name'],
        dataset_config,
        'train'
    )
    
    val_dataset = make_dataset_split(
        dataset_config['name'],
        dataset_config,
        'val'
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4
    )
    
    # Get dataset statistics
    train_stats = train_dataset.get_stats()
    num_concepts = train_stats['num_concepts'] if train_stats['num_concepts'] > 0 else config['model']['num_concepts']
    num_labels = train_stats['num_labels']
    
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
    # quasi_symbolic = torch.compile(quasi_symbolic)
    # reasoning = torch.compile(reasoning)

    # Create optimizer
    optimizer = optim.AdamW(
        list(quasi_symbolic.parameters()) + list(reasoning.parameters()),
        lr=args.lr,
        weight_decay=args.weight_decay
    )

    # Create loss functions
    classification_loss_fn = nn.CrossEntropyLoss()
    concept_loss_fn = nn.BCEWithLogitsLoss()
    
    # Create TensorBoard writer
    if os.path.exists(osp.join(args.output_dir, 'tensorboard')):
        shutil.rmtree(osp.join(args.output_dir, 'tensorboard'))
    writer = SummaryWriter(osp.join(args.output_dir, 'tensorboard'))
    
    # Resume from checkpoint if specified
    start_epoch = 0
    if args.resume:
        if osp.isfile(args.resume):
            print(f"Loading checkpoint '{args.resume}'")
            checkpoint = torch.load(args.resume, map_location=device)
            start_epoch = checkpoint['epoch']
            quasi_symbolic.load_state_dict(checkpoint['quasi_symbolic'])
            reasoning.load_state_dict(checkpoint['reasoning'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print(f"Loaded checkpoint '{args.resume}' (epoch {checkpoint['epoch']})")
        else:
            print(f"No checkpoint found at '{args.resume}'")
    
    # Training loop
    best_val_acc = 0.0
    
    for epoch in range(start_epoch, args.epochs):
        # Training
        train_one_epoch(
            quasi_symbolic,
            reasoning,
            train_loader,
            optimizer,
            classification_loss_fn,
            concept_loss_fn,
            device,
            epoch,
            writer,
            args.accum_steps
        )
        
        # Validation
        val_loss, val_acc, val_f1 = validate(
            quasi_symbolic,
            reasoning,
            val_loader,
            classification_loss_fn,
            device
        )
        
        # TensorBoard logging
        writer.add_scalar('val/loss', val_loss, epoch)
        writer.add_scalar('val/accuracy', val_acc, epoch)
        writer.add_scalar('val/f1', val_f1, epoch)
        
        # Save checkpoint
        is_best = val_acc > best_val_acc
        best_val_acc = max(val_acc, best_val_acc)
        
        save_checkpoint(
            {
                'epoch': epoch + 1,
                'quasi_symbolic': quasi_symbolic.state_dict(),
                'reasoning': reasoning.state_dict(),
                'optimizer': optimizer.state_dict(),
                'val_acc': val_acc,
                'best_val_acc': best_val_acc
            },
            is_best,
            osp.join(args.output_dir, 'checkpoints')
        )
        
        print(f'Epoch {epoch+1}/{args.epochs} complete. Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, Val F1: {val_f1:.4f}, (Best: {best_val_acc:.4f})')
    
    # Close TensorBoard writer
    writer.close()
    
    print('Training complete!')
    print(f'Best validation accuracy: {best_val_acc:.4f}')

def train_one_epoch(quasi_symbolic, reasoning, data_loader, optimizer, cls_loss_fn, 
                   concept_loss_fn, device, epoch, writer, accum_steps):
    """Train for one epoch with gradient accumulation."""
    quasi_symbolic.train()
    reasoning.train()
    
    total_loss = 0.0
    total_cls_loss = 0.0
    total_concept_loss = 0.0
    correct = 0
    total = 0
    
    optimizer.zero_grad()
    accum_counter = 0  # 梯度累积计数器
    
    for batch_idx, batch in enumerate(data_loader):
        # 获取数据
        texts = batch['text']
        labels = batch['label_onehot'].to(device)
        label_indices = batch['label'].to(device)
        
        # 获取概念标注
        has_concept_annot = 'concept_annotations' in batch.keys()
        if has_concept_annot:
            concept_annots = torch.stack([item['concept_annotations'] for item in batch['concept_annotations']]).to(device)
        
        # 前向传播
        quasi_outputs = quasi_symbolic(texts, return_features=True)
        reasoning_outputs = reasoning(
            quasi_outputs['concept_logits'],
            quasi_outputs['concept_embeddings']
        )
        
        # 计算损失
        cls_loss = cls_loss_fn(reasoning_outputs['label_logits'], label_indices)
        if has_concept_annot:
            concept_loss = concept_loss_fn(quasi_outputs['concept_logits'], concept_annots)
            raw_loss = cls_loss + args.concept_loss_weight * concept_loss
        else:
            concept_loss = torch.tensor(0.0, device=device)
            raw_loss = cls_loss
        
        # 梯度缩放与累积
        scaled_loss = raw_loss / accum_steps# 按累积步数缩放
        scaled_loss.backward()
        # raw_loss.backward()
        
        # 更新指标（使用原始loss）
        total_loss += raw_loss.item()
        total_cls_loss += cls_loss.item()
        if has_concept_annot:
            total_concept_loss += concept_loss.item()
        
        # 计算准确率
        _, predicted = torch.max(reasoning_outputs['label_logits'], 1)
        total += label_indices.size(0)
        correct += (predicted == label_indices).sum().item()
        
        # 梯度累积计数器更新
        accum_counter += 1
        
        # 达到累积步数或最后一个batch时更新参数
        if (accum_counter % accum_steps == 0) or (batch_idx == len(data_loader)-1):
            optimizer.step()
            optimizer.zero_grad()
            accum_counter = 0  # 重置计数器
            
            # 打印进度（仅在实际更新参数时）
            if (batch_idx + 1) % (10 * accum_steps) == 0:
                avg_loss_sofar = total_loss / (batch_idx + 1)
                print(f'Epoch {epoch+1}, Batch {batch_idx+1}/{len(data_loader)}, '
                      f'Avg Loss: {avg_loss_sofar:.4f}')
    
    # 计算epoch指标
    avg_loss = total_loss / len(data_loader)
    avg_cls_loss = total_cls_loss / len(data_loader)
    avg_concept_loss = total_concept_loss / len(data_loader) if has_concept_annot else 0.0
    accuracy = correct / total
    
    # TensorBoard日志
    writer.add_scalar('train/loss', avg_loss, epoch)
    writer.add_scalar('train/cls_loss', avg_cls_loss, epoch)
    writer.add_scalar('train/concept_loss', avg_concept_loss, epoch)
    writer.add_scalar('train/accuracy', accuracy, epoch)
    
    print(f'Epoch {epoch+1} - Train Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}')
    return avg_loss, accuracy

def train_one_epoch_p(quasi_symbolic, reasoning, data_loader, optimizer, cls_loss_fn, concept_loss_fn, device, epoch, writer):
    """Train for one epoch."""
    quasi_symbolic.train()
    reasoning.train()
    
    total_loss = 0.0
    total_cls_loss = 0.0
    total_concept_loss = 0.0
    correct = 0
    total = 0
    
    for batch_idx, batch in enumerate(data_loader):
        # Get data
        # print("batch_idx", batch[0])

        # texts = [item for item in batch['text']]
        # labels = torch.stack([item for item in batch['label']]).to(device)
        # print(labels)
        # label_indices = torch.argmax(labels, dim= -1, keepdim = True)
        # print(label_indices)

        texts = batch['text']
        labels = batch['label_onehot'].to(device)
        label_indices = batch['label'].to(device)
        # Get concept annotations if available
        has_concept_annot = 'concept_annotations' in batch.keys()
        if has_concept_annot:
            concept_annots = torch.stack([item['concept_annotations'] for item in batch['concept_annotations']]).to(device)
        
        # Forward pass through quasi-symbolic model
        quasi_outputs = quasi_symbolic(texts, return_features=True)
        # Forward pass through reasoning model
        reasoning_outputs = reasoning(
            quasi_outputs['concept_logits'],
            quasi_outputs['concept_embeddings']
        )
        # Compute classification loss
        cls_loss = cls_loss_fn(reasoning_outputs['label_logits'], label_indices)
        
        # Compute concept loss if concept annotations are available
        if has_concept_annot:
            concept_loss = concept_loss_fn(quasi_outputs['concept_logits'], concept_annots)
            loss = cls_loss + concept_loss
        else:
            concept_loss = torch.tensor(0.0, device=device)
            loss = cls_loss
        
        # Backward pass and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Update metrics
        total_loss += loss.item()
        total_cls_loss += cls_loss.item()
        total_concept_loss += concept_loss.item() if has_concept_annot else 0.0
        
        # Compute accuracy
        _, predicted = torch.max(reasoning_outputs['label_logits'], 1)
        total += label_indices.size(0)
        correct += (predicted == label_indices).sum().item()
        
        # Log progress
        if (batch_idx + 1) % 10 == 0:
            print(f'Epoch {epoch+1}, Batch {batch_idx+1}/{len(data_loader)}, Loss: {loss.item():.4f}')
    
    # Compute epoch metrics
    avg_loss = total_loss / len(data_loader)
    avg_cls_loss = total_cls_loss / len(data_loader)
    avg_concept_loss = total_concept_loss / len(data_loader)
    accuracy = correct / total
    
    # TensorBoard logging
    writer.add_scalar('train/loss', avg_loss, epoch)
    writer.add_scalar('train/cls_loss', avg_cls_loss, epoch)
    writer.add_scalar('train/concept_loss', avg_concept_loss, epoch)
    writer.add_scalar('train/accuracy', accuracy, epoch)
    
    print(f'Epoch {epoch+1} - Train Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}')


def validate(quasi_symbolic, reasoning, data_loader, loss_fn, device):
    """Validate the model."""
    quasi_symbolic.eval()
    reasoning.eval()
    
    total_loss = 0.0
    correct = 0
    total = 0
    
    # Variables for F1 calculation
    total_label = 0
    total_pred = 0
    total_correct = 0
    
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
            
            # Compute loss
            loss = loss_fn(reasoning_outputs['label_logits'], label_indices)
            total_loss += loss.item()
            
            # Compute accuracy
            _, predicted = torch.max(reasoning_outputs['label_logits'], 1)
            total += label_indices.size(0)
            correct += (predicted == label_indices).sum().item()
            
            # Accumulate values for F1 calculation
            batch_size = label_indices.size(0)
            total_label += batch_size
            total_pred += batch_size  # One prediction per sample
            total_correct += (predicted == label_indices).sum().item()
    
    # Compute metrics
    avg_loss = total_loss / len(data_loader)
    accuracy = correct / total
    
    # Calculate global F1 score
    precision = total_correct / (total_pred + 1e-8)
    recall = total_correct / (total_label + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    
    return avg_loss, accuracy, f1



def save_checkpoint(state, is_best, checkpoint_dir):
    """Save checkpoint."""
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Save checkpoint
    torch.save(state, osp.join(checkpoint_dir, 'checkpoint.pth'))
    
    # Save best model
    if is_best:
        torch.save(state, osp.join(checkpoint_dir, 'model_best.pth'))


if __name__ == '__main__':
    main()
