#! /usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F


class GroundingOperator(nn.Module):
    """
    Grounding operator that maps concept detections to symbolic representations.
    """
    
    def __init__(self, concept_embedding_dim):
        super().__init__()
        self.embedding_dim = concept_embedding_dim
        
        # Attention mechanism for focusing on relevant concepts
        self.attention = nn.Sequential(
            nn.Linear(concept_embedding_dim, concept_embedding_dim // 2),
            nn.ReLU(),
            nn.Linear(concept_embedding_dim // 2, 1)
        )
    
    def forward(self, concept_logits, concept_embeddings):
        """
        Ground concepts based on their detection scores.
        
        Args:
            concept_logits: tensor of shape [batch_size, nr_concepts]
            concept_embeddings: tensor of shape [batch_size, nr_concepts, embedding_dim]
            
        Returns:
            grounded_concepts: tensor of shape [batch_size, embedding_dim]
        """
        # Apply sigmoid to get concept probabilities
        concept_probs = torch.sigmoid(concept_logits)  # [batch_size, nr_concepts]
        
        # Compute attention over concepts
        attention_scores = self.attention(concept_embeddings).squeeze(-1)  # [batch_size, nr_concepts]
        attention_weights = F.softmax(attention_scores, dim=1)  # [batch_size, nr_concepts]
        
        # Combine with concept probabilities
        combined_weights = concept_probs * attention_weights  # [batch_size, nr_concepts]
        
        # Normalize weights
        normalized_weights = combined_weights / (combined_weights.sum(dim=1, keepdim=True) + 1e-8)
        
        # Apply attention to get grounded representation
        grounded_concepts = torch.bmm(normalized_weights.unsqueeze(1), concept_embeddings).squeeze(1)
        
        return grounded_concepts


class RuleNetwork(nn.Module):
    """
    Network that learns and applies logical rules for reasoning.
    """
    
    def __init__(self, input_dim, hidden_dim, num_rules, num_concepts):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_rules = num_rules
        self.num_concepts = num_concepts
        
        # Rule representation layers
        self.rule_generator = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        # Rule application layer
        self.rule_weights = nn.Parameter(torch.Tensor(num_rules, num_concepts))
        self.rule_bias = nn.Parameter(torch.Tensor(num_rules))
        
        # Rule combination layer for final reasoning
        self.rule_combiner = nn.Linear(hidden_dim + num_rules, hidden_dim)
        
        # Initialize rule parameters
        nn.init.xavier_uniform_(self.rule_weights)
        nn.init.zeros_(self.rule_bias)
    
    def forward(self, concept_logits, grounded_concepts):
        """
        Apply rule-based reasoning to concepts.
        
        Args:
            concept_logits: tensor of shape [batch_size, num_concepts]
            grounded_concepts: tensor of shape [batch_size, input_dim]
            
        Returns:
            reasoning_output: tensor of shape [batch_size, hidden_dim]
            rule_activations: tensor of shape [batch_size, num_rules]
        """
        # Neural reasoning path
        neural_features = self.rule_generator(grounded_concepts)
        
        # Symbolic rule application
        concept_probs = torch.sigmoid(concept_logits)
        rule_activations = F.linear(concept_probs, self.rule_weights, self.rule_bias)
        rule_probs = torch.sigmoid(rule_activations)
        
        # Combine neural and symbolic paths
        combined = torch.cat([neural_features, rule_probs], dim=1)
        reasoning_output = self.rule_combiner(combined)
        
        return reasoning_output, rule_activations


class NSCLReasoning(nn.Module):
    """
    Main reasoning module for NSCL applied to text labeling.
    """
    
    def __init__(self, concept_embedding_dim, hidden_dim, num_rules, num_concepts, num_labels):
        super().__init__()
        
        self.concept_embedding_dim = concept_embedding_dim
        self.hidden_dim = hidden_dim
        self.num_rules = num_rules
        self.num_concepts = num_concepts
        self.num_labels = num_labels
        
        # Grounding operator
        self.grounding = GroundingOperator(concept_embedding_dim)
        
        # Rule network
        self.rule_network = RuleNetwork(
            input_dim=concept_embedding_dim,
            hidden_dim=hidden_dim,
            num_rules=num_rules,
            num_concepts=num_concepts
        )
        
        # Final classification layer
        self.classifier = nn.Linear(hidden_dim, num_labels)
    
    def forward(self, concept_logits, concept_embeddings):
        """
        Perform neuro-symbolic reasoning for text classification.
        
        Args:
            concept_logits: tensor of shape [batch_size, num_concepts]
            concept_embeddings: tensor of shape [batch_size, num_concepts, concept_embedding_dim]
            
        Returns:
            dictionary containing:
                - label_logits: logits for label prediction
                - rule_activations: rule activation scores
                - concept_probs: concept detection probabilities
        """
        # Ground concepts
        grounded_concepts = self.grounding(concept_logits, concept_embeddings)
        
        # Apply rule-based reasoning
        reasoning_output, rule_activations = self.rule_network(concept_logits, grounded_concepts)
        
        # Final classification
        label_logits = self.classifier(reasoning_output)
        
        return {
            'label_logits': label_logits,
            'rule_activations': rule_activations,
            'concept_probs': torch.sigmoid(concept_logits)
        }
    
    def get_rules(self):
        """
        Extract learned rules from the model.
        
        Returns:
            rules: tensor of shape [num_rules, num_concepts]
            rule_bias: tensor of shape [num_rules]
        """
        return self.rule_network.rule_weights, self.rule_network.rule_bias
