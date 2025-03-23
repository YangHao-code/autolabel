#! /usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn


class ConceptEmbedding(nn.Module):
    """
    Embedding module for concepts in text.
    
    This module creates embeddings for the detected concepts in text data.
    Each concept is represented by a learned vector.
    """
    
    def __init__(self, nr_concepts, concept_dim):
        super().__init__()
        
        self.nr_concepts = nr_concepts
        self.concept_dim = concept_dim
        
        # Initialize concept embeddings
        self.embedding = nn.Embedding(nr_concepts, concept_dim)
    
    def forward(self, x):
        """
        Args:
            x: concept indices of shape [batch_size, nr_concepts]
               or one-hot vectors of shape [batch_size, nr_concepts]
        
        Returns:
            embeddings: tensor of shape [batch_size, nr_concepts, concept_dim]
        """
        if x.dim() == 2 and x.size(1) == self.nr_concepts:
            # One-hot input: perform embedding lookup
            x = x.float()
            embedding = torch.matmul(x, self.embedding.weight)
            return embedding.unsqueeze(1)  # [batch_size, 1, concept_dim]
        else:
            # Index input: use standard embedding lookup
            return self.embedding(x)  # [batch_size, *, concept_dim]


class TextFeatureEmbedding(nn.Module):
    """
    Module to extract concept features from text embeddings.
    
    This module projects text features to the concept space and provides
    a mechanism to detect the presence of concepts in text.
    """
    
    def __init__(self, feature_dim, concept_dim, nr_concepts):
        super().__init__()
        
        self.feature_dim = feature_dim
        self.concept_dim = concept_dim
        self.nr_concepts = nr_concepts
        
        # Linear projection from text feature space to concept space
        self.projection = nn.Linear(feature_dim, concept_dim)
        
        # Concept detectors as linear layers
        self.concept_detectors = nn.Linear(concept_dim, nr_concepts)
    
    def forward(self, features):
        """
        Args:
            features: text features of shape [batch_size, feature_dim]
                     or [batch_size, seq_len, feature_dim]
        
        Returns:
            concept_logits: logits for concept presence [batch_size, nr_concepts]
            projected_features: features in concept space
        """
        # Handle sequence inputs by averaging
        if features.dim() == 3:
            features = features.mean(dim=1)  # [batch_size, feature_dim]
        
        # Project features to concept space
        projected = self.projection(features)  # [batch_size, concept_dim]
        
        # Detect concepts
        concept_logits = self.concept_detectors(projected)  # [batch_size, nr_concepts]
        
        return concept_logits, projected
