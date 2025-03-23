#! /usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
import math 


class ConceptEmbedding(nn.Module):
    """
    Concept embedding for the Neural-Symbolic Concept Learner.
    This component maps text features to concept space and detects concept presence.
    """
    
    def __init__(self, text_feature_dim, embedding_dim, nr_concepts):
        super().__init__()
        
        self.text_feature_dim = text_feature_dim
        self.embedding_dim = embedding_dim
        self.nr_concepts = nr_concepts
        
        # Feature transformation layers
        self.feature_transform = nn.Sequential(
            nn.Linear(text_feature_dim, embedding_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        # Concept detectors
        self.concept_kernels = nn.Parameter(torch.Tensor(nr_concepts, embedding_dim))
        self.concept_biases = nn.Parameter(torch.Tensor(nr_concepts))
        
        # Initialize parameters
        nn.init.kaiming_uniform_(self.concept_kernels, a=math.sqrt(5))
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.concept_kernels)
        bound = 1 / math.sqrt(fan_in)
        nn.init.uniform_(self.concept_biases, -bound, bound)
        
        # Learned concept embeddings for reasoning
        self.embedding = nn.Embedding(nr_concepts, embedding_dim)
    
    def forward(self, features, return_embedding=True):
        """
        Forward pass to detect concepts and get embeddings.
        
        Args:
            features: text features of shape [batch_size, feature_dim]
            return_embedding: whether to return concept embeddings
            
        Returns:
            concept_logits: logits indicating concept presence [batch_size, nr_concepts]
            embeddings: concept embeddings if return_embedding is True
        """
        # Transform features to embedding space
        transformed = self.feature_transform(features)  # [batch_size, embedding_dim]
        
        # Compute concept logits
        concept_logits = F.linear(transformed, self.concept_kernels, self.concept_biases)
        
        if return_embedding:
            # Get concept embeddings for each detected concept
            batch_size = features.size(0)
            concept_probs = torch.sigmoid(concept_logits)
            
            # Weighted concept embeddings
            all_concept_embeddings = self.embedding.weight.unsqueeze(0).expand(
                batch_size, -1, -1)  # [batch_size, nr_concepts, embedding_dim]
            
            # Weight embeddings by concept probabilities
            weighted_embeddings = concept_probs.unsqueeze(-1) * all_concept_embeddings
            
            return concept_logits, weighted_embeddings
        else:
            return concept_logits
    
    def get_concept_embeddings(self):
        """Get all concept embeddings."""
        return self.embedding.weight  # [nr_concepts, embedding_dim]