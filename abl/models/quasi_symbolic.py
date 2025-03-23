#! /usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F


class TextEncoder(nn.Module):
    """
    Text encoder module for the Quasi-Symbolic module.
    Adapts the interface to work with the original NSCL framework.
    """
    
    def __init__(self, text_encoder):
        super().__init__()
        self.text_encoder = text_encoder
    
    def forward(self, text_batch, return_features=False):
        """
        Args:
            text_batch: list of text strings
            return_features: whether to return sequence features
        
        Returns:
            text_features: tensor of shape [batch_size, feature_dim]
            seq_features: tensor of shape [batch_size, seq_len, feature_dim] if return_features=True
        """
        if return_features:
            text_features, seq_features = self.text_encoder(text_batch, return_sequence=True)
            return text_features, seq_features
        else:
            text_features = self.text_encoder(text_batch)
            return text_features


class ConceptDetector(nn.Module):
    """
    Concept detector module for text input.
    Detects concepts present in text based on text features.
    """
    
    def __init__(self, concept_embedding):
        super().__init__()
        self.concept_embedding = concept_embedding
    
    def forward(self, text_features):
        """
        Detect concepts from text features.
        
        Args:
            text_features: tensor of shape [batch_size, feature_dim]
        
        Returns:
            concept_logits: tensor of shape [batch_size, nr_concepts]
            concept_embedding: tensor of shape [batch_size, nr_concepts, embedding_dim]
        """
        return self.concept_embedding(text_features)


class QuasiSymbolicReasoning(nn.Module):
    """
    Main module for Neuro-Symbolic reasoning on text data.
    """
    
    def __init__(self, text_encoder, concept_detector):
        super().__init__()
        
        self.text_encoder = text_encoder
        self.concept_detector = concept_detector
    
    def forward(self, texts, return_features=False):
        """
        Forward pass for quasi-symbolic reasoning on text.
        
        Args:
            texts: list of text strings
            return_features: whether to return intermediate features
            
        Returns:
            dictionary containing:
                - concepts: concept logits
                - concept_embeddings: concept embeddings
                - text_features: text features (if return_features=True)
        """
        # Encode text
        if return_features:
            text_features, seq_features = self.text_encoder(texts, return_features=True)
        else:
            text_features = self.text_encoder(texts)
        
        # Detect concepts
        concept_logits, concept_embeddings = self.concept_detector(text_features)
        
        # Prepare output
        ret = {
            'concept_logits': concept_logits,
            'concept_embeddings': concept_embeddings
        }
        
        if return_features:
            ret.update({
                'text_features': text_features,
                'sequence_features': seq_features
            })
        
        return ret
