#! /usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer, BertConfig


class TextEncoder(nn.Module):
    """Text encoder using BERT for feature extraction"""
    
    def __init__(self, 
                 pretrained_model="bert-base-uncased", 
                 output_dim=768, 
                 use_pooler=False, 
                 freeze_bert=False):
        super().__init__()
        
        self.use_pooler = use_pooler
        self.output_dim = output_dim
        # Initialize the BERT model and tokenizer
        self.tokenizer = BertTokenizer.from_pretrained(pretrained_model)
        self.bert = BertModel.from_pretrained(pretrained_model)
        
        # Freeze BERT parameters if specified
        if freeze_bert:
            for param in self.bert.parameters():
                param.requires_grad = False
        
        # Add projection layer if the output dimension is different from BERT's hidden size
        if output_dim != self.bert.config.hidden_size:
            self.projection = nn.Linear(self.bert.config.hidden_size, output_dim)
            self.use_projection = True
        else:
            self.use_projection = False
    
    def forward(self, texts, return_sequence=False):
        """
        Forward pass through the text encoder.
        
        Args:
            texts: list of string texts
            return_sequence: whether to return the full sequence of token embeddings
            
        Returns:
            If return_sequence is False:
                tensor of shape [batch_size, output_dim]
            If return_sequence is True:
                tuple of (pooled_output, sequence_output)
                pooled_output: tensor of shape [batch_size, output_dim]
                sequence_output: tensor of shape [batch_size, seq_len, output_dim]
        """
        # Tokenize the input texts
        inputs = self.tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
        
        # Move inputs to the same device as the model
        device = next(self.bert.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Get BERT outputs
        outputs = self.bert(**inputs)
        
        # Process the output based on configuration
        if self.use_pooler:
            pooled_output = outputs.pooler_output
        else:
            # Use [CLS] token embedding as the text representation
            pooled_output = outputs.last_hidden_state[:, 0]
        
        # Apply projection if needed
        if self.use_projection:
            pooled_output = self.projection(pooled_output)
            if return_sequence:
                sequence_output = self.projection(outputs.last_hidden_state)
        elif return_sequence:
            sequence_output = outputs.last_hidden_state
        
        if return_sequence:
            return pooled_output, sequence_output
        else:
            return pooled_output
