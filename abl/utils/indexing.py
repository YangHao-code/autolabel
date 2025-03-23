#! /usr/bin/env python3
# -*- coding: utf-8 -*-

import torch


def batch_index_select(tensor, index, dim=1):
    """
    Batch index select.
    
    Args:
        tensor: tensor of shape [batch_size, ...]
        index: tensor of shape [batch_size, num_indices]
        dim: dimension to select
        
    Returns:
        selected: tensor of shape [batch_size, num_indices, ...]
    """
    assert tensor.dim() >= 2, 'Tensor must have at least 2 dimensions'
    assert index.dim() == 2, 'Index must have 2 dimensions'
    
    batch_size = tensor.size(0)
    num_indices = index.size(1)
    
    # Reshape index to [batch_size * num_indices]
    flat_index = index.view(-1)
    
    # Create batch indices
    batch_indices = torch.arange(batch_size, device=tensor.device).repeat_interleave(num_indices)
    
    # Reshape and permute tensor to [batch_size, dim, ...]
    tensor_transposed = tensor.transpose(1, dim)
    
    # Select values
    selected = tensor_transposed[batch_indices, flat_index]
    
    # Reshape to [batch_size, num_indices, ...]
    selected = selected.view(batch_size, num_indices, *tensor.size()[2:])
    
    # Permute back if necessary
    if dim != 1:
        selected = selected.transpose(1, dim)
    
    return selected


def masked_average(tensor, mask, dim=-1, keepdim=False, eps=1e-8):
    """
    Compute masked average along a dimension.
    
    Args:
        tensor: tensor to average
        mask: binary mask
        dim: dimension to average over
        keepdim: whether to keep dimensions
        eps: small epsilon for numerical stability
        
    Returns:
        masked average
    """
    # Convert mask to float and expand it if needed
    mask = mask.float()
    while mask.dim() < tensor.dim():
        mask = mask.unsqueeze(-1)
    
    # Apply mask
    masked_tensor = tensor * mask
    
    # Compute average
    mask_sum = mask.sum(dim=dim, keepdim=keepdim)
    avg = masked_tensor.sum(dim=dim, keepdim=keepdim) / (mask_sum + eps)
    
    return avg
