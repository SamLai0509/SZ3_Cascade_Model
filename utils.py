"""
GPU and utility functions for NeurLZ.
"""

import torch
import torch.nn as nn


def setup_multi_gpu_model(model, device_ids=None):
    """
    Setup model for multi-GPU training using DataParallel.
        
    Args:
        model: PyTorch model
        device_ids: List of GPU IDs to use (None = use all available GPUs)
        
    Returns:
        model: Model wrapped with DataParallel if multiple GPUs available
        device: Primary device (cuda:0)
    """
    if device_ids is None:
        device_ids = list(range(torch.cuda.device_count()))
    
    if len(device_ids) > 1:
        print(f"  Using {len(device_ids)} GPUs: {device_ids}")
        model = nn.DataParallel(model, device_ids=device_ids)
        device = f'cuda:{device_ids[0]}'
    else:
        device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    return model, device


def get_available_gpus():
    """Get list of available GPU IDs."""
    if torch.cuda.is_available():
        return list(range(torch.cuda.device_count()))
    return []