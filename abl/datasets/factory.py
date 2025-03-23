#! /usr/bin/env python3
# -*- coding: utf-8 -*-

from .definition import TextLabelingDataset

def get_dataset(name):
    """
    Get dataset class by name.
    
    Args:
        name: dataset name
        
    Returns:
        dataset_class: the dataset class
    """
    if name == 'text' or name == 'text_dataset':
        return TextLabelingDataset
    else:
        raise ValueError(f'Unknown dataset: {name}')

def make_dataset(name, configs, **kwargs):
    """
    Make a dataset instance.
    
    Args:
        name: dataset name
        configs: dataset configuration
        **kwargs: additional arguments for dataset constructor
        
    Returns:
        dataset: the dataset instance
    """
    dataset_class = get_dataset(name)
    # Remove 'name' from configs before passing to the constructor
    configs_copy = dict(configs)
    if 'name' in configs_copy:
        del configs_copy['name']
    return dataset_class(**configs_copy, **kwargs)
# def make_dataset(name, configs, **kwargs):
#     """
#     Make a dataset instance.
    
#     Args:
#         name: dataset name
#         configs: dataset configuration
#         **kwargs: additional arguments for dataset constructor
        
#     Returns:
#         dataset: the dataset instance
#     """
#     dataset_class = get_dataset(name)
#     return dataset_class(**configs, **kwargs)


def make_dataset_split(name, configs, split, **kwargs):
    """
    Make a dataset instance for a specific split.
    
    Args:
        name: dataset name
        configs: dataset configuration
        split: dataset split (train, val, test)
        **kwargs: additional arguments for dataset constructor
        
    Returns:
        dataset: the dataset instance
    """
    dataset_configs = dict(configs)
    dataset_configs['split'] = split
    return make_dataset(name, dataset_configs, **kwargs)
