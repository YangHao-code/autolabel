#! /usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import os.path as osp
import json
import jsonlines
import torch
from torch.utils.data import Dataset
from collections import OrderedDict


class NSCLDataset(Dataset):
    """Base class for all NSCL datasets."""
    
    def __init__(self):
        super().__init__()
        
    def __getitem__(self, index):
        raise NotImplementedError()
        
    def __len__(self):
        raise NotImplementedError()


class TextLabelingDatasetBase(NSCLDataset):
    """Base dataset for text labeling tasks."""
    def __init__(self, root_dir, split, max_length=128):
        super().__init__()
        
        self.root_dir = root_dir
        self.split = split
        self.max_length = max_length
        self.texts = []
        self.labels = []
        self.concept_annotations = []
        self.load_mappings()
    
    def load_mappings(self):
        """Load label and concept mappings."""
        label_path = osp.join(self.root_dir, 'meta', 'labels.json')
        with open(label_path, 'r') as f:
            self.label_mapping = json.load(f)
        self.inv_label_mapping = {v: k for k, v in self.label_mapping.items()}
        concept_path = osp.join(self.root_dir, 'meta', 'concepts.json')
        if osp.exists(concept_path):
            with open(concept_path, 'r') as f:
                self.concept_mapping = json.load(f)
            self.inv_concept_mapping = {v: k for k, v in self.concept_mapping.items()}
            self.nr_concepts = len(self.concept_mapping)
        else:
            self.concept_mapping = None
            self.inv_concept_mapping = None
            self.nr_concepts = 0
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, index):
        """Get a sample from the dataset."""
        text = self.texts[index]
        label_idx = self.labels[index]
        label_onehot = torch.zeros(len(self.label_mapping))
        label_onehot[label_idx] = 1.0
        
        # Get concept annotations if available
        if len(self.concept_annotations) > 0:
            concept_annot = self.concept_annotations[index]
            concept_vec = torch.zeros(self.nr_concepts)
            for concept_idx in concept_annot:
                concept_vec[concept_idx] = 1.0
        else:
            concept_vec = None
        
        sample = {
            'text': text,
            'label': label_idx,
            'label_onehot': label_onehot
        }
        
        if concept_vec is not None:
            sample['concept_annotations'] = concept_vec
        
        return sample

class TextLabelingDataset(TextLabelingDatasetBase):
    """Dataset for text classification with conceptual annotations."""
    def __init__(self, root_dir, split, max_length=128, concept_threshold=0.5):
        super().__init__(root_dir, split, max_length)
        self.concept_threshold = concept_threshold
        self.load_data()
    
    def load_data(self):
        file_path = osp.join(self.root_dir, f'{self.split}.jsonl')
        if not osp.exists(file_path):
            raise FileNotFoundError(f"Dataset file not found: {file_path}")
        # Load data from JSONL file
        texts = []
        labels = []
        concept_annotations = []
        
        with jsonlines.open(file_path) as reader:
            for item in reader:
                texts.append(item['text'])
                if 'label' in item:
                    label_idx = self.label_mapping[item['label']]
                    labels.append(label_idx)
                else:
                    labels.append(0)
                if 'concepts' in item and self.concept_mapping is not None:
                    # Convert concept names to indices
                    concept_indices = []
                    for concept in item['concepts']:
                        if concept in self.concept_mapping:
                            concept_indices.append(self.concept_mapping[concept]) 
                    concept_annotations.append(concept_indices)
        
        self.texts = texts
        self.labels = labels
        
        if len(concept_annotations) == len(texts):
            self.concept_annotations = concept_annotations
        else:
            # If concept annotations are not available for all samples,
            # use empty list to indicate no annotations
            self.concept_annotations = []
    
    def get_concept_names(self):
        if self.concept_mapping is None:
            return []
        concepts = sorted(self.concept_mapping.items(), key=lambda x: x[1])
        return [name for name, _ in concepts]
    
    def get_label_names(self):
        labels = sorted(self.label_mapping.items(), key=lambda x: x[1])
        return [name for name, _ in labels]
    
    def get_stats(self):
        """Get dataset statistics."""
        stats = {
            'num_samples': len(self.texts),
            'num_labels': len(self.label_mapping),
            'num_concepts': self.nr_concepts if self.concept_mapping else 0,
            'has_concept_annotations': len(self.concept_annotations) > 0
        }
        # Count samples per label
        label_counts = {}
        for label_idx in self.labels:
            label_name = self.inv_label_mapping[label_idx]
            label_counts[label_name] = label_counts.get(label_name, 0) + 1
        
        stats['label_distribution'] = label_counts
        
        return stats