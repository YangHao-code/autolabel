#! /usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import torch.nn.functional as F


class TextProgramExecutor:
    """
    Program executor for text-based tasks.
    This class executes symbolic programs on text representations.
    """
    
    def __init__(self, concept_embeddings=None, concept_names=None):
        self.concept_embeddings = concept_embeddings
        self.concept_names = concept_names
        self.operations = self._get_operations()
        # print("DEBUG: Program executor initialized.")
    
    def _get_operations(self):
        """Define all available operations for the program executor."""
        return {
            'filter_concept': self.filter_concept,
            'filter_not_concept': self.filter_not_concept,
            'filter_multi_concept': self.filter_multi_concept,
            'relate': self.relate,
            'predict_label': self.predict_label,
            'explain_prediction': self.explain_prediction
        }
    
    def filter_concept(self, inputs, concept_idx):
        """
        Filter for texts that contain a specific concept.
        
        Args:
            inputs: dict containing 'concept_probs'
            concept_idx: index of the concept to filter
            
        Returns:
            filtered concept probabilities
        """
        concept_probs = inputs['concept_probs']  # [batch_size, num_concepts]
        concept_prob = concept_probs[:, concept_idx].unsqueeze(1)  # [batch_size, 1]
        return concept_prob
    
    def filter_not_concept(self, inputs, concept_idx):
        """
        Filter for texts that do not contain a specific concept.
        
        Args:
            inputs: dict containing 'concept_probs'
            concept_idx: index of the concept to filter out
            
        Returns:
            filtered concept probabilities
        """
        concept_probs = inputs['concept_probs']
        concept_prob = 1 - concept_probs[:, concept_idx].unsqueeze(1) 
        return concept_prob
    
    def filter_multi_concept(self, inputs, concept_indices, operator='and'):
        """
        Filter for texts that contain multiple concepts.
        
        Args:
            inputs: dict containing 'concept_probs'
            concept_indices: list of concept indices
            operator: 'and' or 'or'
            
        Returns:
            filtered concept probabilities
        """
        concept_probs = inputs['concept_probs']  # [batch_size, num_concepts]
        selected_probs = concept_probs[:, concept_indices]  # [batch_size, len(concept_indices)]
        if operator.lower() == 'and':
            # Probability that all concepts exist (using product)
            combined_prob = selected_probs.prod(dim=1, keepdim=True)
        else:  # 'or'
            # Probability that at least one concept exists (using probabilistic OR)
            combined_prob = 1 - (1 - selected_probs).prod(dim=1, keepdim=True)
        
        return combined_prob
    
    def relate(self, inputs, concept_idx1, concept_idx2, relation_type='cooccurrence'):
        """
        Compute relationship between two concepts.
        
        Args:
            inputs: dict containing 'concept_probs'
            concept_idx1: index of first concept
            concept_idx2: index of second concept
            relation_type: type of relation to compute
            
        Returns:
            relation score
        """
        concept_probs = inputs['concept_probs']  # [batch_size, num_concepts]
        concept_prob1 = concept_probs[:, concept_idx1]  # [batch_size]
        concept_prob2 = concept_probs[:, concept_idx2]  # [batch_size]
        
        if relation_type == 'cooccurrence':
            # Probability of both concepts co-occurring
            relation_score = concept_prob1 * concept_prob2
        elif relation_type == 'conditional':
            # Conditional probability P(concept2 | concept1)
            relation_score = concept_prob2 / (concept_prob1 + 1e-8)
        else:
            raise ValueError(f"Unknown relation type: {relation_type}")
        
        return relation_score.unsqueeze(1)
    
    def predict_label(self, inputs, rule_based=False):
        """
        Predict labels based on concept probabilities.
        
        Args:
            inputs: dict containing 'concept_probs' and 'rule_activations'
            rule_based: whether to use rule-based prediction
            
        Returns:
            predicted label probabilities
        """
        if rule_based and 'rule_activations' in inputs:
            # Use rule activations for prediction
            rule_acts = inputs['rule_activations']  # [batch_size, num_rules]
            label_probs = torch.sigmoid(rule_acts)
        elif 'label_logits' in inputs:
            # Use neural network prediction
            label_logits = inputs['label_logits']  # [batch_size, num_labels]
            label_probs = torch.softmax(label_logits, dim=1)
        else:
            # If neither is available, return zeros
            batch_size = inputs['concept_probs'].size(0)
            num_labels = inputs.get('num_labels', 1)
            label_probs = torch.zeros(batch_size, num_labels)
        
        return label_probs
    
    def explain_prediction(self, inputs, label_idx=None):
        """
        Generate explanation for a prediction based on concepts.
        
        Args:
            inputs: dict containing model outputs
            label_idx: index of the label to explain
            
        Returns:
            dict containing explanation elements
        """
        concept_probs = inputs['concept_probs']  # [batch_size, num_concepts]
        
        # If label_idx is not provided, use the predicted label
        if label_idx is None and 'label_logits' in inputs:
            label_idx = inputs['label_logits'].argmax(dim=1)
        
        # Find top concepts for each sample
        top_concepts_idx = torch.topk(concept_probs, k=min(5, concept_probs.size(1)), dim=1).indices
        
        # Return explanation elements
        explanation = {
            'top_concepts_idx': top_concepts_idx,
            'concept_probs': concept_probs
        }
        
        if 'rule_activations' in inputs:
            explanation['rule_activations'] = inputs['rule_activations']
        
        return explanation
    
    def execute_program(self, program, inputs):
        """
        Execute a program on the given inputs.
        
        Args:
            program: list of operations to execute
            inputs: dict containing model outputs
            
        Returns:
            program execution result
        """
        stack = []
        
        for op in program:
            func_name = op['type']
            if func_name not in self.operations:
                raise ValueError(f"Unknown operation: {func_name}")
            
            func = self.operations[func_name]
            args = op.get('args', [])
            if func_name == 'predict_label' or func_name == 'explain_prediction':
                # These operations directly use inputs
                result = func(inputs, *args)
            else:
                if not stack:
                    result = func(inputs, *args)
                else:
                    prev_result = stack.pop()
                    op_inputs = {**inputs, 'prev_result': prev_result}
                    result = func(op_inputs, *args)
            stack.append(result)
        return stack[-1] if stack else None