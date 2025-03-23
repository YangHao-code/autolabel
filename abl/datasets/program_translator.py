#! /usr/bin/env python3
# -*- coding: utf-8 -*-

import re
import json
import nltk
from nltk.tokenize import word_tokenize
from collections import defaultdict, Counter


class ProgramTranslator:
    """
    Translator for converting natural language questions to programs.
    This class translates text questions into executable programs.
    """
    
    def __init__(self, concept_mapping=None, label_mapping=None):
        self.concept_mapping = concept_mapping or {}
        self.label_mapping = label_mapping or {}
        
        # Initialize NLTK resources
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt', quiet=True)
    
    def translate(self, question):
        """
        Translate a natural language question to a program.
        
        Args:
            question: natural language question string
            
        Returns:
            program: list of program operations
        """
        # Tokenize and lowercase the question
        tokens = word_tokenize(question.lower())
        
        # Identify question type
        if any(word in tokens for word in ['label', 'classify', 'categorize']):
            return self._build_classification_program(question)
        elif any(word in tokens for word in ['explain', 'why', 'how', 'reason']):
            return self._build_explanation_program(question)
        elif any(word in tokens for word in ['concept', 'contains', 'has']):
            return self._build_concept_detection_program(question)
        elif any(word in tokens for word in ['relation', 'related', 'relationship', 'between']):
            return self._build_relation_program(question)
        else:
            # Default to classification
            return self._build_classification_program(question)
    
    def _find_concepts_in_text(self, text):
        """Find mentioned concepts in the text."""
        found_concepts = []
        
        # For each concept name, check if it's in the text
        for concept_name, concept_idx in self.concept_mapping.items():
            # Create a regex pattern to match whole words
            pattern = r'\b' + re.escape(concept_name.lower()) + r'\b'
            if re.search(pattern, text.lower()):
                found_concepts.append((concept_name, concept_idx))
        
        return found_concepts
    
    def _find_labels_in_text(self, text):
        """Find mentioned labels in the text."""
        found_labels = []
        
        # For each label name, check if it's in the text
        for label_name, label_idx in self.label_mapping.items():
            # Create a regex pattern to match whole words
            pattern = r'\b' + re.escape(label_name.lower()) + r'\b'
            if re.search(pattern, text.lower()):
                found_labels.append((label_name, label_idx))
        
        return found_labels
    
    def _build_classification_program(self, question):
        """Build a program for classification."""
        # Check if rule-based classification is requested
        rule_based = any(word in question.lower() for word in ['rule', 'symbolic', 'interpretable'])
        
        # Find mentioned concepts and labels
        concepts = self._find_concepts_in_text(question)
        labels = self._find_labels_in_text(question)
        
        if concepts:
            # If specific concepts are mentioned, create a filter first
            program = []
            # Check for negation
            negation = any(word in question.lower().split() for word in ['not', "doesn't", 'without'])
            if len(concepts) == 1:
                # Single concept filter
                concept_name, concept_idx = concepts[0]
                if negation:
                    program.append({
                        'type': 'filter_not_concept',
                        'args': [concept_idx]
                    })
                else:
                    program.append({
                        'type': 'filter_concept',
                        'args': [concept_idx]
                    })
            else:
                # Multiple concept filter
                # Check for 'or' conjunction
                if any(word in question.lower().split() for word in ['or', 'either']):
                    op = 'or'
                else:
                    op = 'and'
                program.append({
                    'type': 'filter_multi_concept',
                    'args': [[c[1] for c in concepts], op]
                })
            # Add prediction operation
            program.append({
                'type': 'predict_label',
                'args': [rule_based]
            })
            return program
        else:
            # Simple classification program
            return [{
                'type': 'predict_label',
                'args': [rule_based]
            }]
    
    def _build_explanation_program(self, question):
        """Build a program for explanation."""
        # Find mentioned labels
        labels = self._find_labels_in_text(question)
        
        if labels:
            # Explain specific label
            label_name, label_idx = labels[0]
            return [{
                'type': 'explain_prediction',
                'args': [label_idx]
            }]
        else:
            # Explain predicted label
            return [{
                'type': 'explain_prediction',
                'args': []
            }]
    
    def _build_concept_detection_program(self, question):
        """Build a program for concept detection."""
        # Find mentioned concepts
        concepts = self._find_concepts_in_text(question)
        if concepts:
            # Check for negation
            negation = any(word in question.lower().split() for word in ['not', "doesn't", 'without'])
            concept_name, concept_idx = concepts[0]
            if negation:
                return [{
                    'type': 'filter_not_concept',
                    'args': [concept_idx]
                }]
            else:
                return [{
                    'type': 'filter_concept',
                    'args': [concept_idx]
                }]
        else:
            # No specific concept mentioned, return all concept probabilities
            return [{
                'type': 'explain_prediction',
                'args': []
            }]
    
    def _build_relation_program(self, question):
        """Build a program for relation analysis."""
        # Find mentioned concepts
        concepts = self._find_concepts_in_text(question)
        
        if len(concepts) >= 2:
            # Get relation type
            if 'conditional' in question.lower() or 'given' in question.lower():
                relation_type = 'conditional'
            else:
                relation_type = 'cooccurrence'
            
            return [{
                'type': 'relate',
                'args': [concepts[0][1], concepts[1][1], relation_type]
            }]
        else:
            # Not enough concepts mentioned, default to explanation
            return [{
                'type': 'explain_prediction',
                'args': []
            }]
    
    def translate_batch(self, questions):
        """
        Translate a batch of questions to programs.
        
        Args:
            questions: list of question strings
            
        Returns:
            programs: list of programs
        """
        return [self.translate(q) for q in questions]
