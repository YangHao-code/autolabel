#! /usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import numpy as np
import torch


class HTMLTableVisualizer:
    """HTML Table Visualizer for NSCL results."""
    
    def __init__(self, title='Visualization'):
        self.title = title
        self.sections = []
    
    def _repr_html_(self):
        """Generate HTML representation for IPython/Jupyter."""
        return self.html()
    
    def add_section(self, title, content):
        """Add a section to the visualizer."""
        self.sections.append({
            'title': title,
            'content': content
        })
    
    def add_text_sample(self, text, label=None, concept_probs=None, concept_names=None, rule_activations=None):
        """
        Add a text sample visualization.
        
        Args:
            text: text string
            label: predicted label
            concept_probs: concept probabilities
            concept_names: concept names
            rule_activations: rule activation scores
        """
        section_html = f'<div class="sample">'
        section_html += f'<div class="text">{text}</div>'
        
        if label is not None:
            section_html += f'<div class="prediction">Predicted Label: <strong>{label}</strong></div>'
        
        if concept_probs is not None:
            section_html += '<div class="concepts"><h4>Detected Concepts:</h4><table>'
            
            # Convert to numpy array if it's a tensor
            if isinstance(concept_probs, torch.Tensor):
                concept_probs = concept_probs.cpu().numpy()
            
            # Add headers
            section_html += '<tr><th>Concept</th><th>Probability</th></tr>'
            
            # Sort concepts by probability
            indices = np.argsort(-concept_probs)
            
            for idx in indices:
                prob = concept_probs[idx]
                name = concept_names[idx] if concept_names and idx < len(concept_names) else f'Concept-{idx}'
                
                # Skip concepts with very low probability
                if prob < 0.05:
                    continue
                
                # Color coding based on probability
                color = self._get_color_for_prob(prob)
                
                section_html += f'<tr>'
                section_html += f'<td>{name}</td>'
                section_html += f'<td style="background-color: {color}">{prob:.3f}</td>'
                section_html += f'</tr>'
            
            section_html += '</table></div>'
        
        if rule_activations is not None:
            section_html += '<div class="rules"><h4>Rule Activations:</h4><table>'
            
            # Convert to numpy array if it's a tensor
            if isinstance(rule_activations, torch.Tensor):
                rule_activations = rule_activations.cpu().numpy()
            
            # Add headers
            section_html += '<tr><th>Rule</th><th>Activation</th></tr>'
            
            # Sort rules by activation
            indices = np.argsort(-rule_activations)
            
            for idx in indices:
                activation = rule_activations[idx]
                
                # Skip rules with very low activation
                if activation < 0.1:
                    continue
                
                # Color coding based on activation
                color = self._get_color_for_prob(activation)
                
                section_html += f'<tr>'
                section_html += f'<td>Rule-{idx}</td>'
                section_html += f'<td style="background-color: {color}">{activation:.3f}</td>'
                section_html += f'</tr>'
            
            section_html += '</table></div>'
        
        section_html += '</div>'
        
        self.add_section('Text Sample', section_html)
    
    def add_concept_analysis(self, concept_names, concept_embeddings):
        """
        Add concept analysis visualization.
        
        Args:
            concept_names: list of concept names
            concept_embeddings: concept embedding matrix [num_concepts, embedding_dim]
        """
        if concept_embeddings is None:
            return
        
        # Convert to numpy array if it's a tensor
        if isinstance(concept_embeddings, torch.Tensor):
            concept_embeddings = concept_embeddings.cpu().numpy()
        
        # Compute pairwise similarities
        norm = np.linalg.norm(concept_embeddings, axis=1, keepdims=True)
        normalized_embeddings = concept_embeddings / (norm + 1e-8)
        similarities = np.matmul(normalized_embeddings, normalized_embeddings.T)
        
        # Generate HTML
        section_html = '<div class="concept-analysis">'
        section_html += '<h4>Concept Similarities:</h4>'
        section_html += '<table>'
        
        # Add header row
        section_html += '<tr><th></th>'
        for name in concept_names:
            section_html += f'<th>{name}</th>'
        section_html += '</tr>'
        
        # Add data rows
        for i, name in enumerate(concept_names):
            section_html += f'<tr><th>{name}</th>'
            for j in range(len(concept_names)):
                similarity = similarities[i, j]
                color = self._get_color_for_prob(similarity)
                section_html += f'<td style="background-color: {color}">{similarity:.2f}</td>'
            section_html += '</tr>'
        
        section_html += '</table></div>'
        
        self.add_section('Concept Analysis', section_html)
    
    def add_rule_analysis(self, rule_weights, concept_names):
        """
        Add rule analysis visualization.
        
        Args:
            rule_weights: rule weight matrix [num_rules, num_concepts]
            concept_names: list of concept names
        """
        if rule_weights is None:
            return
        
        # Convert to numpy array if it's a tensor
        if isinstance(rule_weights, torch.Tensor):
            rule_weights = rule_weights.cpu().numpy()
        
        # Generate HTML
        section_html = '<div class="rule-analysis">'
        section_html += '<h4>Rule Analysis:</h4>'
        section_html += '<table>'
        
        # Add header row
        section_html += '<tr><th>Rule</th>'
        for name in concept_names:
            section_html += f'<th>{name}</th>'
        section_html += '</tr>'
        
        # Add data rows
        for i in range(rule_weights.shape[0]):
            section_html += f'<tr><th>Rule-{i}</th>'
            for j in range(rule_weights.shape[1]):
                weight = rule_weights[i, j]
                color = self._get_color_for_weight(weight)
                section_html += f'<td style="background-color: {color}">{weight:.2f}</td>'
            section_html += '</tr>'
        
        section_html += '</table></div>'
        
        self.add_section('Rule Analysis', section_html)
    
    def _get_color_for_prob(self, prob):
        """Get color for probability value."""
        r = int(255 * (1 - prob))
        g = int(255 * prob)
        b = 0
        return f'rgba({r}, {g}, {b}, 0.3)'
    
    def _get_color_for_weight(self, weight):
        """Get color for weight value."""
        # Blue for negative, red for positive
        if weight >= 0:
            r = int(255 * min(1, weight))
            g = 0
            b = 0
            return f'rgba({r}, {g}, {b}, 0.3)'
        else:
            r = 0
            g = 0
            b = int(255 * min(1, -weight))
            return f'rgba({r}, {g}, {b}, 0.3)'
    
    def html(self):
        """Generate HTML representation."""
        html = f'''
        <html>
        <head>
            <title>{self.title}</title>
            <style>
                .visualization {{
                    font-family: Arial, sans-serif;
                    margin: 20px;
                    padding: 20px;
                    border: 1px solid #ddd;
                    border-radius: 5px;
                }}
                .section {{
                    margin-bottom: 30px;
                }}
                .section-title {{
                    font-size: 18px;
                    font-weight: bold;
                    margin-bottom: 10px;
                    padding-bottom: 5px;
                    border-bottom: 1px solid #ddd;
                }}
                table {{
                    border-collapse: collapse;
                    width: 100%;
                }}
                th, td {{
                    border: 1px solid #ddd;
                    padding: 8px;
                    text-align: left;
                }}
                th {{
                    background-color: #f2f2f2;
                }}
                .sample {{
                    padding: 10px;
                    border: 1px solid #ddd;
                    border-radius: 5px;
                    margin-bottom: 10px;
                }}
                .text {{
                    font-size: 16px;
                    margin-bottom: 10px;
                    padding: 10px;
                    background-color: #f9f9f9;
                    border-radius: 5px;
                }}
                .prediction {{
                    margin-bottom: 10px;
                    font-size: 14px;
                }}
                .concepts, .rules {{
                    margin-top: 10px;
                }}
            </style>
        </head>
        <body>
            <div class="visualization">
                <h2>{self.title}</h2>
        '''
        
        for section in self.sections:
            html += f'''
                <div class="section">
                    <div class="section-title">{section['title']}</div>
                    <div class="section-content">{section['content']}</div>
                </div>
            '''
        
        html += '''
            </div>
        </body>
        </html>
        '''
        
        return html
    
    def save(self, filename):
        """Save visualization to HTML file."""
        with open(filename, 'w') as f:
            f.write(self.html())
        print(f'Visualization saved to {filename}')