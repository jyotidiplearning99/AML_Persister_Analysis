#!/usr/bin/env python3
"""
Extract complete hyperparameters from Feature Token Transformer model
Fixed version - handles string encoding properly
Author: Jyotidip Barman
"""

import h5py
import json
import numpy as np
from pathlib import Path

def extract_transformer_hyperparams():
    """Extract all hyperparameters from final_model.h5"""
    
    model_path = '/scratch/project_2010376/JDs_Project/AML_Persister_Analysis/results/models/final_model.h5'
    
    print("="*70)
    print("FEATURE TOKEN TRANSFORMER HYPERPARAMETERS EXTRACTION")
    print("="*70)
    
    with h5py.File(model_path, 'r') as f:
        # Get model config - handle both bytes and string
        model_config = f.attrs['model_config']
        
        # Check if it's bytes and decode, otherwise use as is
        if isinstance(model_config, bytes):
            cfg = json.loads(model_config.decode('utf-8'))
        else:
            cfg = json.loads(model_config)
        
        # Initialize hyperparameter collection
        hyperparams = {
            'attention': {},
            'ffn': {},
            'dropout': {},
            'architecture': {},
            'tokenization': {}
        }
        
        # Parse layers
        layers = cfg['config'].get('layers', [])
        
        # Track layer details
        attention_configs = []
        dense_configs = []
        dropout_configs = []
        normalization_configs = []
        
        print("\n" + "="*40)
        print("LAYER ANALYSIS")
        print("="*40)
        
        for i, layer in enumerate(layers):
            class_name = layer.get('class_name')
            config = layer.get('config', {})
            
            print(f"\nLayer {i}: {class_name}")
            
            if class_name == 'MultiHeadAttention':
                attention_config = {
                    'name': config.get('name'),
                    'num_heads': config.get('num_heads'),
                    'key_dim': config.get('key_dim'),
                    'value_dim': config.get('value_dim'),
                    'dropout': config.get('dropout'),
                    'use_bias': config.get('use_bias'),
                }
                attention_configs.append(attention_config)
                
                # Print important params
                print(f"  num_heads: {config.get('num_heads')}")
                print(f"  key_dim: {config.get('key_dim')}")
                print(f"  value_dim: {config.get('value_dim')}")
                print(f"  dropout: {config.get('dropout')}")
                
                # Check for shape information
                if 'query_shape' in config:
                    print(f"  query_shape: {config.get('query_shape')}")
                if 'key_shape' in config:
                    print(f"  key_shape: {config.get('key_shape')}")
                    
            elif class_name == 'Dense':
                dense_config = {
                    'name': config.get('name'),
                    'units': config.get('units'),
                    'activation': config.get('activation')
                }
                dense_configs.append(dense_config)
                print(f"  units: {config.get('units')}")
                print(f"  activation: {config.get('activation')}")
                
            elif class_name == 'Dropout':
                dropout_configs.append({
                    'name': config.get('name'),
                    'rate': config.get('rate')
                })
                print(f"  rate: {config.get('rate')}")
                
            elif class_name in ['LayerNormalization', 'BatchNormalization']:
                normalization_configs.append({
                    'name': config.get('name'),
                    'type': class_name,
                    'epsilon': config.get('epsilon')
                })
                print(f"  epsilon: {config.get('epsilon')}")
        
        # Extract key hyperparameters from attention layers
        if attention_configs:
            attn = attention_configs[0]  # Use first attention layer as reference
            hyperparams['attention'] = {
                'num_heads': attn['num_heads'],
                'key_dim': attn['key_dim'],
                'value_dim': attn['value_dim'],
                'd_model': attn['num_heads'] * attn['key_dim'],  # total dimension
                'dropout': attn['dropout'],
                'n_attention_blocks': len(attention_configs)
            }
        
        # Extract FFN dimensions
        ffn_widths = []
        for d in dense_configs:
            name = d['name'].lower()
            if 'ffn' in name or 'feed' in name or 'intermediate' in name:
                ffn_widths.append(d['units'])
        
        if not ffn_widths and dense_configs:
            # If no explicit FFN naming, use dense layers after attention
            ffn_widths = [d['units'] for d in dense_configs if d['units'] > 100]
        
        if ffn_widths:
            hyperparams['ffn']['width'] = ffn_widths[0]
            hyperparams['ffn']['expansion_factor'] = ffn_widths[0] / 64 if hyperparams['attention'].get('d_model') == 64 else None
        
        # Collect dropout rates
        dropout_rates = list(set([d['rate'] for d in dropout_configs if d['rate'] is not None]))
        hyperparams['dropout']['rates'] = sorted(dropout_rates)
        hyperparams['dropout']['main_rate'] = dropout_rates[0] if dropout_rates else None
        
        # Architecture summary
        hyperparams['architecture'] = {
            'n_attention_blocks': len(attention_configs),
            'n_dense_layers': len(dense_configs),
            'n_dropout_layers': len(dropout_configs),
            'n_normalization_layers': len(normalization_configs),
            'total_layers': len(layers)
        }
        
        # From the error output, we know the input/output shapes
        hyperparams['tokenization'] = {
            'n_tokens': 100,  # From query_shape: [None, 100, 64]
            'token_dim': 64,  # From query_shape
            'input_genes': 100,  # Actual input dimension
        }
        
        # Print formatted results for Methods section
        print("\n" + "="*70)
        print("HYPERPARAMETERS FOR METHODS SECTION")
        print("="*70)
        
        print("\n**Attention Module:**")
        print(f"  - Number of heads: {hyperparams['attention'].get('num_heads', 'N/A')}")
        print(f"  - Key dimension (d_k): {hyperparams['attention'].get('key_dim', 'N/A')}")
        print(f"  - Value dimension (d_v): {hyperparams['attention'].get('value_dim', 'N/A')}")
        print(f"  - Model dimension (d_model): {hyperparams['attention'].get('d_model', 'N/A')}")
        print(f"  - Attention dropout: {hyperparams['attention'].get('dropout', 'N/A')}")
        print(f"  - Number of attention blocks: {hyperparams['attention'].get('n_attention_blocks', 'N/A')}")
        
        print("\n**Feed-Forward Network:**")
        if hyperparams['ffn'].get('width'):
            print(f"  - FFN hidden dimension: {hyperparams['ffn']['width']}")
            if hyperparams['ffn'].get('expansion_factor'):
                print(f"  - Expansion factor: {hyperparams['ffn']['expansion_factor']:.1f}x")
        print(f"  - Total dense layers: {hyperparams['architecture']['n_dense_layers']}")
        
        print("\n**Regularization:**")
        print(f"  - Dropout rates: {hyperparams['dropout']['rates']}")
        print(f"  - Number of dropout layers: {hyperparams['architecture']['n_dropout_layers']}")
        
        print("\n**Tokenization:**")
        print(f"  - Number of tokens: {hyperparams['tokenization']['n_tokens']}")
        print(f"  - Token dimension: {hyperparams['tokenization']['token_dim']}")
        print(f"  - Input genes: {hyperparams['tokenization']['input_genes']}")
        
        # Generate formatted text for Methods section
        print("\n" + "="*70)
        print("FORMATTED TEXT FOR METHODS SECTION")
        print("="*70)
        
        methods_text = f"""
The Feature Token Transformer architecture employed multi-head self-attention 
with {hyperparams['attention'].get('num_heads', 4)} heads, key dimension (d_k) of {hyperparams['attention'].get('key_dim', 16)}, 
and value dimension (d_v) of {hyperparams['attention'].get('value_dim', 16)}, yielding a model dimension 
(d_model) of {hyperparams['attention'].get('d_model', 64)}. The architecture included {hyperparams['attention'].get('n_attention_blocks', 'multiple')} 
attention blocks. Dropout regularization was applied with p={hyperparams['attention'].get('dropout', 0.2)} 
in attention layers and p={hyperparams['dropout'].get('main_rate', 0.3)} in feed-forward layers. 
The input consisted of {hyperparams['tokenization']['n_tokens']} tokens with dimension 
{hyperparams['tokenization']['token_dim']}, processing {hyperparams['tokenization']['input_genes']} 
input features (genes).
"""
        print(methods_text)
        
        return hyperparams

if __name__ == "__main__":
    params = extract_transformer_hyperparams()
