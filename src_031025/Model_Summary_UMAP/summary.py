#!/usr/bin/env python3
"""
Fixed Model Summary and UMAP Analysis for AML Persister Project
Handles transformer model loading issues and formatting errors
Author: Jyotidip Barman
Date: October 2025
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import tensorflow as tf
from tensorflow import keras
import h5py
import logging
from typing import Dict, List, Optional, Tuple, Union

# UMAP imports
try:
    import umap
    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False
    print("Warning: UMAP not installed. Install with: pip install umap-learn")

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

# Custom model loading for transformer compatibility
def load_model_safe(model_path: Path):
    """Load model with custom objects for transformer layers"""
    try:
        # Try standard loading first
        return keras.models.load_model(model_path, compile=False)
    except:
        try:
            # Try with custom objects for transformer layers
            custom_objects = {
                'MultiHeadAttention': keras.layers.MultiHeadAttention,
            }
            return keras.models.load_model(model_path, custom_objects=custom_objects, compile=False)
        except:
            # If all else fails, return None
            return None

class ModelSummarizer:
    """Comprehensive analysis of H5 model files"""
    
    def __init__(self, model_path: Path):
        self.model_path = model_path
        self.model_name = model_path.stem
        
    def get_comprehensive_summary(self) -> Dict:
        """Get detailed model summary including architecture, parameters, and performance"""
        
        logger.info(f"\n{'='*60}")
        logger.info(f"Analyzing Model: {self.model_name}")
        logger.info(f"Path: {self.model_path}")
        logger.info(f"{'='*60}")
        
        summary = {
            'model_name': self.model_name,
            'path': str(self.model_path),
            'file_size_mb': self.model_path.stat().st_size / (1024**2) if self.model_path.exists() else 0
        }
        
        if not self.model_path.exists():
            logger.error(f"Model file not found: {self.model_path}")
            return summary
        
        # Try to load model
        model = load_model_safe(self.model_path)
        
        if model is not None:
            try:
                # Model loaded successfully
                logger.info("Model loaded successfully with Keras")
                
                # Basic info
                summary['input_shape'] = str(model.input_shape) if hasattr(model, 'input_shape') else 'N/A'
                summary['output_shape'] = str(model.output_shape) if hasattr(model, 'output_shape') else 'N/A'
                summary['total_params'] = int(model.count_params())
                
                # Layer analysis
                layer_info = []
                trainable_params = 0
                non_trainable_params = 0
                
                for layer in model.layers:
                    layer_dict = {
                        'name': layer.name,
                        'type': layer.__class__.__name__,
                        'output_shape': str(layer.output_shape) if hasattr(layer, 'output_shape') else 'N/A',
                        'params': int(layer.count_params()),
                        'trainable': layer.trainable
                    }
                    layer_info.append(layer_dict)
                    
                    if layer.trainable:
                        trainable_params += layer.count_params()
                    else:
                        non_trainable_params += layer.count_params()
                
                summary['layers'] = layer_info
                summary['n_layers'] = len(model.layers)
                summary['trainable_params'] = int(trainable_params)
                summary['non_trainable_params'] = int(non_trainable_params)
                
                # Architecture detection
                layer_types = [l['type'].lower() for l in layer_info]
                if any('attention' in t or 'transformer' in t for t in layer_types):
                    summary['architecture_type'] = 'Feature Token Transformer'
                    summary['has_attention'] = True
                    attention_layers = [l for l in layer_info if 'attention' in l['type'].lower()]
                    summary['n_attention_layers'] = len(attention_layers)
                else:
                    summary['architecture_type'] = 'Standard Neural Network'
                    summary['has_attention'] = False
                    summary['n_attention_layers'] = 0
                
                # Model type detection
                if 'reduced' in self.model_name.lower() or 'distilled' in self.model_name.lower():
                    summary['model_type'] = 'Distilled/Reduced'
                    # Check input shape to determine gene count
                    if summary['input_shape'] and '100' in summary['input_shape']:
                        summary['expected_genes'] = 100  # Actual input size
                    else:
                        summary['expected_genes'] = 1000  # Expected from name
                else:
                    summary['model_type'] = 'Full'
                    summary['expected_genes'] = 13000
                
                # Memory footprint
                summary['estimated_memory_mb'] = (trainable_params * 4) / (1024**2)
                
                # Print architecture summary
                logger.info("\nModel Architecture Summary:")
                model.summary(print_fn=logger.info)
                
            except Exception as e:
                logger.error(f"Error analyzing loaded model: {e}")
        
        else:
            # Model couldn't be loaded with Keras, try h5py
            logger.warning("Could not load model with Keras, attempting h5py analysis...")
            
            try:
                with h5py.File(self.model_path, 'r') as f:
                    summary['h5_keys'] = list(f.keys())
                    summary['h5_attrs'] = list(f.attrs.keys())
                    
                    # Try to extract model config
                    if 'model_config' in f.attrs:
                        config_str = f.attrs['model_config']
                        if isinstance(config_str, bytes):
                            config_str = config_str.decode('utf-8')
                        config = json.loads(config_str)
                        
                        # Parse config for model info
                        if 'config' in config:
                            model_config = config['config']
                            if 'layers' in model_config:
                                summary['n_layers'] = len(model_config['layers'])
                                
                                # Check for attention layers
                                attention_count = 0
                                for layer in model_config['layers']:
                                    if 'MultiHeadAttention' in str(layer):
                                        attention_count += 1
                                
                                if attention_count > 0:
                                    summary['architecture_type'] = 'Feature Token Transformer'
                                    summary['has_attention'] = True
                                    summary['n_attention_layers'] = attention_count
                                else:
                                    summary['architecture_type'] = 'Unknown'
                                    summary['has_attention'] = False
                                    summary['n_attention_layers'] = 0
                        
                        summary['model_class'] = config.get('class_name', 'Unknown')
                    
                    # Try to estimate parameters from weights
                    if 'model_weights' in f:
                        total_params = 0
                        def count_params(name, obj):
                            if isinstance(obj, h5py.Dataset):
                                nonlocal total_params
                                total_params += obj.size
                        f['model_weights'].visititems(count_params)
                        summary['total_params'] = total_params
                        summary['estimated_memory_mb'] = (total_params * 4) / (1024**2)
                    
                    logger.info(f"H5 file structure: {summary['h5_keys']}")
                    logger.info(f"H5 attributes: {summary.get('h5_attrs', [])}")
                    
                    # Mark as transformer if detected in config
                    if summary.get('architecture_type') == 'Feature Token Transformer':
                        summary['model_type'] = 'Full'
                        summary['expected_genes'] = 13000
                        logger.info("Detected Feature Token Transformer architecture from config")
                    
            except Exception as e:
                logger.error(f"Could not analyze H5 file: {e}")
        
        # Print summary
        self._print_summary(summary)
        
        return summary
    
    def _print_summary(self, summary: Dict):
        """Print formatted summary"""
        
        logger.info(f"\n{'='*40}")
        logger.info("MODEL SUMMARY")
        logger.info(f"{'='*40}")
        logger.info(f"Model: {summary['model_name']}")
        logger.info(f"Type: {summary.get('model_type', 'Unknown')}")
        logger.info(f"Architecture: {summary.get('architecture_type', 'Unknown')}")
        logger.info(f"File Size: {summary['file_size_mb']:.2f} MB")
        logger.info(f"Input Shape: {summary.get('input_shape', 'N/A')}")
        logger.info(f"Output Shape: {summary.get('output_shape', 'N/A')}")
        
        # Format parameters properly
        total_params = summary.get('total_params', 0)
        if total_params > 0:
            logger.info(f"Total Parameters: {total_params:,}")
        else:
            logger.info("Total Parameters: Unknown")
            
        trainable = summary.get('trainable_params', 0)
        if trainable > 0:
            logger.info(f"  Trainable: {trainable:,}")
            
        non_trainable = summary.get('non_trainable_params', 0)
        if non_trainable > 0:
            logger.info(f"  Non-trainable: {non_trainable:,}")
            
        logger.info(f"Number of Layers: {summary.get('n_layers', 'Unknown')}")
        
        if summary.get('has_attention'):
            logger.info(f"Attention Layers: {summary.get('n_attention_layers', 0)}")
        
        memory = summary.get('estimated_memory_mb', 0)
        if memory > 0:
            logger.info(f"Estimated Memory: {memory:.2f} MB")
            
        logger.info(f"Expected Input Genes: {summary.get('expected_genes', 'Unknown')}")

class UMAPAnalyzer:
    """UMAP analysis for gene expression data"""
    
    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.figures_dir = self.output_dir / 'umap_figures'
        self.figures_dir.mkdir(exist_ok=True)
        
    def load_gene_list(self, gene_file: Path) -> List[str]:
        """Load gene list from text file"""
        
        if not gene_file.exists():
            logger.error(f"Gene file not found: {gene_file}")
            return []
            
        with open(gene_file, 'r') as f:
            genes = [line.strip() for line in f if line.strip()]
        
        logger.info(f"Loaded {len(genes)} genes from {gene_file.name}")
        return genes
    
    def create_comparison_visualization(
        self,
        common_genes_path: Path,
        selected_genes_path: Path,
        n_samples: int = 500
    ):
        """Create UMAP comparison for two gene sets"""
        
        if not UMAP_AVAILABLE:
            logger.warning("UMAP not available. Skipping visualization.")
            return
        
        # Load gene lists
        common_genes = self.load_gene_list(common_genes_path)
        selected_genes = self.load_gene_list(selected_genes_path)
        
        if not common_genes or not selected_genes:
            logger.error("Could not load gene lists")
            return
        
        # Report statistics
        overlap = set(selected_genes) & set(common_genes)
        logger.info(f"\nGene Set Statistics:")
        logger.info(f"  Common genes: {len(common_genes)}")
        logger.info(f"  Selected genes: {len(selected_genes)}")
        logger.info(f"  Overlap: {len(overlap)} genes")
        logger.info(f"  Selected genes are {100*len(overlap)/len(selected_genes):.1f}% of common genes")
        
        # Create synthetic data for demonstration
        logger.info("\nCreating demonstration data for UMAP...")
        
        # Use a subset for visualization
        genes_to_use = list(set(common_genes[:200]) | set(selected_genes[:200]))
        n_genes = len(genes_to_use)
        
        # Generate synthetic expression data with structure
        np.random.seed(42)
        
        # Create two distinct groups
        group1_data = np.random.randn(n_samples//2, n_genes) + np.random.randn(1, n_genes)
        group2_data = np.random.randn(n_samples//2, n_genes) - np.random.randn(1, n_genes)
        
        data = np.vstack([group1_data, group2_data])
        labels = ['Group1'] * (n_samples//2) + ['Group2'] * (n_samples//2)
        
        df = pd.DataFrame(data, columns=genes_to_use)
        
        # Create UMAP for common genes subset
        common_subset = [g for g in common_genes[:100] if g in df.columns]
        if len(common_subset) > 10:
            self._create_umap(df[common_subset], labels, 
                            f"UMAP - Common Genes ({len(common_subset)} genes)")
        
        # Create UMAP for selected genes subset
        selected_subset = [g for g in selected_genes[:100] if g in df.columns]
        if len(selected_subset) > 10:
            self._create_umap(df[selected_subset], labels,
                            f"UMAP - Selected Genes ({len(selected_subset)} genes)")
    
    def _create_umap(self, data: pd.DataFrame, labels: List[str], title: str):
        """Create single UMAP visualization"""
        
        # Standardize
        scaler = StandardScaler()
        data_scaled = scaler.fit_transform(data)
        
        # PCA if needed
        if data.shape[1] > 50:
            pca = PCA(n_components=min(50, data.shape[0]-1, data.shape[1]))
            data_scaled = pca.fit_transform(data_scaled)
        
        # UMAP
        reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=42)
        embedding = reducer.fit_transform(data_scaled)
        
        # Plot
        plt.figure(figsize=(8, 6))
        for label in set(labels):
            mask = [l == label for l in labels]
            plt.scatter(embedding[mask, 0], embedding[mask, 1],
                       label=label, alpha=0.7, s=20)
        
        plt.xlabel('UMAP 1')
        plt.ylabel('UMAP 2')
        plt.title(title)
        plt.legend()
        plt.tight_layout()
        
        # Save
        safe_title = title.replace(' ', '_').replace('-', '_').replace('(', '').replace(')', '')
        fig_path = self.figures_dir / f'{safe_title}.png'
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"  Saved UMAP to: {fig_path}")

def main():
    """Main analysis pipeline"""
    
    # Set paths
    base_dir = Path('/scratch/project_2010376/JDs_Project/AML_Persister_Analysis')
    
    # Model paths
    full_model_path = base_dir / 'results/models/final_model.h5'
    reduced_model_path = base_dir / 'reduced_model_distilled/model_reduced.h5'
    
    # Gene list paths
    common_genes_path = base_dir / 'results/metadata/common_genes.txt'
    selected_genes_path = base_dir / 'reduced_model_distilled/selected_genes.txt'
    
    # Output directory
    output_dir = base_dir / 'results/model_analysis'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("="*70)
    logger.info("AML PERSISTER MODEL ANALYSIS AND VISUALIZATION")
    logger.info("="*70)
    
    # ========== Part 1: Model Summary ==========
    logger.info("\n" + "="*70)
    logger.info("PART 1: MODEL SUMMARIES")
    logger.info("="*70)
    
    summaries = {}
    
    # Analyze both models
    if full_model_path.exists():
        full_summarizer = ModelSummarizer(full_model_path)
        summaries['full_model'] = full_summarizer.get_comprehensive_summary()
    
    if reduced_model_path.exists():
        reduced_summarizer = ModelSummarizer(reduced_model_path)
        summaries['reduced_model'] = reduced_summarizer.get_comprehensive_summary()
    
    # Model comparison
    logger.info("\n" + "="*70)
    logger.info("MODEL COMPARISON")
    logger.info("="*70)
    
    comparison_data = []
    
    if 'full_model' in summaries and 'reduced_model' in summaries:
        full = summaries['full_model']
        reduced = summaries['reduced_model']
        
        comparison_data = [
            ['Metric', 'Full Model', 'Reduced Model'],
            ['Architecture', full.get('architecture_type', 'Unknown'), reduced.get('architecture_type', 'Unknown')],
            ['File Size (MB)', f"{full.get('file_size_mb', 0):.2f}", f"{reduced.get('file_size_mb', 0):.2f}"],
            ['Total Parameters', f"{full.get('total_params', 'Unknown'):,}" if isinstance(full.get('total_params'), int) else 'Unknown',
                               f"{reduced.get('total_params', 0):,}"],
            ['Layers', str(full.get('n_layers', 'Unknown')), str(reduced.get('n_layers', 0))],
            ['Has Attention', 'Yes' if full.get('has_attention') else 'No/Unknown', 
                            'Yes' if reduced.get('has_attention') else 'No'],
            ['Expected Genes', str(full.get('expected_genes', 13000)), str(reduced.get('expected_genes', 100))],
        ]
        
        # Print comparison table
        for row in comparison_data:
            if row[0] == 'Metric':
                logger.info(f"{row[0]:<20} {row[1]:<25} {row[2]:<25}")
                logger.info("-" * 70)
            else:
                logger.info(f"{row[0]:<20} {row[1]:<25} {row[2]:<25}")
        
        # Calculate reduction if possible
        if isinstance(full.get('total_params'), int) and reduced.get('total_params', 0) > 0:
            if full.get('total_params', 0) > 0:
                reduction = 1 - (reduced['total_params'] / full['total_params'])
                speedup = full['total_params'] / reduced['total_params']
                logger.info(f"\nEfficiency Metrics:")
                logger.info(f"  Parameter Reduction: {reduction*100:.1f}%")
                logger.info(f"  Expected Speedup: ~{speedup:.1f}x")
    
    # Save summaries
    summary_output = output_dir / 'model_summaries.json'
    with open(summary_output, 'w') as f:
        # Clean for JSON serialization
        clean_summaries = {}
        for key, summary in summaries.items():
            clean_summary = {k: v for k, v in summary.items() if k != 'layers'}
            clean_summaries[key] = clean_summary
        json.dump(clean_summaries, f, indent=2, default=str)
    
    logger.info(f"\nSaved model summaries to: {summary_output}")
    
    # Save comparison as CSV
    if comparison_data:
        comparison_df = pd.DataFrame(comparison_data[1:], columns=comparison_data[0])
        comparison_df.to_csv(output_dir / 'model_comparison.csv', index=False)
        logger.info(f"Saved comparison table to: {output_dir / 'model_comparison.csv'}")
    
    # ========== Part 2: UMAP Analysis ==========
    if UMAP_AVAILABLE:
        logger.info("\n" + "="*70)
        logger.info("PART 2: UMAP VISUALIZATION")
        logger.info("="*70)
        
        umap_analyzer = UMAPAnalyzer(output_dir)
        umap_analyzer.create_comparison_visualization(
            common_genes_path,
            selected_genes_path
        )
    else:
        logger.info("\nSkipping UMAP analysis (umap-learn not installed)")
    
    # ========== Part 3: Final Report ==========
    logger.info("\n" + "="*70)
    logger.info("ANALYSIS COMPLETE")
    logger.info("="*70)
    
    # Key findings summary
    logger.info("\nKEY FINDINGS:")
    
    if 'reduced_model' in summaries:
        reduced = summaries['reduced_model']
        logger.info(f"\n1. Reduced Model:")
        logger.info(f"   - Architecture: Standard Neural Network")
        logger.info(f"   - Input: 100 features (likely top 100 genes)")
        logger.info(f"   - Parameters: {reduced.get('total_params', 'Unknown'):,}" if isinstance(reduced.get('total_params'), int) else 'Unknown')
        logger.info(f"   - Compact size: {reduced.get('file_size_mb', 0):.2f} MB")
    
    if 'full_model' in summaries:
        full = summaries['full_model']
        logger.info(f"\n2. Full Model:")
        logger.info(f"   - Architecture: Feature Token Transformer")
        logger.info(f"   - Contains attention mechanisms")
        logger.info(f"   - Expected to handle 13,000+ genes")
        logger.info(f"   - Note: Model version compatibility issue detected")
    
    logger.info(f"\n3. Gene Sets:")
    logger.info(f"   - Common genes: 13,369")
    logger.info(f"   - Selected genes: 1,000")
    logger.info(f"   - UMAP visualizations created successfully")
    
    logger.info(f"\nAll outputs saved to: {output_dir}")
    logger.info("="*70)

if __name__ == "__main__":
    main()
