#!/usr/bin/env python3
"""
Extract REAL embeddings from trained models - NO SYNTHETIC DATA
For manuscript methods section and UMAP visualization
Author: Jyotidip Barman
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from sklearn.neighbors import KNeighborsClassifier
import logging
import pickle

try:
    import umap
    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False
    print("Install umap-learn for visualization: pip install umap-learn")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

class RealEmbeddingExtractor:
    """Extract real embeddings from trained models - no synthetic data"""
    
    def __init__(self, base_dir: Path):
        self.base_dir = base_dir
        self.output_dir = base_dir / 'results' / 'real_embeddings_only'
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.figures_dir = self.output_dir / 'figures'
        self.figures_dir.mkdir(exist_ok=True)
        
    def extract_reduced_model_embeddings(self, cohort_name: str) -> dict:
        """Extract real embeddings from reduced MLP model"""
        
        logger.info(f"\n{'='*60}")
        logger.info(f"Extracting REAL embeddings for {cohort_name} - Reduced Model")
        logger.info(f"{'='*60}")
        
        # Paths
        model_path = self.base_dir / 'reduced_model_distilled' / 'model_reduced.h5'
        data_path = self.base_dir / 'results' / f'bulk_{cohort_name}' / 'predictions_final.csv'
        genes_path = self.base_dir / 'reduced_model_distilled' / 'selected_genes.txt'
        
        # Verify paths exist
        if not model_path.exists():
            logger.error(f"Model not found: {model_path}")
            return {}
        if not data_path.exists():
            logger.error(f"Data not found: {data_path}")
            return {}
            
        # Load model
        model = keras.models.load_model(model_path, compile=False)
        
        # Get embedding layer (penultimate dense layer before output)
        embedding_layer = model.get_layer('dense_2')  # 64-dimensional embeddings
        embedder = keras.Model(inputs=model.input, outputs=embedding_layer.output)
        
        logger.info(f"Model loaded, extracting from layer: dense_2 (64-dim)")
        
        # Load the 100 selected genes
        selected_genes = []
        if genes_path.exists():
            with open(genes_path, 'r') as f:
                all_genes = [line.strip() for line in f if line.strip()]
                selected_genes = all_genes[:100]  # Use first 100 genes
            logger.info(f"Loaded {len(selected_genes)} selected genes")
        
        # Load predictions and prepare real input data
        pred_df = pd.read_csv(data_path)
        logger.info(f"Loaded predictions: {pred_df.shape}")
        
        # Try to load the actual gene expression data used for predictions
        # Look for the processed data file with gene expressions
        expression_path = self.base_dir / 'results' / f'bulk_{cohort_name}' / 'processed' / 'expression_matrix.csv'
        
        if not expression_path.exists():
            # Alternative path
            expression_path = self.base_dir / 'results' / f'bulk_{cohort_name}' / 'expression_data.csv'
        
        if expression_path.exists():
            logger.info(f"Loading expression data from: {expression_path}")
            expr_df = pd.read_csv(expression_path)
            
            # Get the 100 selected genes
            available_genes = [g for g in selected_genes if g in expr_df.columns]
            
            if len(available_genes) == 100:
                X_real = expr_df[available_genes].values
                logger.info(f"Using real expression data for {len(available_genes)} genes")
            else:
                logger.error(f"Only {len(available_genes)} genes found in expression data")
                return {}
        else:
            # Try to reconstruct from saved preprocessing
            scaler_path = self.base_dir / 'reduced_model_distilled' / 'scaler_100genes.pkl'
            
            if scaler_path.exists():
                logger.info("Loading saved preprocessing pipeline")
                with open(scaler_path, 'rb') as f:
                    scaler = pickle.load(f)
                
                # The predictions file should have been made with the same preprocessing
                # Extract the features that were used
                logger.warning("Expression data not found - cannot extract real embeddings")
                logger.info("Please provide the expression matrix used for predictions")
                return {}
            else:
                logger.error("Cannot find expression data or preprocessing pipeline")
                return {}
        
        # Normalize the data
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_real)
        
        # Extract embeddings
        logger.info("Extracting embeddings from real data...")
        embeddings = embedder.predict(X_scaled, batch_size=32, verbose=1)
        
        logger.info(f"Extracted embeddings shape: {embeddings.shape}")
        
        # Get predictions and metadata
        results = {
            'embeddings': embeddings,
            'sample_ids': pred_df['sample_id'].values if 'sample_id' in pred_df else np.arange(len(pred_df)),
            'predictions': pred_df['persister_probability'].values if 'persister_probability' in pred_df else None,
            'cohort': cohort_name,
            'model_type': 'reduced_mlp',
            'embedding_dim': embeddings.shape[1],
            'n_samples': len(embeddings)
        }
        
        return results
    
    def extract_transformer_embeddings(self, cohort_name: str) -> dict:
        """
        Extract transformer embeddings using saved preprocessing
        Note: This requires the preprocessing pipeline from training
        """
        
        logger.info(f"\n{'='*60}")
        logger.info(f"Extracting Transformer embeddings for {cohort_name}")
        logger.info(f"{'='*60}")
        
        # Check for saved embeddings from training
        saved_embeddings_path = self.base_dir / 'results' / 'models' / f'{cohort_name}_transformer_embeddings.npy'
        
        if saved_embeddings_path.exists():
            logger.info(f"Loading saved transformer embeddings from: {saved_embeddings_path}")
            embeddings = np.load(saved_embeddings_path)
            
            # Load predictions for metadata
            pred_path = self.base_dir / 'results' / f'bulk_{cohort_name}' / 'predictions_final.csv'
            pred_df = pd.read_csv(pred_path)
            
            results = {
                'embeddings': embeddings,
                'sample_ids': pred_df['sample_id'].values if 'sample_id' in pred_df else np.arange(len(pred_df)),
                'predictions': pred_df['persister_probability'].values if 'persister_probability' in pred_df else None,
                'cohort': cohort_name,
                'model_type': 'transformer',
                'embedding_dim': embeddings.shape[1],
                'n_samples': len(embeddings)
            }
            
            return results
        else:
            logger.warning("Transformer embeddings not found")
            logger.info("To generate transformer embeddings:")
            logger.info("1. Run inference with the transformer model")
            logger.info("2. Save embeddings from the penultimate layer during inference")
            logger.info("3. Place in: results/models/{cohort_name}_transformer_embeddings.npy")
            return {}
    
    def create_publication_umap(self, embeddings_dict: dict):
        """Create publication-quality UMAP visualization"""
        
        if not UMAP_AVAILABLE:
            logger.error("UMAP not installed. Cannot create visualization.")
            return None
            
        if not embeddings_dict or 'embeddings' not in embeddings_dict:
            logger.error("No embeddings to visualize")
            return None
        
        embeddings = embeddings_dict['embeddings']
        predictions = embeddings_dict.get('predictions')
        cohort = embeddings_dict['cohort']
        model_type = embeddings_dict['model_type']
        
        logger.info(f"\nCreating UMAP for {cohort} - {model_type}")
        logger.info(f"Embeddings shape: {embeddings.shape}")
        
        # UMAP with best practice parameters
        reducer = umap.UMAP(
            n_neighbors=30,  # Higher for better global structure
            min_dist=0.1,
            n_components=2,
            metric='euclidean',
            random_state=42
        )
        
        # Fit UMAP
        umap_coords = reducer.fit_transform(embeddings)
        
        # Create figure
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        # Panel 1: Persister tertiles
        ax = axes[0]
        if predictions is not None:
            tertiles = pd.qcut(predictions, q=3, labels=['Low', 'Intermediate', 'High'])
            colors = {'Low': '#2ecc71', 'Intermediate': '#f39c12', 'High': '#e74c3c'}
            
            for tertile in ['Low', 'Intermediate', 'High']:
                mask = tertiles == tertile
                ax.scatter(
                    umap_coords[mask, 0],
                    umap_coords[mask, 1],
                    c=colors[tertile],
                    label=f'{tertile} (n={mask.sum()})',
                    alpha=0.6,
                    s=40,
                    edgecolors='black',
                    linewidth=0.5
                )
            ax.legend(title='Persister Tertile')
        else:
            ax.scatter(umap_coords[:, 0], umap_coords[:, 1], alpha=0.6, s=40)
        
        ax.set_xlabel('UMAP 1', fontsize=12)
        ax.set_ylabel('UMAP 2', fontsize=12)
        ax.set_title(f'{cohort} - Tertile Stratification', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        # Panel 2: Continuous persister probability
        ax = axes[1]
        if predictions is not None:
            scatter = ax.scatter(
                umap_coords[:, 0],
                umap_coords[:, 1],
                c=predictions,
                cmap='RdYlBu_r',
                alpha=0.6,
                s=40,
                edgecolors='black',
                linewidth=0.5,
                vmin=0, vmax=1
            )
            cbar = plt.colorbar(scatter, ax=ax)
            cbar.set_label('Persister Probability', fontsize=11)
        else:
            ax.scatter(umap_coords[:, 0], umap_coords[:, 1], alpha=0.6, s=40)
        
        ax.set_xlabel('UMAP 1', fontsize=12)
        ax.set_ylabel('UMAP 2', fontsize=12)
        ax.set_title(f'{cohort} - Continuous Score', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        # Panel 3: High vs Low separation with kNN metrics
        ax = axes[2]
        if predictions is not None:
            tertiles = pd.qcut(predictions, q=3, labels=['Low', 'Intermediate', 'High'])
            high_mask = tertiles == 'High'
            low_mask = tertiles == 'Low'
            
            # Plot only high and low for clarity
            ax.scatter(
                umap_coords[low_mask, 0],
                umap_coords[low_mask, 1],
                c='#2ecc71',
                label=f'Low (n={low_mask.sum()})',
                alpha=0.7,
                s=50,
                marker='o'
            )
            ax.scatter(
                umap_coords[high_mask, 0],
                umap_coords[high_mask, 1],
                c='#e74c3c',
                label=f'High (n={high_mask.sum()})',
                alpha=0.7,
                s=50,
                marker='^'
            )
            
            # Calculate kNN separation metric
            subset_embeddings = np.vstack([embeddings[high_mask], embeddings[low_mask]])
            subset_labels = np.concatenate([np.ones(high_mask.sum()), np.zeros(low_mask.sum())])
            
            knn = KNeighborsClassifier(n_neighbors=15)
            knn.fit(subset_embeddings, subset_labels)
            knn_probs = knn.predict_proba(subset_embeddings)[:, 1]
            knn_auc = roc_auc_score(subset_labels, knn_probs)
            
            # Add metrics inset
            textstr = f'kNN Separation (k=15)\nAUC = {knn_auc:.3f}'
            props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
            ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=11,
                   verticalalignment='top', bbox=props)
            
            ax.legend(loc='lower right')
        
        ax.set_xlabel('UMAP 1', fontsize=12)
        ax.set_ylabel('UMAP 2', fontsize=12)
        ax.set_title(f'{cohort} - High vs Low Separation', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        # Overall title
        fig.suptitle(f'{model_type.upper()} Model Embeddings - Real Data Only', 
                    fontsize=16, fontweight='bold', y=1.02)
        plt.tight_layout()
        
        # Save figure
        fig_path = self.figures_dir / f'{cohort}_{model_type}_real_embeddings_umap.png'
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved UMAP visualization to: {fig_path}")
        
        return umap_coords

def main():
    """Main pipeline - real embeddings only"""
    
    # Base directory
    base_dir = Path('/scratch/project_2010376/JDs_Project/AML_Persister_Analysis')
    
    # Initialize extractor
    extractor = RealEmbeddingExtractor(base_dir)
    
    logger.info("="*70)
    logger.info("REAL EMBEDDINGS EXTRACTION - NO SYNTHETIC DATA")
    logger.info("="*70)
    
    # Process each cohort
    for cohort in ['BeatAML', 'TCGA']:
        
        # Extract reduced model embeddings (this works)
        reduced_embeddings = extractor.extract_reduced_model_embeddings(cohort)
        
        if reduced_embeddings:
            # Create UMAP visualization
            umap_coords = extractor.create_publication_umap(reduced_embeddings)
            
            # Save embeddings
            np.save(extractor.output_dir / f'{cohort}_reduced_embeddings.npy', 
                   reduced_embeddings['embeddings'])
            
            if umap_coords is not None:
                np.save(extractor.output_dir / f'{cohort}_reduced_umap.npy', umap_coords)
            
            logger.info(f"\n{cohort} Reduced Model Summary:")
            logger.info(f"  Samples: {reduced_embeddings['n_samples']}")
            logger.info(f"  Embedding dimension: {reduced_embeddings['embedding_dim']}")
        
        # Try transformer embeddings (if available)
        transformer_embeddings = extractor.extract_transformer_embeddings(cohort)
        
        if transformer_embeddings:
            # Create UMAP visualization
            umap_coords = extractor.create_publication_umap(transformer_embeddings)
            
            # Save embeddings
            np.save(extractor.output_dir / f'{cohort}_transformer_embeddings.npy',
                   transformer_embeddings['embeddings'])
            
            if umap_coords is not None:
                np.save(extractor.output_dir / f'{cohort}_transformer_umap.npy', umap_coords)
            
            logger.info(f"\n{cohort} Transformer Summary:")
            logger.info(f"  Samples: {transformer_embeddings['n_samples']}")
            logger.info(f"  Embedding dimension: {transformer_embeddings['embedding_dim']}")
    
    # Generate methods text
    logger.info("\n" + "="*70)
    logger.info("METHODS SECTION TEXT")
    logger.info("="*70)
    
    methods_text = """
Model embeddings were extracted from the penultimate layer of the reduced model 
(64-dimensional dense layer) using the actual gene expression data for the 100 
selected genes. UMAP projection parameters: n_neighbors=30, min_dist=0.1, 
metric='euclidean', random_state=42. Tertiles were defined using the 33.33 
and 66.67 percentiles of persister probability. Separation quality between 
high and low tertiles was quantified using k-nearest neighbors classification 
(k=15) with AUC as the metric. All visualizations used real patient data only, 
with no synthetic samples.
"""
    
    print(methods_text)
    
    # Save methods text
    with open(extractor.output_dir / 'methods_text.txt', 'w') as f:
        f.write(methods_text)
    
    logger.info(f"\nAll outputs saved to: {extractor.output_dir}")
    logger.info("="*70)

if __name__ == "__main__":
    from typing import Dict
    main()
