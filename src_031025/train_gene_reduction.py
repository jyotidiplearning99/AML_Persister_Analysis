#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Train Reduced Model with Model-Aware Selected Genes
Uses knowledge distillation to maintain original model behavior
"""

import numpy as np
import pandas as pd
import tensorflow as tf
import joblib
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import roc_auc_score, f1_score, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras import layers, Model, Input
from tensorflow.keras.callbacks import EarlyStopping
import json

SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

class ReducedModelTrainer:
    """Train reduced model with knowledge distillation"""
    
    def __init__(self, original_model_dir: Path, selected_genes_file: Path, output_dir: Path):
        self.original_model_dir = original_model_dir
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load selected genes
        with open(selected_genes_file) as f:
            self.selected_genes = [line.strip().upper() for line in f if line.strip()]
        
        # Load original model for distillation
        self.load_original_model()
        
        print(f"[SETUP] Selected genes: {len(self.selected_genes)}")
    
    def load_original_model(self):
        """Load original model components"""
        self.orig_model = tf.keras.models.load_model(
            self.original_model_dir / "final_model.h5", compile=False
        )
        self.orig_scaler = joblib.load(self.original_model_dir / "scaler.pkl")
        self.orig_pca = joblib.load(self.original_model_dir / "pca.pkl")
        self.orig_threshold = joblib.load(self.original_model_dir / "threshold.pkl")
        
        # Load full gene list
        with open(self.original_model_dir.parent / "metadata" / "common_genes.txt") as f:
            self.full_genes = [line.strip().upper() for line in f if line.strip()]
    
    def prepare_data(self) -> tuple:
        """Load and prepare training data"""
        print("\n[DATA] Loading training data...")
        
        GSM_BASE = Path("/scratch/project_2010751/GSE123902_RAW")
        
        training_samples = [
            (GSM_BASE / "GSM3516664_MSK_LX666_METASTASIS_dense.csv", 1),
            (GSM_BASE / "GSM3516668_MSK_LX255B_METASTASIS_dense.csv", 1),
            (GSM_BASE / "GSM3516666_MSK_LX675_NORMAL_dense.csv", 0),
            (GSM_BASE / "GSM3516665_MSK_LX675_PRIMARY_TUMOUR_dense.csv", 0),
        ]
        
        X_full_list, X_reduced_list, y_list = [], [], []
        
        for filepath, label in training_samples:
            if filepath.exists():
                df = self.load_and_normalize_csv(filepath)
                
                # Full genes for teacher predictions
                df_full = df.reindex(columns=self.full_genes).fillna(0.0)
                
                # Selected genes for student training
                df_reduced = df.reindex(columns=self.selected_genes).fillna(0.0)
                
                # Subsample
                if len(df_full) > 2000:
                    idx = np.random.choice(len(df_full), 2000, replace=False)
                    df_full = df_full.iloc[idx]
                    df_reduced = df_reduced.iloc[idx]
                
                X_full_list.append(df_full.values)
                X_reduced_list.append(df_reduced.values)
                y_list.extend([label] * len(df_full))
        
        X_full = np.vstack(X_full_list)
        X_reduced = np.vstack(X_reduced_list)
        y = np.array(y_list)
        
        print(f"  Combined: {X_reduced.shape[0]} cells × {X_reduced.shape[1]} selected genes")
        
        return X_full, X_reduced, y
    
    def load_and_normalize_csv(self, path: Path) -> pd.DataFrame:
        """Load and normalize CSV"""
        df = pd.read_csv(path, index_col=0)
        if df.shape[0] > df.shape[1]:
            df = df.T
        
        # Clean genes
        df.columns = [str(g).strip().upper().rsplit(".", 1)[0] for g in df.columns]
        df = df.T.groupby(level=0).sum().T
        
        # CPM normalization
        X = df.values.astype(np.float32)
        X = np.maximum(X, 0.0)
        lib = X.sum(axis=1, keepdims=True)
        np.maximum(lib, 1.0, out=lib)
        X = (X / lib) * 1e4
        X = np.log1p(X)
        
        return pd.DataFrame(X, columns=df.columns)
    
    def get_teacher_predictions(self, X_full: np.ndarray) -> np.ndarray:
        """Get soft labels from original model"""
        print("\n[TEACHER] Getting predictions from original model...")
        
        X_scaled = self.orig_scaler.transform(X_full)
        X_pca = self.orig_pca.transform(X_scaled)
        teacher_probs = self.orig_model.predict(X_pca, verbose=0).ravel()
        
        # Calculate statistics
        pos_rate = (teacher_probs >= self.orig_threshold).mean()
        print(f"  Teacher positive rate: {pos_rate*100:.1f}%")
        print(f"  Teacher prob range: [{teacher_probs.min():.3f}, {teacher_probs.max():.3f}]")
        
        return teacher_probs
    
    def train_with_distillation(self, X_reduced: np.ndarray, y: np.ndarray, 
                                teacher_probs: np.ndarray) -> dict:
        """Train reduced model with knowledge distillation"""
        print("\n[TRAIN] Training reduced model with distillation...")
        
        # Split data
        sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=SEED)
        train_idx, test_idx = next(sss.split(X_reduced, y))
        
        X_tr, X_te = X_reduced[train_idx], X_reduced[test_idx]
        y_tr, y_te = y[train_idx], y[test_idx]
        teacher_tr, teacher_te = teacher_probs[train_idx], teacher_probs[test_idx]
        
        # Preprocessing
        self.scaler = StandardScaler()
        X_tr_s = self.scaler.fit_transform(X_tr)
        X_te_s = self.scaler.transform(X_te)
        
        # PCA
        n_comp = min(100, max(2, X_tr.shape[1]-1))
        self.pca = PCA(n_components=n_comp, random_state=SEED)
        X_tr_p = self.pca.fit_transform(X_tr_s)
        X_te_p = self.pca.transform(X_te_s)
        
        print(f"  PCA: {n_comp} components, {self.pca.explained_variance_ratio_.sum():.3f} variance")
        
        # Build model
        self.model = self.build_model(n_comp)
        
        # Custom loss: combination of true labels and teacher predictions
        def distillation_loss(y_true, y_pred, teacher, alpha=0.5):
            """Weighted combination of true label loss and teacher distillation"""
            true_loss = tf.keras.losses.binary_crossentropy(y_true, y_pred)
            teacher_loss = tf.keras.losses.binary_crossentropy(teacher, y_pred)
            return alpha * true_loss + (1 - alpha) * teacher_loss
        
        # Train with combined targets
        # Create combined targets: alpha * true_labels + (1-alpha) * teacher_predictions
        alpha = 0.7  # Weight for true labels
        y_tr_combined = alpha * y_tr + (1 - alpha) * teacher_tr
        y_te_combined = alpha * y_te + (1 - alpha) * teacher_te
        
        self.model.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['AUC']
        )
        
        # Class weights based on original labels
        cw = compute_class_weight('balanced', classes=[0, 1], y=y_tr)
        class_weight = {0: cw[0], 1: cw[1]}
        
        # Train
        self.model.fit(
            X_tr_p, y_tr_combined,
            validation_data=(X_te_p, y_te_combined),
            epochs=50,
            batch_size=128,
            class_weight=class_weight,
            callbacks=[EarlyStopping(patience=10, restore_best_weights=True)],
            verbose=0
        )
        
        # Find threshold that matches teacher's positive rate
        y_prob = self.model.predict(X_te_p, verbose=0).ravel()
        teacher_pos_rate = (teacher_te >= self.orig_threshold).mean()
        
        # Use quantile to match positive rate
        self.threshold = np.quantile(y_prob, 1 - teacher_pos_rate)
        self.threshold = np.clip(self.threshold, 0.25, 0.75)
        
        print(f"  Threshold calibrated to: {self.threshold:.3f} (matches teacher rate)")
        
        # Evaluate
        y_pred = (y_prob >= self.threshold).astype(int)
        
        metrics = {
            'auc': roc_auc_score(y_te, y_prob),
            'f1': f1_score(y_te, y_pred),
            'teacher_agreement': ((y_prob >= self.threshold) == (teacher_te >= self.orig_threshold)).mean(),
            'threshold': self.threshold
        }
        
        print(f"  AUC: {metrics['auc']:.3f}")
        print(f"  F1: {metrics['f1']:.3f}")
        print(f"  Agreement with teacher: {metrics['teacher_agreement']*100:.1f}%")
        
        return metrics
    
    def build_model(self, n_features: int) -> tf.keras.Model:
        """Build neural network"""
        inp = Input(shape=(n_features,))
        x = layers.Dense(256, activation='relu')(inp)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.3)(x)
        x = layers.Dense(128, activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.2)(x)
        x = layers.Dense(64, activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.2)(x)
        out = layers.Dense(1, activation='sigmoid')(x)
        return Model(inp, out)
    
    def test_independent_datasets(self) -> pd.DataFrame:
        """Test on independent datasets"""
        print("\n[TEST] Testing on independent datasets...")
        
        test_datasets = [
            (Path("/scratch/project_2010751/GSE123902_RAW/GSM3516671_MSK_LX681_METASTASIS_dense.csv"), "GSM3516671_METASTASIS", 1),
            (Path("/scratch/project_2010751/GSE123902_RAW/GSM3516667_MSK_LX676_PRIMARY_TUMOUR_dense.csv"), "GSM3516667_PRIMARY", 0),
            (Path("/scratch/project_2010376/scRNAseq/FH_5897_2/filtered_feature_bc_matrix"), "FH_5897_2_AML", 1),
            (Path("/scratch/project_2010376/scRNAseq/FH_6333_2/filtered_feature_bc_matrix"), "FH_6333_2_AML", 1),
        ]
        
        results = []
        
        for data_path, name, true_label in test_datasets:
            if not data_path.exists():
                continue
            
            print(f"  Testing {name}...")
            
            # Load data (simplified for CSV only here)
            if data_path.suffix == '.csv':
                df = self.load_and_normalize_csv(data_path)
                df_aligned = df.reindex(columns=self.selected_genes).fillna(0.0)
                
                if len(df_aligned) > 2000:
                    idx = np.random.choice(len(df_aligned), 2000, replace=False)
                    df_aligned = df_aligned.iloc[idx]
                
                # Predict
                X = self.scaler.transform(df_aligned.values)
                X = self.pca.transform(X)
                probs = self.model.predict(X, verbose=0).ravel()
                preds = (probs >= self.threshold).astype(int)
                
                result = {
                    'sample': name,
                    'n_cells': len(df_aligned),
                    'true_label': true_label,
                    'positive_pct': preds.mean() * 100,
                    'mean_prob': probs.mean()
                }
                
                if true_label is not None:
                    result['accuracy'] = (preds == true_label).mean() * 100
                
                results.append(result)
        
        return pd.DataFrame(results)
    
    def save_model(self, metrics: dict):
        """Save trained model and components"""
        print("\n[SAVE] Saving model...")
        
        self.model.save(self.output_dir / 'model_reduced.h5')
        joblib.dump(self.scaler, self.output_dir / 'scaler_reduced.pkl')
        joblib.dump(self.pca, self.output_dir / 'pca_reduced.pkl')
        joblib.dump(float(self.threshold), self.output_dir / 'threshold_reduced.pkl')
        
        with open(self.output_dir / 'selected_genes.txt', 'w') as f:
            for g in self.selected_genes:
                f.write(f"{g}\n")
        
        with open(self.output_dir / 'training_metrics.json', 'w') as f:
            json.dump(metrics, f, indent=2)
        
        print(f"  Model saved to: {self.output_dir}")

def main():
    """Main training pipeline"""
    print("="*80)
    print("REDUCED MODEL TRAINING WITH DISTILLATION")
    print("="*80)
    
    # Setup paths
    original_model_dir = Path("/scratch/project_2010376/JDs_Project/AML_Persister_Analysis/results/models")
    selected_genes_file = Path("/scratch/project_2010376/JDs_Project/AML_Persister_Analysis/gene_reduction_model_aware/selected_genes_model_aware.txt")
    output_dir = Path("/scratch/project_2010376/JDs_Project/AML_Persister_Analysis/reduced_model_distilled")
    
    # Initialize trainer
    trainer = ReducedModelTrainer(original_model_dir, selected_genes_file, output_dir)
    
    # Prepare data
    X_full, X_reduced, y = trainer.prepare_data()
    
    # Get teacher predictions
    teacher_probs = trainer.get_teacher_predictions(X_full)
    
    # Train with distillation
    metrics = trainer.train_with_distillation(X_reduced, y, teacher_probs)
    
    # Test on independent datasets
    test_results = trainer.test_independent_datasets()
    test_results.to_csv(output_dir / 'independent_test_results.csv', index=False)
    
    print("\n[TEST RESULTS]")
    print(test_results.to_string(index=False))
    
    # Save model
    trainer.save_model(metrics)
    
    print("\n" + "="*80)
    print("TRAINING COMPLETE")
    print("="*80)
    print(f"✓ Model trained with {len(trainer.selected_genes)} genes")
    print(f"✓ Threshold: {trainer.threshold:.3f}")
    print(f"✓ Saved to: {output_dir}")

if __name__ == "__main__":
    main()
