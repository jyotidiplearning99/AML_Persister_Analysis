AML Persister Cell Classifier Pipeline
Overview
A production-ready deep learning pipeline for identifying persister cells in Acute Myeloid Leukemia (AML) using single-cell RNA-sequencing data. The pipeline implements a transformer-based architecture with knowledge distillation, multi-stage gene selection, and comprehensive pathway analysis.
Key Features
SOTA Transformer Architecture: Feature Token Transformer with multi-head attention for cell classification
Intelligent Gene Reduction: From 13,000+ genes to 1,000 then 500 most informative genes
Knowledge Distillation: Maintains performance while reducing computational requirements
DepMap Integration: Identifies therapeutically relevant targets using CRISPR dependency data
Comprehensive Pathway Analysis: KEGG pathways, transcription factor networks, drug-pathway mapping
Production Safeguards: Forced threshold bounds, robust validation, coverage guards
Architecture
text


13,000 genes (Full Model)
    ↓ [Feature Token Transformer]
    ↓ [Stratified K-fold Validation]
    ↓ [Knowledge Distillation]
1,000 genes (Reduced Model)
    ↓ [DepMap Refinement]
500 genes (Therapeutic Panel)
    ↓ [Pathway/Network Analysis]
Biological Insights & Drug Targets
Requirements
System Requirements
GPU: NVIDIA A100 recommended (40GB+ VRAM)
RAM: 180GB for full model training
Storage: ~500GB for datasets and intermediate files
SLURM: HPC cluster with GPU support
Software Dependencies
python


tensorflow>=2.13.0
numpy>=1.21.0
pandas>=1.3.0
scikit-learn>=1.0.0
scipy>=1.7.0
joblib>=1.1.0
statsmodels>=0.13.0
networkx>=2.6.0
matplotlib>=3.4.0
plotly>=5.3.0
requests>=2.26.0
python-louvain  # For community detection
gseapy  # Optional: for ssGSEA scoring
Pipeline Steps
Step 1: Initial Full Model Training
Script: production_transformer_13092025.py
Purpose: Train transformer on full gene set (~13,000 genes)
bash


python production_transformer_13092025.py
Key Features:
CPM → log1p normalization
Feature Token Transformer architecture
Stratified k-fold validation (never LOGO)
Forced threshold bounds (0.25-0.75)
Class-weighted training
Outputs:
final_model.h5: Trained model
scaler.pkl, pca.pkl: Preprocessing artifacts
threshold.pkl: Calibrated threshold
common_genes.txt: Training gene order
Step 2: HPC Job Submission
Script: run_training.sbatch
bash


sbatch run_training.sbatch
Step 3-4: Production Inference
Script: inference_persister_transformer_14092025.py
Purpose: Apply model to new samples with automatic format detection
Supported Formats:
10x Genomics folders (filtered_feature_bc_matrix)
Dense CSV (cells × genes or genes × cells)
MTX triplets (GSE120221 format)
bash


python inference_persister_transformer_14092025.py \
    --model-dir /path/to/models \
    --out-dir /path/to/predictions \
    --aml-root /path/to/AML_data \
    --healthy-root /path/to/healthy_data
Step 5: Model-Aware Gene Selection
Script: model_aware_gene_selection_fixed.py
Purpose: Reduce from 13,000 to 1,000 genes using three complementary methods
Methods:
Method A: Differential expression on high-confidence predictions
Method B: PCA-based feature importance from trained model
Method C: Filter housekeeping genes while preserving cancer markers
bash


python model_aware_gene_selection_fixed.py
Step 6-7: Reduced Model Training with Distillation
Script: train_reduced_model.py
Purpose: Train 1,000-gene model using soft labels from original model
Features:
Knowledge distillation (α=0.7 for true labels)
Threshold calibration on healthy controls
Independent dataset validation
bash


python train_reduced_model.py
Step 8: DepMap Integration
Script: depmap_integration_v2.py
Purpose: Further refine to 500 genes based on AML dependency scores
bash


python depmap_integration_v2.py \
    --genes-file selected_genes.txt \
    --depmap-dir /path/to/DepMap_Datasets \
    --output-dir /path/to/depmap_refined
Outputs:
genes_500_depmap.txt: Final therapeutic gene panel
wet_lab_top100_genes.csv: Prioritized for validation
aml_dependencies_ranked.csv: Full dependency metrics
Step 9-10: Pathway & Network Analysis
Script: pathway_analysis_complete_fixed.py
Purpose: Comprehensive biological interpretation
Analyses:
KEGG pathway enrichment with FDR correction
Transcription factor regulatory networks
Drug-pathway integration
Interactive network visualizations
bash


python pathway_analysis_complete_fixed.py \
    --genes genes_500_depmap.txt \
    --output /path/to/pathway_analysis \
    --background-size 19000
Step 11: Module-Based Scoring
Script: module_score_analysis.py
Purpose: Compute per-sample pathway/module activity scores
bash


python module_score_analysis.py \
    --expression expression.csv \
    --metadata metadata.csv \
    --kegg-tf-dir /path/to/pathway_analysis \
    --method pca \
    --outdir /path/to/module_scores
Directory Structure
text


AML_Persister_Analysis/
├── src/
│   ├── production_transformer_13092025.py
│   ├── inference_persister_transformer_14092025.py
│   ├── gene_Reduction_production_transformer_20092025.py
│   ├── train_reduced_model.py
|
├── results/
│   ├── models/
│   │   ├── final_model.h5
│   │   ├── model_reduced.h5
│   │   └── *.pkl
│   ├── predictions/
│   │   └── *_predictions.csv
│── src/── pathway_analysis/
│       ├── pathway_analysis.py
│       └── depmap_gene_refinement.py
|       |___generate_data_expression_for_scoring.py
|       |___pathway_module_score_analysis.py
|       |___Refined_Analysis_on_score.py
|       |___depmap_aml_primary_analysis.py
├── data/
│   ├── GSE123902_RAW/
│   ├── AML_scRNA_decrypted/
│   └── GSE120221_RAW/
└── logs/
Performance Metrics
Full Model (13,000 genes)
AUC-ROC: 0.85-0.92
Balanced Accuracy: 0.80-0.85
MCC: 0.65-0.75
Reduced Model (1,000 genes)
AUC-ROC: 0.82-0.88
Teacher Agreement: >85%
Inference Speed: 10x faster
Quality Control
Threshold Calibration
Target FPR on healthy controls: 0.05
Forced bounds: [0.25, 0.75]
Validation: Stratified k-fold (never LOGO)
Gene Coverage Guards
Minimum coverage: 50% of training genes
Automatic harmonization: Ensembl IDs → symbols
Duplicate gene merging: Sum aggregation
Known AML Targets Tracked
FLT3, NPM1, DNMT3A: Common mutations
BCL2, MCL1: Anti-apoptotic targets
CD33, CD34: Surface markers
IDH1, IDH2: Metabolic enzymes

Contact
Lead Developer: Jyotidip Barman
Email:jyotidip.barman@helsinki.fi
Institution: University of Helsinki

Acknowledgments
DepMap Consortium for dependency data
KEGG database for pathway information
DoRothEA for TF-target relationships
Last updated: September 2025