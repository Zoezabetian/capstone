# immune cell classification capstone project - ML pipeline

This repository contains scripts for processing biological data, converting file formats, and training machine learning models. I am using these files for my immune cell classification research. 

## Files

### 1. `data-clean.py`
- **Purpose**: Preprocess and visualize `.h5ad` datasets.
- **Key Features**:
  - Normalizes, log-transforms, and scales gene expression data.
  - Performs dimensionality reduction (t-SNE, UMAP).
  - Maps clusters to specific labels.
  - Filters and saves cleaned datasets.

### 2. `my-pipeline.py`
- **Purpose**: Machine learning pipeline for scRNA-seq analysis.
- **Key Features**:
  - Classes: Preprocessor, Trainer, Predicter, as well as Pipeline Class
  - Uses PyTorch Lightning and SIMS for model training.
  - Includes validation and evaluation with confusion matrix visualization.

### 3. `predict.ipynb`
- **Purpose**: Notebook for making predictions on data with best models.
- **Key Features**:
  - For immune cell project. 
  - Predicts with two models.
  - Shows distribution of predictions.

### 3. `Rdata-to-h5ad.R`
- **Purpose**: Process `.RData` files and convert `.h5Seurat` to `.h5ad`.
- **Key Features**:
  - Extracts counts, metadata, and gene names from `.RData`.
  - Converts `.h5Seurat` to `.h5ad` format using SeuratDisk.
  - Handles file naming conflicts by appending timestamps.
