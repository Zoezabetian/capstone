# immune cell classification capstone project - ML pipeline

This repository contains scripts for processing biological data, converting file formats, and training machine learning models. I am using these files for my immune cell classification research. 

## Files

### 1. `Rdata-to-h5ad.R`
- **Purpose**: Process `.RData` files and convert `.h5Seurat` to `.h5ad`.
- **Key Features**:
  - Extracts counts, metadata, and gene names from `.RData`.
  - Converts `.h5Seurat` to `.h5ad` format using SeuratDisk.
  - Handles file naming conflicts by appending timestamps.

### 2. `data-clean.py`
- **Purpose**: Clean, preprocess, and visualize `.h5ad` datasets.
- **Key Features**:
  - Normalizes, log-transforms, and scales gene expression data.
  - Performs dimensionality reduction (t-SNE, UMAP).
  - Maps clusters to specific labels.
  - Filters and saves cleaned datasets.

### 3. `exploratory_analysis.ipynb`
- **Purpose**: Exploratory Data Analysis with cleaned `.h5ad` datasets.
- **Key Features**:
  - Compute and visualize UMAPS for both datasets.
  - Compare clustering methods.
  - View distributions of data.

### 4. `my-pipeline.py`
- **Purpose**: Machine learning pipeline for scRNA-seq analysis.
- **Key Features**:
  - Classes: Preprocessor, Trainer, Predicter, as well as Pipeline Class
  - Uses PyTorch Lightning and SIMS for model training.
  - Includes validation and evaluation with confusion matrix visualization.

### 5. `predict.ipynb`
- **Purpose**: Main notebook for making predictions on data with best models.
- **Key Features**:
  - Predicts with two models.
  - Shows distribution of predictions.

### 6. `predictions.py`
- **Purpose**: Concise python script with predictions code.
- **Key Features**:
  - Preprocess, predict, display.
  - Runs predictions for both datasets.
