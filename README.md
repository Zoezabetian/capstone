# bme capstone project - training pipeline

This repository contains scripts for processing biological data, converting file formats, and training machine learning models.

## Files

### 1. `data-clean.py`
- **Purpose**: Preprocess and visualize `.h5ad` datasets.
- **Key Features**:
  - Normalizes, log-transforms, and scales gene expression data.
  - Performs dimensionality reduction (t-SNE, UMAP).
  - Maps clusters to specific labels.
  - Filters and saves cleaned datasets.

### 2. `my-training.py`
- **Purpose**: Train machine learning models on biological data.
- **Key Features**:
  - Implements grid search for hyperparameter tuning.
  - Uses PyTorch Lightning and SIMS for model training.
  - Includes validation and evaluation with confusion matrix visualization.

### 3. `Rdata-to-h5ad.R`
- **Purpose**: Process `.RData` files and convert `.h5Seurat` to `.h5ad`.
- **Key Features**:
  - Extracts counts, metadata, and gene names from `.RData`.
  - Converts `.h5Seurat` to `.h5ad` format using SeuratDisk.
  - Handles file naming conflicts by appending timestamps.

## Usage

### Preprocessing (`data-clean.py`)
1. Update file paths in the script to match your dataset locations.
2. Run the script to normalize, reduce dimensions, and filter the data.

### Training (`my-training.py`)
1. Set the dataset path and target column in the command-line arguments.
2. Specify hyperparameter ranges for the grid search.
3. Run the script to train models and evaluate performance.

### Conversion (`Rdata-to-h5ad.R`)
1. Replace the placeholder paths with your input and output file paths.
2. Run the script in R to process `.RData` files and convert `.h5Seurat` to `.h5ad`.

## Requirements
- Python: `scanpy`, `matplotlib`, `numpy`, `scsims`, `pytorch-lightning`
- R: `Seurat`, `SeuratDisk`, `Matrix`