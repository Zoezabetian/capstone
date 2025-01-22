import scanpy as sc
import pandas as pd
import numpy as np
import torch
from scsims import SIMS

# Define file paths for Wend and Luo datasets
wend_adata_path = '/Users/zoezabetian/Desktop/M-R-lab/immune-sims/leo_data/fer_cd4_treg_ngs_final.h5ad'
wend_checkpoint_path = '/Users/zoezabetian/Desktop/M-R-lab/immune-sims/models/wend_best_model.ckpt'

luo_adata_path = '/Users/zoezabetian/Desktop/M-R-lab/immune-sims/leo_data/fer_cd4_treg_ngs_final.h5ad'
luo_checkpoint_path = '/Users/zoezabetian/Desktop/M-R-lab/immune-sims/models/luo_best_model.ckpt'

# Function to preprocess data
def preprocess_data(adata):
    sc.pp.filter_cells(adata, min_genes=100)
    sc.pp.filter_genes(adata, min_cells=3)
    sc.pp.scale(adata)
    return adata

# Function to predict and analyze data
def predict_and_analyze(adata, checkpoint_path):
    sims = SIMS(weights_path=checkpoint_path, map_location=torch.device('cpu'))
    cell_predictions = sims.predict(adata, num_workers=0).reset_index(drop=True)
    adata.obs = adata.obs.join(cell_predictions)
    adata.obs['cell_predictions'] = cell_predictions['pred_0'].values
    adata.obs['prob_0'] = cell_predictions['prob_0'].values
    adata.obs['prob_1'] = cell_predictions['prob_1'].values

    # Summarize predictions
    total_cells = len(adata.obs)
    counts = adata.obs['cell_predictions'].value_counts()
    percentages = (counts / total_cells) * 100
    print("\nPrediction Summary:")
    for cell_type, percentage in percentages.items():
        print(f"{cell_type}: {percentage:.2f}% of total cells")

    # Low-probability cells
    threshold = 0.6
    low_prob_cells = adata.obs[adata.obs['prob_0'] < threshold]
    low_prob_cells.to_csv('low_prob_cells.csv', index=False)
    print("Low-probability cells saved to low_prob_cells.csv")

    # Filter for analysis
    filtered_adata = adata[~adata.obs['sample'].str.contains("NoAct"), :]
    return adata, filtered_adata

# Function to display sample predictions
def display_sample_predictions(filtered_adata):
    unique_samples = filtered_adata.obs['sample'].unique()
    for sample in unique_samples:
        sample_data = filtered_adata.obs[filtered_adata.obs['sample'] == sample]
        prediction_counts = sample_data['cell_predictions'].value_counts()
        total_count = len(sample_data)
        prediction_percentages = (prediction_counts / total_count) * 100
        print(f"\nSample: {sample}")
        print(f"Treg: {prediction_percentages.get('Treg', 0):.2f}%")
        print(f"Teffector: {prediction_percentages.get('Teffector', 0):.2f}%")

# Process Wend dataset
print("\nProcessing Wend dataset...")
wend_adata = sc.read_h5ad(wend_adata_path)
wend_adata = preprocess_data(wend_adata)
wend_adata, filtered_wend_adata = predict_and_analyze(wend_adata, wend_checkpoint_path)
display_sample_predictions(filtered_wend_adata)

# Process Luo dataset
print("\nProcessing Luo dataset...")
luo_adata = sc.read_h5ad(luo_adata_path)
luo_adata = preprocess_data(luo_adata)
luo_adata, filtered_luo_adata = predict_and_analyze(luo_adata, luo_checkpoint_path)
display_sample_predictions(filtered_luo_adata)
