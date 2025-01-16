import scanpy as sc
import matplotlib.pyplot as plt
import anndata as ad
import numpy as np

# Preprocess data function
def preprocess_data(h5ad_file_path):
    adata = sc.read_h5ad(h5ad_file_path)
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    sc.pp.scale(adata)
    return adata

# Perform dimensionality reduction (t-SNE/UMAP)
def perform_dimensionality_reduction(adata, method='umap', n_pcs=50, n_neighbors=15, perplexity=30):
    sc.pp.pca(adata, n_comps=n_pcs)
    sc.pp.neighbors(adata, n_neighbors=n_neighbors, n_pcs=n_pcs)
    if method == 'umap':
        sc.tl.umap(adata)
    elif method == 'tsne':
        sc.tl.tsne(adata, n_pcs=n_pcs, perplexity=perplexity, random_state=42)
    return adata

# Map clusters to specific labels
def map_clusters(adata, cluster_column, mapping):
    adata.obs['mapped_labels'] = adata.obs[cluster_column].map(mapping).fillna('NA')
    return adata

# Plot t-SNE or UMAP results
def plot_reduction(adata, color, method='umap', palette=None):
    if method == 'umap':
        sc.pl.umap(adata, color=color, legend_loc='right margin', frameon=False, palette=palette)
    elif method == 'tsne':
        sc.pl.tsne(adata, color=color, legend_loc='right margin', frameon=False, palette=palette)

# Load and preprocess Luo dataset
h5ad_file_path_luo = 'Luo2021nature_cleaned.h5ad'
adata_luo = preprocess_data(h5ad_file_path_luo)

# Perform UMAP on Luo dataset
adata_luo = perform_dimensionality_reduction(adata_luo, method='umap', n_pcs=30)

# Define a mapping for clusters to labels
cluster_mapping = {
    '1': 'naive', '3': 'naive', '6': 'naive', '7': 'naive',
    '2': 'effector', '9': 'effector', '11': 'effector'
}
adata_luo = map_clusters(adata_luo, cluster_column='leiden', mapping=cluster_mapping)

# Plot UMAP for mapped labels
plot_reduction(adata_luo, color='mapped_labels', method='umap')

# Save the modified Luo dataset
adata_luo.write('Luo2021_mapped_labels.h5ad')

# Load and preprocess Wend dataset
h5ad_file_path_wend = 'Wend2024sciadv_cleaned.h5ad'
adata_wend = preprocess_data(h5ad_file_path_wend)

# Perform t-SNE on Wend dataset
adata_wend = perform_dimensionality_reduction(adata_wend, method='tsne', n_pcs=50)

# Filter out 'none' labels and save
adata_wend = adata_wend[~adata_wend.obs['subset'].str.contains("none", na=False)].copy()
adata_wend.write('Wend2024_filtered.h5ad')

# Verify unique labels in both datasets
print("Luo unique labels:", adata_luo.obs['mapped_labels'].unique())
print("Wend unique labels:", adata_wend.obs['subset'].unique())