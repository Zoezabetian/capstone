if (!requireNamespace("Seurat", quietly = TRUE)) install.packages("Seurat")
if (!requireNamespace("SeuratDisk", quietly = TRUE)) install.packages("SeuratDisk")
if (!requireNamespace("Matrix", quietly = TRUE)) install.packages("Matrix")

library(Seurat)
library(SeuratDisk)
library(Matrix)

# function to load .RData and extract counts and metadata
process_rdata <- function(input_rdata, counts_file, metadata_file, genes_file) {
  # load  RData file
  load(input_rdata)
  
  counts_matrix <- assay(sce.sling, "counts") #single cell experiment object
  metadata <- data.frame(
    Unique_ID = rownames(colData(sce.sling)),  # extracting unique identifiers
    subset = colData(sce.sling)$subset          # extracting the subset column
  )
  
  # save counts matrix, metadata, and gene names
  writeMM(counts_matrix, file = counts_file)
  write.csv(metadata, file = metadata_file, row.names = FALSE)
  write.csv(rownames(counts_matrix), file = genes_file, row.names = FALSE)
  
  print("Counts, metadata, and gene names saved.")
}

# function to convert .h5Seurat to .h5ad
convert_h5seurat_to_h5ad <- function(input_file, output_file) {
  # check input file
  if (!file.exists(input_file)) {
    stop(paste("Input file does not exist:", input_file))
  }
  
  # handle existing output file
  if (file.exists(output_file)) {
    timestamp <- format(Sys.time(), "%Y%m%d-%H%M%S")
    output_file <- paste0(sub(".h5ad$", "", output_file), "_", timestamp, ".h5ad")
    message("Output file exists. Renaming to: ", output_file)
  }
  
  # convert the file
  Convert(input_file, dest = "h5ad", overwrite = TRUE)
  file.rename(sub(".h5Seurat$", ".h5ad", input_file), output_file)
  message("File successfully converted and saved as: ", output_file)
}

# process RData file
process_rdata(
  input_rdata = "/path/to/Wend2024sciadv.RData",
  counts_file = "counts_matrix.mtx",
  metadata_file = "metadata.csv",
  genes_file = "genes.csv"
)

# convert .h5Seurat to .h5ad
convert_h5seurat_to_h5ad(
  input_file = "/path/to/input_file.h5Seurat",
  output_file = "output_file.h5ad"
)
