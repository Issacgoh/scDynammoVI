import anndata
import scanpy as sc
import pandas as pd
import numpy as np

class DataPreprocessor:
    def __init__(self, scRNA_path, ATAC_path, SNV_path, metadata_path):
        self.scRNA_path = scRNA_path
        self.ATAC_path = ATAC_path
        self.SNV_path = SNV_path
        self.metadata_path = metadata_path

    def load_and_preprocess_data(self):
        # Load scRNA-seq and ATAC-seq data
        adata_scRNA = sc.read_h5ad(self.scRNA_path)
        adata_ATAC = sc.read_h5ad(self.ATAC_path)
        
        # Load SNV and metadata
        snv_data = pd.read_csv(self.SNV_path, index_col=0)
        metadata = pd.read_csv(self.metadata_path, index_col=0)
        
        # Preprocess scRNA-seq and ATAC-seq data
        self.preprocess_scRNA(adata_scRNA)
        self.preprocess_ATAC(adata_ATAC)
        
        # Filter SNVs based on VAF and integrate with scRNA data
        self.filter_and_integrate_SNV(adata_scRNA, snv_data, metadata)
        
        # Integrate metadata with scRNA and ATAC data
        self.integrate_metadata(adata_scRNA, metadata)
        self.integrate_metadata(adata_ATAC, metadata)
        
        return adata_scRNA, adata_ATAC

    def preprocess_scRNA(self, adata):
        sc.pp.filter_cells(adata, min_genes=200)
        sc.pp.filter_genes(adata, min_cells=3)
        sc.pp.normalize_total(adata, target_sum=1e4)
        sc.pp.log1p(adata)
        sc.pp.highly_variable_genes(adata, min_mean=0.0125, max_mean=3, min_disp=0.5)
        adata = adata[:, adata.var.highly_variable]
        sc.pp.scale(adata, max_value=10)
        sc.tl.pca(adata, svd_solver='arpack')
        sc.pp.neighbors(adata)
        sc.tl.umap(adata)

    def preprocess_ATAC(self, adata):
        # Assuming ATAC-seq data is already in a suitable matrix format for analysis
        # Basic preprocessing: normalization and filtering
        sc.pp.normalize_total(adata, target_sum=1e4)
        sc.pp.log1p(adata)
        sc.pp.highly_variable_genes(adata, flavor='seurat_v3')
        adata = adata[:, adata.var.highly_variable]
        sc.pp.scale(adata, max_value=10)
        sc.tl.pca(adata, svd_solver='arpack')

    def filter_and_integrate_SNV(self, adata_scRNA, snv_data, metadata):
        # Calculate VAF for SNVs and filter based on 1% threshold
        # For simplicity, assuming 'metadata' contains cell IDs and donor-specific info to calculate VAF
        # This is a simplified example; actual implementation will depend on how SNV data and cell metadata are structured
        vaf_data = snv_data.apply(lambda x: (x > 0).mean(), axis=1)
        filtered_snv_data = snv_data.loc[vaf_data > 0.01]
        
        # Encode SNVs as binary and integrate with scRNA data
        binary_snv_data = filtered_snv_data.applymap(lambda x: 1 if x > 0 else 0)
        adata_scRNA.obsm['SNV'] = binary_snv_data.loc[adata_scRNA.obs_names].values

    def integrate_metadata(self, adata, metadata):
        for col in metadata.columns:
            adata.obs[col] = metadata.loc[adata.obs_names, col]

