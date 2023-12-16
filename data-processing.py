import os
import time
import json
import itertools
from pathlib import Path

import scvi
import anndata
import numpy as np
import scanpy as sc
from tqdm import tqdm
from gears import PertData, GEARS
from sklearn.model_selection import train_test_split
    
def read_data(config: dict) -> anndata.AnnData:
    
    Path("./data").mkdir(parents=True, exist_ok=True)

    if config["dataset_name"] == "adamson":
        
        data_dir = Path("./data")
        pert_data = PertData(data_dir)
        pert_data.load(data_name="adamson")

        adata = sc.read(data_dir / "adamson/perturb_processed.h5ad")
        
        ori_batch_col = "control"
        adata.obs["celltype"] = adata.obs["condition"].astype("category")
        
        adata.obs["str_batch"] = adata.obs[ori_batch_col].astype(str)
        batch_id_labels = adata.obs["str_batch"].astype("category").cat.codes.values
        adata.obs["batch_id"] = batch_id_labels
        adata.var["gene_name"] = adata.var.index.tolist()
        
        genes = adata.var["gene_name"].tolist()
        
    elif config["dataset_name"] == "PBMC":
        
        adata = scvi.data.pbmc_dataset()  # 11990 × 3346
        ori_batch_col = "batch"
        adata.obs["celltype"] = adata.obs["str_labels"].astype("category")
        adata.var = adata.var.set_index("gene_symbols")

        adata.obs["str_batch"] = adata.obs[ori_batch_col].astype(str)
        batch_id_labels = adata.obs["str_batch"].astype("category").cat.codes.values
        adata.obs["batch_id"] = batch_id_labels
        adata.var["gene_name"] = adata.var.index.tolist()

        genes = adata.var["gene_name"].tolist()

    else:
        raise FileNotFoundError("dataset_name must be one of \"adamson\" or \"PBMC\"")
    
    return adata

def preprocessing(config: dict, adata: anndata.AnnData) -> np.ndarray:
    
    # Filter low-expression cells and genes
    sc.pp.filter_genes(
        adata,
        min_counts=config["filter_gene_cutoff"]
    )
    sc.pp.filter_cells(
        adata,
        min_counts=config["filter_cell_cutoff"]
    )

    # Normalize total expression count of each sample
    if config["normalization"]:
        sc.pp.normalize_total(
            adata,
            target_sum=1500,
            exclude_highly_expressed=True,
            inplace=True,
        )
    
    # Filter HVGs
    sc.pp.highly_variable_genes(
        adata,
        n_top_genes=config["hvg_num"],
        flavor="seurat_v3",
        subset=True,
    )

    # Put raw counts into modified log-bins
    X_binned = np.zeros((adata.X.shape[0], adata.X.shape[1]), dtype='int')
    print(X_binned.shape)
    if config["log_transform"]:
        for i, j in tqdm(zip(*adata.X.nonzero()), total=len(adata.X.nonzero()[0])):
            v = adata.X[i, j]
            if v > config['bin_cutoff']:
                X_binned[i][j] = np.ceil((np.log2(v) - np.log2(config['bin_cutoff'])) / config['bin_size']) + config['bin_cutoff']
            else:
                X_binned[i][j] = v
    else:
        for i, j in tqdm(zip(*adata.X.nonzero()), total=len(adata.X.nonzero()[0])):
            v = adata.X[i, j]
            X_binned[i][j] = v / config['bin_size']
            
    return X_binned
        
def main():
    
    with open("./settings.json") as f:
        config = json.loads(f.read())
        
    adata = read_data(config)
    X_binned = preprocessing(config, adata)
    
    all_counts = X_binned
    gene_names = adata.var["gene_name"].tolist()

    celltypes_labels = adata.obs["celltype"].tolist()
    # num_types = len(set(celltypes_labels))
    celltypes_labels = np.array(celltypes_labels)

    batch_ids = adata.obs["batch_id"].tolist()
    # num_batch_types = len(set(batch_ids))
    batch_ids = np.array(batch_ids)

    (
        train_data,
        valid_data,
        train_celltype_labels,
        valid_celltype_labels,
        train_batch_labels,
        valid_batch_labels,
    ) = train_test_split(
        all_counts, celltypes_labels, batch_ids, test_size=0.1, shuffle=True
    )
    
    Path("./processed_data").mkdir(parents=True, exist_ok=True)

    np.save("./processed_data/train_data.npy", train_data)
    np.save("./processed_data/valid_data.npy", valid_data)
    np.save("./processed_data/train_celltype_labels.npy", train_celltype_labels)
    np.save("./processed_data/valid_celltype_labels.npy", valid_celltype_labels)
    np.save("./processed_data/train_batch_labels.npy", train_batch_labels)
    np.save("./processed_data/valid_batch_labels.npy", valid_batch_labels)
    np.save("./processed_data/gene_names.npy", gene_names)
    
if __name__ == "__main__":
    main()