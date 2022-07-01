
import os
import sys
os.environ["OMP_NUM_THREADS"] = "11"
os.environ["OPENBLAS_NUM_THREADS"] = "8" # export OPENBLAS_NUM_THREADS=4 
os.environ["MKL_NUM_THREADS"] = "11" # export MKL_NUM_THREADS=6
os.environ["VECLIB_MAXIMUM_THREADS"] = "8" # export VECLIB_MAXIMUM_THREADS=4
os.environ["NUMEXPR_NUM_THREADS"] = "11" # export NUMEXPR_NUM_THREADS=6
os.environ["NUMBA_CACHE_DIR"]='/tmp/numba_cache'
import numpy as np
import pandas as pd
import scipy as sp
import scipy.sparse
import h5py

import scvi
import scanpy as sc

import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, precision_score, recall_score, matthews_corrcoef, balanced_accuracy_score, average_precision_score, f1_score, log_loss

with h5py.File('data/DOGMA_pbmc.h5', 'r') as f:
    print(f.keys())
    peak_names = np.array(f['peak_names'], dtype='S32').astype(str)
    ADT_names = np.array(f['ADT_names'], dtype='S32').astype(str)
    gene_names = np.array(f['gene_names'], dtype='S32').astype(str)
    X = sp.sparse.csc_matrix(
        (np.array(f['RNA.data'], dtype=np.float32), 
         np.array(f['RNA.indices'], dtype=np.int32),
         np.array(f['RNA.indptr'], dtype=np.int32)
        ), 
        shape = np.array(f['RNA.shape'], dtype=np.int32)).tocsc().astype(np.float32).T.toarray()
    Y = np.array(f['ADT'], dtype=np.float32)
    Z = sp.sparse.csc_matrix(
        (np.array(f['peaks.data'], dtype=np.float32),
         np.array(f['peaks.indices'], dtype=np.int32),
         np.array(f['peaks.indptr'], dtype=np.int32)
        ), 
        shape = np.array(f['peaks.shape'], dtype=np.int32)).tocsc().astype(np.float32).T.toarray()
    cell_types = np.array(f['predicted.celltypes'], dtype='S32').astype(str)
    batches = np.array(f['batches'], dtype=np.float32)
    cell_ids = np.array(f['cell_ids'], dtype='S32').astype(str)
    
    
def comp_cor_flatten(x, y, file_name, plot=True):
    pearson_r, pearson_p = scipy.stats.pearsonr(x, y)
    print(f"Found pearson's correlation/p of {pearson_r:.4f}/{pearson_p:.4g}")
    spearman_corr, spearman_p = scipy.stats.spearmanr(x, y)
    print(f"Found spearman's collelation/p of {spearman_corr:.4f}/{spearman_p:.4g}")
    if plot:
        fig, axes = plt.subplots(1,1, figsize=(5,5))
        plt.scatter(x, y, alpha=0.2)
        plt.savefig(path_root+file_name)
    return pearson_r, spearman_corr


cell_type_list =  ['CD4 T', 'CD8 T', 'B', 'NK', 'DC', 'Mono']
cell_type = cell_type_list[int(sys.argv[1])]
print(cell_type)
path_root = 'result/ex2/%s/totalVI/'%cell_type


adata = sc.AnnData(X = pd.DataFrame(
    X[cell_types!=cell_type,:], 
    index=cell_ids[cell_types!=cell_type], 
    columns=gene_names))
adata.obsm['ADT'] = pd.DataFrame(
    Y[cell_types!=cell_type,:],
    index=cell_ids[cell_types!=cell_type], 
    columns=ADT_names)
adata.obs['batch'] = pd.DataFrame(
    batches[cell_types!=cell_type,:],
    index=cell_ids[cell_types!=cell_type])
adata.layers["counts"] = adata.X.copy()
sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata)
adata.raw = adata


scvi.model.TOTALVI.setup_anndata(
    adata,
    layer="counts",
    batch_key="batch",
    protein_expression_obsm_key="ADT"
)

vae = scvi.model.TOTALVI(adata, 
                         n_latent=32,
                         latent_distribution="normal")
vae.view_anndata_setup()

vae.train(
   train_size=0.9,
   early_stopping=15,
   reduce_lr_on_plateau=False
)

vae.save(path_root+'model/', overwrite=True)


vae = scvi.model.TOTALVI.load(path_root+'model/', adata)


def get_norm_imputation_RNA(vae, adata_test, chunk_size=10000,
        n_samples=50, scale_protein=True):
    X_hat = []
    for j in np.arange(int(np.ceil(adata_test.shape[0]/chunk_size))):
        _X_hat, _ = vae.get_normalized_expression(
                adata_test[j*chunk_size:(j+1)*chunk_size],
                n_samples=n_samples, scale_protein=scale_protein)
        X_hat.append(_X_hat)
    X_hat = np.concatenate(X_hat, axis=0)
    return X_hat


def get_norm_imputation_ADT(vae, adata_test, chunk_size=10000,
        n_samples=50, scale_protein=True):
    Y_hat = []
    for j in np.arange(int(np.ceil(adata_test.shape[0]/chunk_size))):
        _, _Y_hat = vae.get_normalized_expression(
                adata_test[j*chunk_size:(j+1)*chunk_size],
                include_protein_background=True,
                n_samples=n_samples, scale_protein=scale_protein)
        Y_hat.append(_Y_hat)
    Y_hat = np.concatenate(Y_hat, axis=0)
    return Y_hat



############################################################
#
# Test on the hold-out cell type with random missing
#
############################################################

res = []
for batch in [0,1]:
    id_cell = ((cell_types==cell_type)&(batches[:,0]==batch))
    X_test = np.log1p(X[id_cell,:]/np.sum(X[id_cell,:], axis=-1, keepdims=True)*1e4) 
    Y_test = np.log1p(Y[id_cell,:]/np.sum(Y[id_cell,:], axis=-1, keepdims=True)*1e4)

    # mask on rna
    adata_test = sc.AnnData(X = pd.DataFrame(
        np.zeros_like(X[id_cell,:]), 
        index=cell_ids[id_cell], 
        columns=gene_names))
    adata_test.obsm['ADT'] = pd.DataFrame(
        Y[id_cell,:], 
        index=cell_ids[id_cell], 
        columns=ADT_names)
    adata_test.obs['batch'] = pd.DataFrame(
        batches[id_cell,:],
        index=cell_ids[id_cell])
    adata_test.layers["counts"] = adata_test.X.copy()
    sc.pp.normalize_total(adata_test, target_sum=1e4)
    sc.pp.log1p(adata_test)
    adata_test.raw = adata_test
    X_hat = get_norm_imputation_RNA(vae, adata_test)
    X_hat = np.log1p(X_hat * 1e4)

    pr, sr = comp_cor_flatten(X_test.flatten(), X_hat.flatten(), 'adt_rna.png', plot=False)
    mse = np.mean((X_test.flatten()-X_hat.flatten())**2)
    res.append(['RNA', batch, pr, sr, mse])
    _df = pd.DataFrame({
        'RNA':gene_names,
        'Pearson r':[np.corrcoef(X_hat[:,i], X_test[:,i])[0,1] for i in np.arange(len(gene_names))],
        'Spearman r':[scipy.stats.spearmanr(X_hat[:,i], X_test[:,i])[0] for i in np.arange(len(gene_names))],
        'MSE':np.mean((X_hat-X_test)**2, axis=0)
    })
    print(np.quantile(_df['MSE'].values, [0.,0.5,1.0]))
    _df.to_csv(path_root+'res_totalVI_RNA_%d.csv'%(int(batch)))
    
    # mask on adt
    adata_test = sc.AnnData(X = pd.DataFrame(
        X[id_cell,:], 
        index=cell_ids[id_cell], 
        columns=gene_names))
    adata_test.obsm['ADT'] = pd.DataFrame(
        np.zeros_like(Y[id_cell,:]), 
        index=cell_ids[id_cell], 
        columns=ADT_names)
    adata_test.obs['batch'] = pd.DataFrame(
        batches[id_cell,:],
        index=cell_ids[id_cell])
    adata_test.layers["counts"] = adata_test.X.copy()
    sc.pp.normalize_total(adata_test, target_sum=1e4)
    sc.pp.log1p(adata_test)
    adata_test.raw = adata_test
    Y_hat = get_norm_imputation_ADT(vae, adata_test)
    Y_hat = np.log1p(Y_hat * 1e4)

    pr, sr = comp_cor_flatten(Y_test.flatten(), Y_hat.flatten(), 'rna_adt.png', plot=False)
    mse = np.mean((Y_test.flatten()-Y_hat.flatten())**2)
    res.append(['ADT', batch, pr, sr, mse])

    _df = pd.DataFrame({
        'ADT':ADT_names,
        'Pearson r':[np.corrcoef(Y_hat[:,i], Y_test[:,i])[0,1] for i in np.arange(len(ADT_names))],
        'Spearman r':[scipy.stats.spearmanr(Y_hat[:,i], Y_test[:,i])[0] for i in np.arange(len(ADT_names))],
        'MSE':np.mean((Y_hat-Y_test)**2, axis=0)
    })
    print(np.quantile(_df['Pearson r'].values, [0.,0.5,1.0]))
    _df.to_csv(path_root+'res_totalVI_ADT_%d.csv'%(int(batch)))
    
pd.DataFrame(res, columns=['Target', 'Batch', 'Metric_1', 'Metric_2', 'MSE']
            ).to_csv(path_root+'res_totalVI_overall.csv')



############################################################
#
# Test on the hold-out cell type with random missing
#
############################################################

def get_norm_imputation(vae, adata_test, chunk_size=5000,
        n_samples=50, scale_protein=True):
    X_hat = []
    Y_hat = []
    for j in np.arange(int(np.ceil(adata_test.shape[0]/chunk_size))):
        _X_hat, _Y_hat = vae.get_normalized_expression(
                adata_test[j*chunk_size:(j+1)*chunk_size], 
                n_samples=n_samples, scale_protein=scale_protein)
        X_hat.append(_X_hat)
        Y_hat.append(_Y_hat)
    X_hat = np.concatenate(X_hat, axis=0)
    Y_hat = np.concatenate(Y_hat, axis=0)
    return X_hat, Y_hat



mask_list = pd.read_csv('data/mask_dogma.csv', index_col=0).values
from tqdm import tqdm

dim_input_rna = X.shape[1]
dim_input_adt = Y.shape[1]
dim_input_atac = Z.shape[1]

X_test = np.log1p(X[cell_types==cell_type,:]/np.sum(X[cell_types==cell_type,:], axis=-1, keepdims=True)*1e4) 
Y_test = np.log1p(Y[cell_types==cell_type,:]/np.sum(Y[cell_types==cell_type,:], axis=-1, keepdims=True)*1e4)
batch_test = batches[cell_types==cell_type]
X_test_raw = X[cell_types==cell_type,:]
Y_test_raw = Y[cell_types==cell_type,:]
del X, Y

for ip in np.arange(1,9):
    res = []
    p = ip/10.    
    for i in tqdm(range(10)):
        vae = scvi.model.TOTALVI.load(path_root+'model/', adata)
        
        mask = mask_list[(10*(ip-1)+i):(10*(ip-1)+i+1),:] 

        id_rna = np.where(mask[0,:dim_input_rna]==0)[0]
        id_adt = np.where(mask[0,dim_input_rna:-dim_input_atac]==0)[0] 
        id_rna_test = np.where(mask[0,:dim_input_rna]==1)[0]
        id_adt_test = np.where(mask[0,dim_input_rna:-dim_input_atac]==1)[0] 
        X_sub = np.zeros_like(X_test, dtype=np.float32)
        Y_sub = np.zeros_like(Y_test, dtype=np.float32)
        X_sub[:, id_rna] = X_test_raw[:, id_rna]
        Y_sub[:, id_adt] = Y_test_raw[:, id_adt]

        adata_test = sc.AnnData(X = pd.DataFrame(
            X_sub, 
            index=cell_ids[cell_types==cell_type], 
            columns=gene_names))
        adata_test.obsm['ADT'] = pd.DataFrame(
            Y_sub, 
            index=cell_ids[cell_types==cell_type], 
            columns=ADT_names)
        adata_test.layers["counts"] = adata_test.X.copy()
        adata_test.obs['batch'] = pd.DataFrame(
            batch_test,
            index=cell_ids[cell_types==cell_type]
        )
        sc.pp.normalize_total(adata_test, target_sum=1e4)
        sc.pp.log1p(adata_test)
        adata_test.raw = adata_test
        X_hat, Y_hat = get_norm_imputation(vae, adata_test)

        X_hat = np.log1p(X_hat * 1e4)
        Y_hat = np.log1p(Y_hat * 1e4)
        
        for batch in [0,1]:                                    

            _X_hat = X_hat[batch_test[:,0]==batch,:][:,id_rna_test].flatten()
            _X_test = X_test[batch_test[:,0]==batch,:][:,id_rna_test].flatten()
            res.append([i, batch, 'RNA',
                        np.corrcoef(_X_hat, _X_test)[0,1],
                        scipy.stats.spearmanr(_X_hat, _X_test)[0],
                        np.mean((_X_hat-_X_test)**2)
                       ])


            _Y_hat = Y_hat[batch_test[:,0]==batch,:][:,id_adt_test].flatten()
            _Y_test = Y_test[batch_test[:,0]==batch,:][:,id_adt_test].flatten()
            res.append([i, batch, 'ADT',
                        np.corrcoef(_Y_hat, _Y_test)[0,1],
                        scipy.stats.spearmanr(_Y_hat, _Y_test)[0],
                        np.mean((_Y_hat-_Y_test)**2)
                       ])

        del X_hat, Y_hat, adata_test, vae
    
    pd.DataFrame(
        res, columns=['i', 'Batch', 'Target', 'Pearson r', 'Spearman r', 'MSE']
    ).to_csv(path_root+'res_totalVI_masked_%.01f.csv'%(p))
