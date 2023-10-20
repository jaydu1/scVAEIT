
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


with h5py.File('data/pbmc_count.h5', 'r') as f:
    print(f.keys())
    ADT_names = np.array(f['ADT_names'], dtype='S32').astype(str)
    gene_names = np.array(f['gene_names'], dtype='S32').astype(str)
    X = sp.sparse.csc_matrix(
        (np.array(f['RNA.data'], dtype=np.float32), 
         np.array(f['RNA.indices'], dtype=np.int32),
         np.array(f['RNA.indptr'], dtype=np.int32)
        ), 
        shape = np.array(f['RNA.shape'], dtype=np.int32)).tocsc().astype(np.float32).T.toarray()
    Y = np.array(f['ADT'], dtype=np.float32)
    cell_types = np.array(f['celltype.l1'], dtype='S32').astype(str)
    cell_ids = np.array(f['cell_ids'], dtype='S32').astype(str)
    
    
def comp_cor_flatten(x, y):
    pearson_r, pearson_p = scipy.stats.pearsonr(x, y)
    print(f"Found pearson's correlation/p of {pearson_r:.4f}/{pearson_p:.4g}")
    spearman_corr, spearman_p = scipy.stats.spearmanr(x, y)
    print(f"Found spearman's collelation/p of {spearman_corr:.4f}/{spearman_p:.4g}")
    return pearson_r, spearman_corr

path_root = 'result/ex1/full/TotalVI/'

adata = sc.AnnData(X = pd.DataFrame(
    X, 
    index=cell_ids, 
    columns=gene_names))
adata.obsm['ADT'] = pd.DataFrame(
    Y,
    index=cell_ids, 
    columns=ADT_names)
adata.layers["counts"] = adata.X.copy()
sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata)
adata.raw = adata


scvi.model.TOTALVI.setup_anndata(
    adata,
    layer="counts",
    protein_expression_obsm_key="ADT"
)


vae = scvi.model.TOTALVI(adata, 
                         n_latent=32,
                         latent_distribution="normal")

vae.train(
    max_epochs=500,
    train_size=0.9,
    early_stopping=True,
    early_stopping_patience=15, 
    reduce_lr_on_plateau=False,
    plan_kwargs={'pro_recons_weight':1.}
)

vae.save(path_root+'model/', overwrite=True)
vae = scvi.model.TOTALVI.load(path_root+'model/', adata)


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


for name in ['cbmc','reap']:
    with h5py.File('data/{}_count.h5'.format(name), 'r') as f:
        print(f.keys())
        ADT_names_new = np.array(f['ADT_names'], dtype='S32').astype(str)
        gene_names_new = np.array(f['gene_names'], dtype='S32').astype(str)
        X_new = sp.sparse.csc_matrix(
            (np.array(f['RNA.data'], dtype=np.float32), 
             np.array(f['RNA.indices'], dtype=np.int32),
             np.array(f['RNA.indptr'], dtype=np.int32)
            ), 
            shape = np.array(f['RNA.shape'], dtype=np.int32)).tocsc().astype(np.float32).T.toarray()
        Y_new = np.array(f['ADT'], dtype=np.float32)
        cell_ids_new = np.array(f['cell_ids'], dtype='S32').astype(str)
        
    gene_names = np.char.upper(np.char.replace(np.char.replace(gene_names, '_', '-'), '.', '-'))
    gene_names_new = np.char.upper(np.char.replace(np.char.replace(gene_names_new, '_', '-'), '.', '-'))
    gene_shared = np.intersect1d(gene_names, gene_names_new)
    id_rna = np.array([np.where(gene_names==i)[0][0] for i in gene_shared])
    id_rna_new = np.array([np.where(gene_names_new==i)[0][0] for i in gene_shared])    
    ADT_shared = np.intersect1d(ADT_names, ADT_names_new)
    id_adt = np.array([np.where(ADT_names==i)[0][0] for i in ADT_shared])
    id_adt_new = np.array([np.where(ADT_names_new==i)[0][0] for i in ADT_shared])
    
    X_test_raw = np.zeros((X_new.shape[0], X.shape[1]))
    X_test_raw[:, id_rna] = X_new[:, id_rna_new]
    Y_test_raw = np.zeros((Y_new.shape[0], Y.shape[1]))
    Y_test_raw[:, id_adt] = Y_new[:, id_adt_new]

    X_test = np.log(X_test_raw[:, id_rna]/np.sum(X_test_raw[:, id_rna], axis=1, keepdims=True)*1e4+1.)
    Y_test = np.log(Y_test_raw[:, id_adt]/np.sum(Y_test_raw[:, id_adt], axis=1, keepdims=True)*1e4+1.)
    
    path = path_root+name+'/'
    os.makedirs(path, exist_ok=True)
    
    # mask on rna
    adata_test = sc.AnnData(X = pd.DataFrame(
        np.zeros_like(X_test_raw), 
        index=cell_ids_new, 
        columns=gene_names))
    adata_test.obsm['ADT'] = pd.DataFrame(
        Y_test_raw,
        index=cell_ids_new, 
        columns=ADT_names)
    
    adata_test.layers["counts"] = adata_test.X.copy()
    sc.pp.normalize_total(adata_test, target_sum=1e4)
    sc.pp.log1p(adata_test)
    adata_test.raw = adata_test
    X_hat, Y_hat = get_norm_imputation(vae, adata_test)
    X_hat = np.log1p(X_hat[:, id_rna] * 1e4)
    Y_hat = np.log1p(Y_hat[:, id_adt] * 1e4)

    
    res = []
    pr, sr = comp_cor_flatten(Y_test.flatten(), Y_hat.flatten())
    res.append(['ADT', 'ADT', pr, sr])
    pr, sr = comp_cor_flatten(X_test.flatten(), X_hat.flatten())
    res.append(['ADT', 'RNA', pr, sr])

    _df = pd.DataFrame({
        'RNA':gene_shared,
        'Pearson r':[np.corrcoef(X_hat[:,i], X_test[:,i])[0,1] for i in np.arange(len(id_rna))],
        'Spearman r':[scipy.stats.spearmanr(X_hat[:,i], X_test[:,i])[0] for i in np.arange(len(id_rna))],
        'MSE':np.mean((X_hat-X_test)**2, axis=0)
    })
    print(np.quantile(_df['MSE'].values, [0.,0.5,1.0]))
    _df.to_csv(path+'res_totalVI_RNA.csv')
    

    # mask on adt
    adata_test = sc.AnnData(X = pd.DataFrame(
        X_test_raw, 
        index=cell_ids_new, 
        columns=gene_names))
    adata_test.obsm['ADT'] = pd.DataFrame(
        np.zeros_like(Y_test_raw), 
        index=cell_ids_new, 
        columns=ADT_names)
    adata_test.layers["counts"] = adata_test.X.copy()
    sc.pp.normalize_total(adata_test, target_sum=1e4)
    sc.pp.log1p(adata_test)
    adata_test.raw = adata_test
    X_hat, Y_hat = get_norm_imputation(vae, adata_test)
    X_hat = np.log1p(X_hat[:, id_rna] * 1e4)
    Y_hat = np.log1p(Y_hat[:, id_adt] * 1e4)
    
    pr, sr = comp_cor_flatten(Y_test.flatten(), Y_hat.flatten())
    res.append(['RNA', 'ADT', pr, sr])
    pr, sr = comp_cor_flatten(X_test.flatten(), X_hat.flatten())
    res.append(['RNA', 'RNA', pr, sr])

    _df = pd.DataFrame({
        'ADT':ADT_shared,
        'Pearson r':[np.corrcoef(Y_hat[:,i], Y_test[:,i])[0,1] for i in np.arange(len(id_adt))],
        'Spearman r':[scipy.stats.spearmanr(Y_hat[:,i], Y_test[:,i])[0] for i in np.arange(len(id_adt))],
        'MSE':np.mean((Y_hat-Y_test)**2, axis=0)
    })
    print(np.quantile(_df['Pearson r'].values, [0.,0.5,1.0]))
    _df.to_csv(path+'res_totalVI_ADT.csv')


    pd.DataFrame(res, columns=['Source', 'Target', 'Pearson r', 'Spearman r']
            ).to_csv(path+'res_totalVI_%s_overall.csv'%name)

    
    


