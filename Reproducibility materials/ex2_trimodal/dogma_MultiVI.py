
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
    X_raw = sp.sparse.csc_matrix(
        (np.array(f['RNA.data'], dtype=np.float32), 
         np.array(f['RNA.indices'], dtype=np.int32),
         np.array(f['RNA.indptr'], dtype=np.int32)
        ), 
        shape = np.array(f['RNA.shape'], dtype=np.int32)).tocsc().astype(np.float32).T.toarray()
    # Y = np.array(f['ADT'], dtype=np.float32)
    Z = sp.sparse.csc_matrix(
        (np.array(f['peaks.data'], dtype=np.float32),
         np.array(f['peaks.indices'], dtype=np.int32),
         np.array(f['peaks.indptr'], dtype=np.int32)
        ), 
        shape = np.array(f['peaks.shape'], dtype=np.int32)).tocsc().astype(np.float32).T.toarray()
    cell_types = np.array(f['predicted.celltypes'], dtype='S32').astype(str)
    cell_ids = np.array(f['cell_ids'], dtype='S32').astype(str)
    batches = np.array(f['batches'], dtype=np.float32)
    
X = np.log1p(X_raw/np.sum(X_raw, axis=1, keepdims=True)*1e4)
Z = (Z>0).astype(np.float32)

    
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
path_root = 'result/ex2/%s/MultiVI/'%cell_type


adata_paired = sc.AnnData(
    X = pd.DataFrame(
        np.c_[X,Z][cell_types!=cell_type,:], 
        index=cell_ids[cell_types!=cell_type], 
        columns=np.r_[gene_names, peak_names]), 
    var = pd.DataFrame(
        np.r_[np.repeat("Gene Expression", len(gene_names)), np.repeat("Peaks", len(peak_names))],
        columns=['modality'],
        index=np.r_[gene_names, peak_names]
    ),
)
adata_paired.obs['batch'] = pd.DataFrame(batches[cell_types!=cell_type,:], index=cell_ids[cell_types!=cell_type])

adata_rna = sc.AnnData(
    X = pd.DataFrame(
        np.c_[X,np.zeros_like(Z)][cell_types!=cell_type,:], 
        index=cell_ids[cell_types!=cell_type], 
        columns=np.r_[gene_names, peak_names]), 
    var = pd.DataFrame(
        np.r_[np.repeat("Gene Expression", len(gene_names)), np.repeat("Peaks", len(peak_names))],
        columns=['modality'],
        index=np.r_[gene_names, peak_names]
    ),
)
adata_rna.obs['batch'] = pd.DataFrame(batches[cell_types!=cell_type,:], index=cell_ids[cell_types!=cell_type])

adata_atac = sc.AnnData(
    X = pd.DataFrame(
        np.c_[np.zeros_like(X),Z][cell_types!=cell_type,:], 
        index=cell_ids[cell_types!=cell_type], 
        columns=np.r_[gene_names, peak_names]),
    var = pd.DataFrame(
        np.r_[np.repeat("Gene Expression", len(gene_names)), np.repeat("Peaks", len(peak_names))],
        columns=['modality'],
        index=np.r_[gene_names, peak_names]
    ),
)
adata_atac.obs['batch'] = pd.DataFrame(batches[cell_types!=cell_type,:], index=cell_ids[cell_types!=cell_type])

adata = scvi.data.organize_multiome_anndatas(adata_paired, adata_rna, adata_atac)
import gc
del adata_atac, adata_rna, adata_paired
gc.collect()


scvi.model.MULTIVI.setup_anndata(
    adata, batch_key='modality', continuous_covariate_keys=['batch'])

vae = scvi.model.MULTIVI(
        adata, n_latent=32, gene_likelihood='nb',
    n_genes=(adata.var['modality']=='Gene Expression').sum(),
    n_regions=(adata.var['modality']=='Peaks').sum(),
)

vae.view_anndata_setup()

vae.train(
   train_size=0.9,
   early_stopping=15
)


vae.save(path_root+'model/', overwrite=True)

vae = scvi.model.MULTIVI.load(path_root+'model/', adata)





############################################################
#
# Test on the hold-out cell type
#
############################################################

def get_norm_imputation_RNA(vae, adata_test, chunk_size=512,
        n_samples=50):
    X_hat = []
    for j in np.arange(int(np.ceil(adata_test.shape[0]/chunk_size))):
        _X_hat = np.zeros((adata_test[j*chunk_size:(j+1)*chunk_size].shape[0], len(gene_names)), dtype=np.float32)
        for _ in range(n_samples):
            _X_hat += vae.get_normalized_expression(
                    adata_test[j*chunk_size:(j+1)*chunk_size], 
                    n_samples=1, return_numpy=True)
        _X_hat /= n_samples
        X_hat.append(_X_hat)        
    X_hat = np.concatenate(X_hat, axis=0)
    return X_hat

def get_norm_imputation_ATAC(vae, adata_test, chunk_size=512,
        n_samples=50):
    Z_hat = []
    for j in np.arange(int(np.ceil(adata_test.shape[0]/chunk_size))):
        _Z_hat = vae.get_accessibility_estimates(
                adata_test[j*chunk_size:(j+1)*chunk_size],
                return_numpy=True)
        Z_hat.append(_Z_hat)
    Z_hat = np.concatenate(Z_hat, axis=0)
    return Z_hat

res = []
for batch in [0,1]:
    id_cell = ((cell_types==cell_type)&(batches[:,0]==batch))
    X_test = X[id_cell,:] 
    Z_test = Z[id_cell,:]

    # mask on rna
    adata_test = sc.AnnData(
        X = pd.DataFrame(
            np.c_[np.zeros_like(X_test),Z_test], 
            index=cell_ids[id_cell], 
            columns=np.r_[gene_names, peak_names]), 
        var = pd.DataFrame(
            np.r_[np.repeat("Gene Expression", len(gene_names)), np.repeat("Peaks", len(peak_names))],
            columns=['modality'],
            index=np.r_[gene_names, peak_names]
        ),
    )
    adata_test.obs['batch'] = pd.DataFrame(batches[id_cell,:], index=cell_ids[id_cell])
    adata_test.obs['modality'] = pd.DataFrame(np.repeat('accessibility', np.sum(id_cell)), index=cell_ids[id_cell])
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
    _df.to_csv(path_root+'res_MultiVI_RNA_%d.csv'%(int(batch)))

    # mask on atac
    adata_test = sc.AnnData(
        X = pd.DataFrame(
            np.c_[X_test,np.zeros_like(Z_test)], 
            index=cell_ids[id_cell], 
            columns=np.r_[gene_names, peak_names]), 
        var = pd.DataFrame(
            np.r_[np.repeat("Gene Expression", len(gene_names)), np.repeat("Peaks", len(peak_names))],
            columns=['modality'],
            index=np.r_[gene_names, peak_names]
        ),
    )
    adata_test.obs['batch'] = pd.DataFrame(batches[id_cell,:], index=cell_ids[id_cell])
    adata_test.obs['modality'] = pd.DataFrame(np.repeat('expression', np.sum(id_cell)), index=cell_ids[id_cell])
    Z_hat = get_norm_imputation_ATAC(vae, adata_test)

    auc = roc_auc_score(Z_test.flatten(), Z_hat.flatten())
    acc = np.mean(Z_test==(Z_hat>0.5))
    mse = np.mean((Z_hat-Z_test)**2)
    print(auc, acc)
    res.append(['ATAC', batch, auc, acc, mse])

    _df = pd.DataFrame({
        'ATAC':peak_names,
        'AUC':[np.nan if len(Z_test[:,i])==np.sum(Z_test[:,i]==0) or len(Z_test[:,i])==np.sum(Z_test[:,i]==1)
                     else roc_auc_score(Z_test[:,i], Z_hat[:,i]) for i in np.arange(len(peak_names))],
        'Accuracy':np.mean(Z_test==(Z_hat>0.5), axis=0),
        'AR':[balanced_accuracy_score(Z_test[:,i], Z_hat[:,i]>0.5) for i in np.arange(len(peak_names))],
        'AP':[average_precision_score(Z_test[:,i], Z_hat[:,i]) for i in np.arange(len(peak_names))],
        'BCE':[log_loss(Z_test[:,i], Z_hat[:,i], labels=np.arange(2)) for i in np.arange(len(peak_names))],
        'MSE':np.mean((Z_hat-Z_test)**2, axis=0)
    })
    print(np.quantile(_df['MSE'].values, [0.,0.5,1.0]))
    _df.to_csv(path_root+'res_MultiVI_ATAC_%d.csv'%(int(batch)))
    

pd.DataFrame(res, columns=['Target', 'Batch', 'Metric_1', 'Metric_2', 'MSE']
            ).to_csv(path_root+'res_MultiVI_overall.csv')



############################################################
#
# Test on the hold-out cell type with random missing
#
############################################################


mask_list = pd.read_csv('data/mask_dogma.csv', index_col=0).values

dim_input_rna = X.shape[1]
dim_input_atac = Z.shape[1]

X_test = X[cell_types==cell_type,:]
Z_test = Z[cell_types==cell_type,:]
batch_test = batches[cell_types==cell_type]
X_test_raw = X_raw[cell_types==cell_type,:]

del X, Z

for ip in range(1,9):
    p = ip/10.
    res = []
    for i in range(10):
        if os.path.exists(path_root+'res_MultiVI_masked_%.01f_%d.csv'%(p,i)):
            continue
        vae = scvi.model.MULTIVI.load(path_root+'model/', adata)
        
        mask = mask_list[(10*(ip-1)+i):(10*(ip-1)+i+1),:] 

        id_rna = np.where(mask[0,:dim_input_rna]==0)[0]
        id_atac = np.where(mask[0,-dim_input_atac:]==0)[0]
        id_rna_test = np.where(mask[0,:dim_input_rna]==1)[0]
        id_atac_test = np.where(mask[0,-dim_input_atac:]==1)[0]      
        X_sub = np.zeros_like(X_test, dtype=np.float32)
        Z_sub = np.zeros_like(Z_test, dtype=np.float32)
        X_sub[:, id_rna] = X_test_raw[:, id_rna]
        l_rna = np.median(np.sum(X_test_raw[:, id_rna], axis=-1))
        X_sub = np.log1p(X_sub/np.sum(X_sub,axis=-1,keepdims=True)*l_rna)
        Z_sub[:, id_atac] = Z_test[:, id_atac]

        
        adata_test = sc.AnnData(
            X = pd.DataFrame(
                np.c_[X_sub,Z_sub], 
                index=cell_ids[cell_types==cell_type], 
                columns=np.r_[gene_names, peak_names]), 
            var = pd.DataFrame(
                np.r_[np.repeat("Gene Expression", len(gene_names)), np.repeat("Peaks", len(peak_names))],
                columns=['modality'],
                index=np.r_[gene_names, peak_names]
            ),
        )
        adata_test.obs['batch'] = pd.DataFrame(batch_test, index=cell_ids[cell_types==cell_type])
        adata_test.obs['modality'] = pd.DataFrame(
            np.repeat('paired', np.sum(cell_types==cell_type)), index=cell_ids[cell_types==cell_type])
        
        X_hat = get_norm_imputation_RNA(vae, adata_test)
        X_hat = np.log1p(X_hat * 1e4)
        Z_hat = get_norm_imputation_ATAC(vae, adata_test)
        
        for batch in [0,1]:
            _X_hat = X_hat[batch_test[:,0]==batch,:][:,id_rna_test].flatten()
            _X_test = X_test[batch_test[:,0]==batch,:][:,id_rna_test].flatten()
            res.append([i, batch, 'RNA',
                        np.corrcoef(_X_hat, _X_test)[0,1],
                        scipy.stats.spearmanr(_X_hat, _X_test)[0],
                        np.mean((_X_hat-_X_test)**2)
                       ])       

            _Z_hat = Z_hat[batch_test[:,0]==batch,:][:,id_atac_test].flatten()
            _Z_test = Z_test[batch_test[:,0]==batch,:][:,id_atac_test].flatten()
            res.append([i, batch, 'ATAC',
                        roc_auc_score(_Z_test, _Z_hat),
                        log_loss(_Z_test, _Z_hat),
                        np.mean((_Z_hat-_Z_test)**2)
                       ])       

        del X_hat, Z_hat, adata_test, vae
        gc.collect()
        
        pd.DataFrame(
            res, columns=['i', 'Batch', 'Target', 'Metric 1', 'Metric 2', 'MSE']
        ).to_csv(path_root+'res_MultiVI_masked_%.01f_%d.csv'%(p,i))
    pd.DataFrame(
        res, columns=['i', 'Batch', 'Target', 'Metric 1', 'Metric 2', 'MSE']
    ).to_csv(path_root+'res_MultiVI_masked_%.01f.csv'%(p))

