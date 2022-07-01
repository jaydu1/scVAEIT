
import os
import sys
os.environ["OMP_NUM_THREADS"] = "16"
os.environ["OPENBLAS_NUM_THREADS"] = "16" # export OPENBLAS_NUM_THREADS=4 
os.environ["MKL_NUM_THREADS"] = "32" # export MKL_NUM_THREADS=6
os.environ["VECLIB_MAXIMUM_THREADS"] = "16" # export VECLIB_MAXIMUM_THREADS=4
os.environ["NUMEXPR_NUM_THREADS"] = "32" # export NUMEXPR_NUM_THREADS=6
# os.environ["NUMBA_CACHE_DIR"]='/tmp/numba_cache'
import numpy as np
import pandas as pd
import scipy as sp
import scipy.sparse
import h5py

import tensorflow as tf
import tensorflow_probability as tfp

from tensorflow.keras.layers import Layer, Dense, BatchNormalization, LeakyReLU
from tensorflow.keras.utils import Progbar

tfd = tfp.distributions

import matplotlib.pyplot as plt
from types import SimpleNamespace
from sklearn.model_selection import train_test_split


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
    
    
path_root = 'result/ex1/full/'

dim_input_arr = [len(gene_names), len(ADT_names)]
config = {
    'dim_input_arr': dim_input_arr,
    'dimensions':[256], 
    'dim_latent':32,
    'dim_block': np.array(dim_input_arr),
    'dist_block':['NB','NB'], 
    'dim_block_enc':np.array([256, 128]),
    'dim_block_dec':np.array([256, 128]),
    'dim_block_embed':np.array([32, 16]),
    
    'block_names':np.array(['rna', 'adt']),
    'uni_block_names':np.array(['rna','adt']),
    
    'beta_kl':1.,
    'beta_unobs':2./3.,
    'beta_modal':np.array([0.1,0.9]),
    'beta_reverse':0.5,

    "p_feat" : 0.2,
    "p_modal" : np.ones(2)/2,
    
}
config = SimpleNamespace(**config)
n_samples = 50


# preprocess
X = np.log(X/np.sum(X, axis=1, keepdims=True)*1e4+1.)
Y = np.log(Y/np.sum(Y, axis=1, keepdims=True)*1e4+1.)

# data spliting
data = np.c_[X, Y]
data_train_norm = np.exp(data)-1


from functools import partial
from scVAEIT.VAEIT import scVAEIT
model = scVAEIT(config, data)

model.train(
    valid=False, num_epoch=500, batch_size=512, save_every_epoch=50,
    verbose=True, checkpoint_dir=path_root+'checkpoint/'
)



def comp_cor_flatten(x, y):
    pearson_r, pearson_p = scipy.stats.pearsonr(x, y)
    print(f"Found pearson's correlation/p of {pearson_r:.4f}/{pearson_p:.4g}")
    spearman_corr, spearman_p = scipy.stats.spearmanr(x, y)
    print(f"Found spearman's collelation/p of {spearman_corr:.4f}/{spearman_p:.4g}")
    return pearson_r, spearman_corr


def evaluate(vae, dataset_test, X_test, Y_test, XY_test_norm,
             gene_names, ADT_names, id_rna, id_adt, l_rna, l_adt, path):
    
    mask = -np.ones((1, XY_test_norm.shape[1]), dtype=tf.keras.backend.floatx())
    mask[:, id_adt+vae.config.dim_input_arr[0]] = 0.
    recon = vae.get_recon(dataset_test, mask)
    X_hat = recon[:, :vae.config.dim_input_arr[0]]
    Y_hat = recon[:, vae.config.dim_input_arr[0]:]    
    X_hat = X_hat[:, id_rna]
    Y_hat = Y_hat[:, id_adt]
    X_hat = np.expm1(X_hat) / l_rna * np.sum(
            XY_test_norm[:, id_rna], axis=1, keepdims=True)
    X_hat = np.log1p( X_hat/np.sum(X_hat, axis=1, keepdims=True) * 1e4)
    Y_hat = np.expm1(Y_hat) / l_adt * np.sum(
            XY_test_norm[:, id_adt+vae.config.dim_input_arr[0]], axis=1, keepdims=True)    
    Y_hat = np.log1p( Y_hat/np.sum(Y_hat, axis=1, keepdims=True) * 1e4)
    
    
    res = []
    _df = pd.DataFrame({
        'RNA':gene_names[id_rna],
        'Pearson r':[np.corrcoef(X_hat[:,i], X_test[:,i])[0,1] for i in np.arange(len(id_rna))],
        'Spearman r':[scipy.stats.spearmanr(X_hat[:,i], X_test[:,i])[0] for i in np.arange(len(id_rna))],
        'MSE':np.mean((X_hat-X_test)**2, axis=0)
    })
    print(np.nanquantile(_df['Pearson r'].values, [0.,0.5,1.0]))
    _df.to_csv(path+'res_scVAEIT_RNA.csv')
    pr, sr = comp_cor_flatten(Y_test.flatten(), Y_hat.flatten())
    res.append(['ADT', 'ADT', pr, sr])
    pr, sr = comp_cor_flatten(X_test.flatten(), X_hat.flatten())
    res.append(['ADT', 'RNA', pr, sr])
    
    
    mask = -np.ones((1, XY_test_norm.shape[1]), dtype=tf.keras.backend.floatx())
    mask[:, id_rna] = 0.
    recon = vae.get_recon(dataset_test, mask)
    X_hat = recon[:, :vae.config.dim_input_arr[0]]
    Y_hat = recon[:, vae.config.dim_input_arr[0]:]    
    X_hat = X_hat[:, id_rna]
    Y_hat = Y_hat[:, id_adt]
    X_hat = np.expm1(X_hat) / l_rna * np.sum(
            XY_test_norm[:, id_rna], axis=1, keepdims=True)
    X_hat = np.log1p( X_hat/np.sum(X_hat, axis=1, keepdims=True) * 1e4)
    Y_hat = np.expm1(Y_hat) / l_adt * np.sum(
            XY_test_norm[:, id_adt+config.dim_input_arr[0]], axis=1, keepdims=True)    
    Y_hat = np.log1p( Y_hat/np.sum(Y_hat, axis=1, keepdims=True) * 1e4)    

    _df = pd.DataFrame({
        'ADT':ADT_names[id_adt],
        'Pearson r':[np.corrcoef(Y_hat[:,i], Y_test[:,i])[0,1] for i in np.arange(len(id_adt))],
        'Spearman r':[scipy.stats.spearmanr(Y_hat[:,i], Y_test[:,i])[0] for i in np.arange(len(id_adt))],
        'MSE':np.mean((Y_hat-Y_test)**2, axis=0)
    })
    print(np.nanquantile(_df['Pearson r'].values, [0.,0.5,1.0]))
    _df.to_csv(path+'res_scVAEIT_ADT.csv')
    pr, sr = comp_cor_flatten(Y_test.flatten(), Y_hat.flatten())
    res.append(['RNA', 'ADT', pr, sr])
    pr, sr = comp_cor_flatten(X_test.flatten(), X_hat.flatten())
    res.append(['RNA', 'RNA', pr, sr])
    
    pd.DataFrame(res, columns=['Source', 'Target', 'Pearson r', 'Spearman r']
            ).to_csv(path+'res_scVAEIT_overall.csv')
    return res


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

    gene_names = np.char.upper(np.char.replace(np.char.replace(gene_names, '_', '-'), '.', '-'))
    gene_names_new = np.char.upper(np.char.replace(np.char.replace(gene_names_new, '_', '-'), '.', '-'))
    gene_shared = np.intersect1d(gene_names, gene_names_new)
    id_rna = np.array([np.where(gene_names==i)[0][0] for i in gene_shared])
    id_rna_new = np.array([np.where(gene_names_new==i)[0][0] for i in gene_shared])
    
    ADT_shared = np.intersect1d(ADT_names, ADT_names_new)
    id_adt = np.array([np.where(ADT_names==i)[0][0] for i in ADT_shared])
    id_adt_new = np.array([np.where(ADT_names_new==i)[0][0] for i in ADT_shared])
    
    l_rna = np.median(np.sum(data_train_norm[:, id_rna], axis=-1))
    l_adt = np.median(np.sum(data_train_norm[:, id_adt+config.dim_input_arr[0]], axis=-1))
    
    X_test = X_new[:, id_rna_new]
    Y_test = Y_new[:, id_adt_new]
    X_test = np.log1p( X_test/np.sum(X_test, axis=1, keepdims=True) * 1e4)
    Y_test = np.log1p( Y_test/np.sum(Y_test, axis=1, keepdims=True) * 1e4)    
    
    XY_test_norm = np.zeros((X_new.shape[0], len(gene_names)+len(ADT_names)))
    XY_test_norm[:, id_rna] = X_new[:, id_rna_new]
    XY_test_norm[:, id_adt+config.dim_input_arr[0]] = Y_new[:, id_adt_new]
    
    _XY_test = XY_test_norm.copy()
    _XY_test[:, id_rna] = np.log1p(
        XY_test_norm[:, id_rna]/np.sum(XY_test_norm[:, id_rna], axis=1, keepdims=True)*l_rna)
    _XY_test[:, id_adt+config.dim_input_arr[0]] = np.log1p(
        XY_test_norm[:, id_adt+config.dim_input_arr[0]]/np.sum(
            XY_test_norm[:, id_adt+config.dim_input_arr[0]], axis=1, keepdims=True)*l_adt)
    dataset_test = tf.data.Dataset.from_tensor_slices((
        _XY_test.astype(tf.keras.backend.floatx()),
        model.cat_enc.transform(np.zeros((_XY_test.shape[0],1))).toarray().astype(np.float32),
        np.zeros(_XY_test.shape[0]).astype(np.int32)
    )).batch(512).prefetch(tf.data.experimental.AUTOTUNE)
    
    path = path_root+name+'/'
    os.makedirs(path, exist_ok=True)
    
    evaluate(model.vae, dataset_test, X_test, Y_test, XY_test_norm,
             gene_names, ADT_names, id_rna, id_adt, l_rna, l_adt, path)