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

import tensorflow as tf
import tensorflow_probability as tfp

physical_devices = tf.config.list_physical_devices('GPU')
try:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
except:
    # Invalid device or cannot modify virtual devices once initialized.
    pass

from tensorflow.keras.layers import Layer, Dense, BatchNormalization, LeakyReLU
from tensorflow.keras.utils import Progbar

tfd = tfp.distributions

import matplotlib.pyplot as plt
from types import SimpleNamespace
from sklearn.model_selection import train_test_split


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

id_peak_XY = ~(np.char.startswith(peak_names, 'chrX') | np.char.startswith(peak_names, 'chrY'))
Z = Z[:,id_peak_XY]
peak_names = peak_names[id_peak_XY]
chunk_atac = np.array([
    np.sum(np.char.startswith(peak_names, 'chr%d-'%i)) for i in range(1,23)
    ], dtype=np.int32)

print(
X.shape, Y.shape, Z.shape
)
    

cell_type_list =  ['CD4 T', 'CD8 T', 'Mono', 'B', 'DC', 'NK']
cell_type_test = cell_type_list[int(sys.argv[1])]
print(cell_type_test)
path_root = 'result/ex2/%s/'%cell_type_test

dim_input_arr = [X.shape[1], Y.shape[1], Z.shape[1]]
config = {
    'dim_input_arr': dim_input_arr,
    'dimensions':[256], 
    'dim_latent':32,
    'dim_block': np.append([len(gene_names),len(ADT_names)], chunk_atac), 
    'dist_block':['NB','NB'] + ['Bernoulli' for _ in chunk_atac], 
    'dim_block_enc':np.array([256, 128] + [16 for _ in chunk_atac]),
    'dim_block_dec':np.array([256, 128] + [16 for _ in chunk_atac]),
    'dim_block_embed':np.array([32, 16] + [2 for _ in chunk_atac]),
    
    'block_names':np.array(['rna', 'adt'] + ['atac' for _ in range(len(chunk_atac))]),
    'uni_block_names':np.array(['rna','adt','atac']),
    
    'beta_kl':1.,
    'beta_unobs':2./3.,
    'beta_modal':np.array([0.14,0.85,0.01]),
    'beta_reverse':0.,

    "p_feat" : 0.2,
    "p_modal" : np.ones(3)/3,
    
}
config = SimpleNamespace(**config)
n_samples = 50


# preprocess
Z = (Z>0).astype(np.float32)

data_raw = np.c_[X/np.sum(X, axis=1, keepdims=True)*1e4, Y/np.sum(Y, axis=1, keepdims=True)*1e4, Z]
data_raw_test = data_raw[cell_types==cell_type_test]
data_train_norm = data_raw[cell_types!=cell_type_test]

X = np.log(X/np.sum(X, axis=1, keepdims=True)*1e4+1.)
Y = np.log(Y/np.sum(Y, axis=1, keepdims=True)*1e4+1.)

# data spliting
data = np.c_[X, Y, Z].astype(np.float32)
batches = np.c_[batches, np.zeros((data.shape[0],1))].astype(np.float32)
print(np.sum(cell_types!=cell_type_test))

from scVAEIT.VAEIT import scVAEIT
masks = np.zeros((1,data.shape[1]), dtype=np.float32)
model = scVAEIT(config, data[cell_types!=cell_type_test], masks, batches[cell_types!=cell_type_test])


mask_adt = np.zeros((1,data.shape[1]), dtype=np.float32)
mask_adt[:,config.dim_input_arr[0]:-config.dim_input_arr[2]] = -1.
dataset_adt = tf.data.Dataset.from_tensor_slices(
        (data[cell_types==cell_type_test], 
         model.cat_enc.transform(batches[cell_types==cell_type_test]).toarray().astype(np.float32),
         np.zeros(np.sum(cell_types==cell_type_test)).astype(np.int32)
        )
    ).batch(512).prefetch(tf.data.experimental.AUTOTUNE)
Y_test = Y[cell_types==cell_type_test,:].copy()
batch_test = batches[cell_types==cell_type_test]
del X, Y, Z, data


model.train(
    valid=False, num_epoch=300, batch_size=512,  save_every_epoch=50,
    verbose=True, checkpoint_dir=path_root+'checkpoint/'
)


############################################################
#
# Test on the hold-out cell type
#
############################################################
def comp_cor_flatten(x, y):
    pearson_r, pearson_p = scipy.stats.pearsonr(x, y)
    print(f"Found pearson's correlation/p of {pearson_r:.4f}/{pearson_p:.4g}")
    spearman_corr, spearman_p = scipy.stats.spearmanr(x, y)
    print(f"Found spearman's collelation/p of {spearman_corr:.4f}/{spearman_p:.4g}")
    return pearson_r, spearman_corr

def evaluate(vae, dataset_test, X_test, Y_test, Z_test):
    mask_rna = np.zeros((1,np.sum(vae.config.dim_input_arr)), dtype=np.float32)
    mask_rna[:,:vae.config.dim_input_arr[0]] = -1.
    recon = vae.get_recon(dataset_test, mask_rna)
    X_hat = recon[:, :vae.config.dim_input_arr[0]]    
    
    res = []
    
    for batch in [0,1]:
        _X_hat = X_hat[batch_test[:,0]==batch,:]
        _X_test = X_test[batch_test[:,0]==batch,:]
        _df = pd.DataFrame({
            'RNA':gene_names,
            'Pearson r':[np.corrcoef(_X_hat[:,i], _X_test[:,i])[0,1] for i in np.arange(len(gene_names))],
            'Spearman r':[scipy.stats.spearmanr(_X_hat[:,i], _X_test[:,i])[0] for i in np.arange(len(gene_names))],
            'MSE':np.mean((_X_hat-_X_test)**2, axis=0)
        })
        print(np.quantile(_df['MSE'].values, [0.,0.5,1.0]))
        _df.to_csv(path_root+'res_scVAEIT_RNA_%d.csv'%(int(batch)))    
        
        pr, sr = comp_cor_flatten(_X_test.flatten(), _X_hat.flatten())
        mse = np.mean((_X_test.flatten()-_X_hat.flatten())**2)
        res.append(['RNA', batch, pr, sr, mse])
    
    
    mask_adt = np.zeros((1,np.sum(vae.config.dim_input_arr)), dtype=np.float32)
    mask_adt[:,vae.config.dim_input_arr[0]:-vae.config.dim_input_arr[-1]] = -1.
    recon = vae.get_recon(dataset_test, mask_adt)
    Y_hat = recon[:, vae.config.dim_input_arr[0]:-vae.config.dim_input_arr[-1]]

    if cell_type_test=='CD4 T':
        pd.DataFrame(X_hat).to_csv(path_root+'scVAEIT_RNA_pred.csv')
        pd.DataFrame(Y_hat).to_csv(path_root+'scVAEIT_ADT_pred.csv')

    for batch in [0,1]:
        _Y_hat = Y_hat[batch_test[:,0]==batch,:]
        _Y_test = Y_test[batch_test[:,0]==batch,:]
        _df = pd.DataFrame({
            'ADT':ADT_names,
            'Pearson r':[np.corrcoef(_Y_hat[:,i], _Y_test[:,i])[0,1] for i in np.arange(len(ADT_names))],
            'Spearman r':[scipy.stats.spearmanr(_Y_hat[:,i], _Y_test[:,i])[0] for i in np.arange(len(ADT_names))],
            'MSE':np.mean((_Y_hat-_Y_test)**2, axis=0)
        })
        print(np.quantile(_df['Pearson r'].values, [0.,0.5,1.0]))
        _df.to_csv(path_root+'res_scVAEIT_ADT_%d.csv'%(int(batch)))
        
        pr, sr = comp_cor_flatten(_Y_test.flatten(), _Y_hat.flatten())
        mse = np.mean((_Y_test.flatten()-_Y_hat.flatten())**2)
        res.append(['ADT', batch, pr, sr, mse])
        
    mask_atac = np.zeros((1,np.sum(vae.config.dim_input_arr)), dtype=np.float32)
    mask_atac[:,-vae.config.dim_input_arr[-1]:] = -1.
    recon = vae.get_recon(dataset_test, mask_atac)
    Z_hat = recon[:, -vae.config.dim_input_arr[-1]:]
    for batch in [0,1]:
        _Z_hat = Z_hat[batch_test[:,0]==batch,:]
        _Z_test = Z_test[batch_test[:,0]==batch,:]
        _df = pd.DataFrame({
            'ATAC':peak_names,
            'AUC':[np.nan if len(_Z_test[:,i])==np.sum(_Z_test[:,i]==0) or len(_Z_test[:,i])==np.sum(_Z_test[:,i]==1)
                         else roc_auc_score(_Z_test[:,i], _Z_hat[:,i]) for i in np.arange(len(peak_names))],
            'Accuracy':np.mean(_Z_test==(_Z_hat>0.5), axis=0),
            'AR':[balanced_accuracy_score(_Z_test[:,i], _Z_hat[:,i]>0.5) for i in np.arange(len(peak_names))],
            'AP':[average_precision_score(_Z_test[:,i], _Z_hat[:,i]) for i in np.arange(len(peak_names))],
            'BCE':[log_loss(_Z_test[:,i], _Z_hat[:,i], labels=np.arange(2)) for i in np.arange(len(peak_names))],
            'MSE':np.mean((_Z_hat-_Z_test)**2, axis=0)
        })
        print(np.quantile(_df['MSE'].values, [0.,0.5,1.0]))
        _df.to_csv(path_root+'res_scVAEIT_ATAC_%d.csv'%(int(batch)))

        auc = roc_auc_score(_Z_test.flatten(), _Z_hat.flatten())
        acc = np.mean(_Z_test==(_Z_hat>0.5))
        mse = np.mean((_Z_hat-_Z_test)**2)
        print(auc, acc)
        res.append(['ATAC', batch, auc, acc, mse])

    pd.DataFrame(res, 
        columns=['Target', 'Batch', 'Metric_1', 'Metric_2', 'MSE']).to_csv(path_root+'res_scVAEIT_overall.csv')


evaluate(model.vae, dataset_test, X_test, Y_test, Z_test)



############################################################
#
# Test on the hold-out cell type with random missing
#
############################################################
def comp_cor_flatten(x, y):
    pearson_r, pearson_p = scipy.stats.pearsonr(x, y)
    print(f"Found pearson's correlation/p of {pearson_r:.4f}/{pearson_p:.4g}")
    spearman_corr, spearman_p = scipy.stats.spearmanr(x, y)
    print(f"Found spearman's collelation/p of {spearman_corr:.4f}/{spearman_p:.4g}")
    return pearson_r, spearman_corr


def evaluate(vae, dataset_test, mask, X_test, Y_test, Z_test, 
             gene_names, ADT_names, peak_names, id_rna_test, id_adt_test, id_atac_test, p, i):

    recon = vae.get_recon(dataset_test, mask)
    X_hat = recon[:, :vae.config.dim_input_arr[0]]
    Y_hat = recon[:, vae.config.dim_input_arr[0]:-vae.config.dim_input_arr[-1]]
    Z_hat = recon[:, -vae.config.dim_input_arr[-1]:]
    
    res = []    
    for batch in [0,1]:

        _X_hat = X_hat[batch_test[:,0]==batch,:][:,id_rna_test]
        _X_test = X_test[batch_test[:,0]==batch,:][:,id_rna_test]
        _df = pd.DataFrame({
            'RNA':gene_names[id_rna_test],
            'Pearson r':[np.corrcoef(_X_hat[:,i], _X_test[:,i])[0,1] for i in range(len(id_rna_test))],
            'Spearman r':[scipy.stats.spearmanr(_X_hat[:,i], _X_test[:,i])[0] for i in range(len(id_rna_test))],
            'MSE':np.mean((_X_hat-_X_test)**2, axis=0)
        })
        print(np.quantile(_df['Pearson r'].values, [0.,0.5,1.0]))
        _df.to_csv(path_root+'res_scVAEIT_masked_RNA_%.01f_%d_%d.csv'%(p,batch,i))
        _X_hat = _X_hat.flatten()
        _X_test = _X_test.flatten()
        res.append([i, batch, 'RNA',
                    np.corrcoef(_X_hat, _X_test)[0,1],
                    scipy.stats.spearmanr(_X_hat, _X_test)[0],
                    np.mean((_X_hat-_X_test)**2)
                   ])
        

        _Y_hat = Y_hat[batch_test[:,0]==batch,:][:,id_adt_test]
        _Y_test = Y_test[batch_test[:,0]==batch,:][:,id_adt_test]
        _df = pd.DataFrame({
            'ADT':ADT_names[id_adt_test],
            'Pearson r':[np.corrcoef(_Y_hat[:,i], _Y_test[:,i])[0,1] for i in range(len(id_adt_test))],
            'Spearman r':[scipy.stats.spearmanr(_Y_hat[:,i], _Y_test[:,i])[0] for i in range(len(id_adt_test))],
            'MSE':np.mean((_Y_hat-_Y_test)**2, axis=0)
        })        
        print(np.quantile(_df['Pearson r'].values, [0.,0.5,1.0]))
        _df.to_csv(path_root+'res_scVAEIT_masked_ADT_%.01f_%d_%d.csv'%(p,batch,i))
        _Y_hat = _Y_hat.flatten()
        _Y_test = _Y_test.flatten()
        res.append([i, batch, 'ADT',
                    np.corrcoef(_Y_hat, _Y_test)[0,1],
                    scipy.stats.spearmanr(_Y_hat, _Y_test)[0],
                    np.mean((_Y_hat-_Y_test)**2)
                   ])
        
        _Z_hat = Z_hat[batch_test[:,0]==batch,:][:,id_atac_test]
        _Z_test = Z_test[batch_test[:,0]==batch,:][:,id_atac_test]
        _df = pd.DataFrame({
            'ATAC':peak_names[id_atac_test],
            'AUC':[np.nan if len(_Z_test[:,i])==np.sum(_Z_test[:,i]==0) or len(_Z_test[:,i])==np.sum(_Z_test[:,i]==1)
                         else roc_auc_score(_Z_test[:,i], _Z_hat[:,i]) for i in range(len(id_atac_test))],            
            'BCE':[log_loss(_Z_test[:,i], _Z_hat[:,i], labels=np.arange(2)) for i in range(len(id_atac_test))],
            'MSE':np.mean((_Z_hat-_Z_test)**2, axis=0)
        })
        print(np.quantile(_df['MSE'].values, [0.,0.5,1.0]))
        _df.to_csv(path_root+'res_scVAEIT_masked_ATAC_%.01f_%d_%d.csv'%(p,batch,i))
        _Z_hat = _Z_hat.flatten()
        _Z_test = _Z_test.flatten()
        res.append([i, batch, 'ATAC',
                    roc_auc_score(_Z_test, _Z_hat),
                    log_loss(_Z_test, _Z_hat),
                    np.mean((_Z_hat-_Z_test)**2)
                   ])        

    return res



# np.random.seed(0)
# for ip in np.arange(1,9):
#     p = ip/10.
#     for i in tqdm(range(10)):
#         mask = np.zeros((1, config.dim_input), dtype=np.float32)
#         mask[0,np.random.permutation(np.arange(config.dim_input_arr[0]))[:int(config.dim_input_arr[0]*p)]] = 1.
#         mask[0,config.dim_input_arr[0]+np.random.permutation(np.arange(config.dim_input_arr[1]))[:int(config.dim_input_arr[1]*p)]] = 1.
#         mask[0,config.dim_input_arr[0]+config.dim_input_arr[1]+
#              np.random.permutation(np.arange(config.dim_input_arr[2]))[:int(config.dim_input_arr[2]*p)]] = 1.
#         mask_list.append(mask)
# pd.DataFrame(
#        np.concatenate(mask_list), columns=np.r_[gene_names, ADT_names, peak_names]
#    ).to_csv('data/mask_dogma.csv')
mask_list = pd.read_csv('data/mask_dogma.csv', index_col=0).values
res = []
for ip in np.arange(1,9):
    p = ip/10.
    for i in range(10):
        
        mask = mask_list[(10*(ip-1)+i):(10*(ip-1)+i+1),:] 

        id_rna = np.where(mask[0,:config.dim_input_arr[0]]==0)[0]
        id_adt = np.where(mask[0,config.dim_input_arr[0]:-config.dim_input_arr[2]]==0)[0] 
        id_atac = np.where(mask[0,-config.dim_input_arr[2]:]==0)[0]
        id_rna_test = np.where(mask[0,:config.dim_input_arr[0]]==1)[0]
        id_adt_test = np.where(mask[0,config.dim_input_arr[0]:-config.dim_input_arr[2]]==1)[0] 
        id_atac_test = np.where(mask[0,-config.dim_input_arr[2]:]==1)[0]        
        l_rna = np.median(np.sum(data_train_norm[:, id_rna], axis=-1))
        l_adt = np.median(np.sum(data_train_norm[:, id_adt + config.dim_input_arr[0]], axis=-1))

        data_test_sub = data_raw_test.copy()
        data_test_sub[:, id_rna] = np.log(
            data_test_sub[:, id_rna]/np.sum(data_test_sub[:, id_rna], axis=1, keepdims=True)*l_rna+1.)
        data_test_sub[:, id_adt + config.dim_input_arr[0]] = np.log(
            data_test_sub[:, id_adt + config.dim_input_arr[0]]/np.sum(data_test_sub[:, id_adt + config.dim_input_arr[0]], axis=1, keepdims=True)*l_adt+1.)

        dataset_test_sub = tf.data.Dataset.from_tensor_slices((
            data_test_sub.astype(tf.keras.backend.floatx()),
            model.cat_enc.transform(batch_test).toarray().astype(np.float32),
            np.zeros(np.sum(cell_types==cell_type_test)).astype(np.int32)
            )
        ).batch(512).prefetch(tf.data.experimental.AUTOTUNE)

        res += evaluate(model.vae, dataset_test_sub, mask, X_test, Y_test, Z_test, 
                        gene_names, ADT_names, peak_names, id_rna_test, id_adt_test, id_atac_test, p, i)

pd.DataFrame(
    res, columns=['i', 'Batch', 'Target', 'Metric 1', 'Metric 2', 'MSE']
).to_csv(path_root+'res_scVAEIT_masked.csv')
