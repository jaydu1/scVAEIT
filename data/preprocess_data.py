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


with h5py.File('data/DOGMA_pbmc.h5', 'r') as f:
    print(f.keys())
    peak_names_dogma = np.array(f['peak_names'], dtype='S32').astype(str)
    ADT_names_dogma = np.array(f['ADT_names'], dtype='S32').astype(str)
    gene_names_dogma = np.array(f['gene_names'], dtype='S32').astype(str)
    X_dogma = sp.sparse.csc_matrix(
        (np.array(f['RNA.data'], dtype=np.float32), 
         np.array(f['RNA.indices'], dtype=np.int32),
         np.array(f['RNA.indptr'], dtype=np.int32)
        ), 
        shape = np.array(f['RNA.shape'], dtype=np.int32)).tocsc().astype(np.float32).T.toarray()
    Y_dogma = np.array(f['ADT'], dtype=np.float32)
    Z_dogma = sp.sparse.csc_matrix(
        (np.array(f['peaks.data'], dtype=np.float32),
         np.array(f['peaks.indices'], dtype=np.int32),
         np.array(f['peaks.indptr'], dtype=np.int32)
        ), 
        shape = np.array(f['peaks.shape'], dtype=np.int32)).tocsc().astype(np.float32).T.toarray()
    celltypes_dogma = np.array(f['predicted.celltypes'], dtype='S32').astype(str)
    batches_dogma = np.array(f['batches'], dtype=np.float32)
    cell_ids_dogma = np.array(f['cell_ids'], dtype='S32').astype(str)

print(
X_dogma.shape, Y_dogma.shape, Z_dogma.shape
)

with h5py.File('data/cite_pbmc.h5', 'r') as f:
    print(f.keys())
    ADT_names_cite = np.array(f['ADT_names'], dtype='S32').astype(str)
    gene_names_cite = np.array(f['gene_names'], dtype='S32').astype(str)
    X_cite = sp.sparse.csc_matrix(
        (np.array(f['RNA.data'], dtype=np.float32), 
         np.array(f['RNA.indices'], dtype=np.int32),
         np.array(f['RNA.indptr'], dtype=np.int32)
        ), 
        shape = np.array(f['RNA.shape'], dtype=np.int32)).tocsc().astype(np.float32).T.toarray()
    Y_cite = np.array(f['ADT'], dtype=np.float32)
    batches_cite = np.array(f['batches'], dtype=np.float32)
    celltypes_cite = np.array(f['celltypes'], dtype='S32').astype(str)
    cell_ids_cite = np.array(f['cell_ids'], dtype='S32').astype(str)
print(
X_cite.shape, Y_cite.shape
)

with h5py.File('data/asap_pbmc.h5', 'r') as f:
    print(f.keys())
    ADT_names_asap = np.array(f['ADT_names'], dtype='S32').astype(str)
    peak_names_asap = np.array(f['peak_names'], dtype='S32').astype(str)
    Y_asap = np.array(f['ADT'], dtype=np.float32)
    Z_asap = sp.sparse.csc_matrix(
        (np.array(f['peaks.data'], dtype=np.float32),
         np.array(f['peaks.indices'], dtype=np.int32),
         np.array(f['peaks.indptr'], dtype=np.int32)
        ), 
        shape = np.array(f['peaks.shape'], dtype=np.int32)).tocsc().astype(np.float32).T.toarray()
    batches_asap = np.array(f['batches'], dtype=np.float32)
    celltypes_asap = np.array(f['celltypes'], dtype='S32').astype(str)
    cell_ids_asap = np.array(f['cell_ids'], dtype='S32').astype(str)
print(
Y_asap.shape, Z_asap.shape
)


gene_names_dogma = np.char.upper(np.char.replace(np.char.replace(gene_names_dogma, '_', '-'), '.', '-'))
gene_names_cite = np.char.upper(np.char.replace(np.char.replace(gene_names_cite, '_', '-'), '.', '-'))
gene_names = np.union1d(gene_names_dogma, gene_names_cite)

id_X_dogma = np.array([np.where(gene_names==i)[0][0] for i in gene_names_dogma])
id_X_cite = np.array([np.where(gene_names==i)[0][0] for i in gene_names_cite])

ADT_names = np.union1d(np.union1d(ADT_names_cite, ADT_names_dogma), ADT_names_asap)

id_Y_dogma = np.array([np.where(ADT_names==i)[0][0] for i in ADT_names_dogma])
id_Y_cite = np.array([np.where(ADT_names==i)[0][0] for i in ADT_names_cite])
id_Y_asap = np.array([np.where(ADT_names==i)[0][0] for i in ADT_names_asap])


peak_names = np.union1d(peak_names_dogma, peak_names_asap)
peak_names = np.concatenate(
    [peak_names[np.char.startswith(peak_names, 'chr%d-'%i)] for i in range(1,23)])

id_Z_dogma = np.array([np.where(peak_names==i)[0][0] for i in peak_names_dogma])
id_Z_asap = np.array([np.where(peak_names==i)[0][0] for i in peak_names_asap])

n_cells = Y_dogma.shape[0]+Y_cite.shape[0]+Y_asap.shape[0]
X = np.zeros((n_cells, len(gene_names)))
Y = np.zeros((n_cells, len(ADT_names)))
Z = np.zeros((n_cells, len(peak_names)))

X[:Y_dogma.shape[0],id_X_dogma] = X_dogma
X[Y_dogma.shape[0]:-Y_asap.shape[0],id_X_cite] = X_cite
Y[:Y_dogma.shape[0],id_Y_dogma] = Y_dogma
Y[Y_dogma.shape[0]:-Y_asap.shape[0],id_Y_cite] = Y_cite
Y[-Y_asap.shape[0]:,id_Y_asap] = Y_asap
Z[:Y_dogma.shape[0],id_Z_dogma] = Z_dogma
Z[-Y_asap.shape[0]:,id_Z_asap] = Z_asap


batches = np.r_[batches_dogma, batches_cite, batches_asap]
batches = np.c_[batches, np.repeat(np.arange(3), [Y_dogma.shape[0],Y_cite.shape[0],Y_asap.shape[0]])]

cell_types = np.r_[celltypes_dogma, celltypes_cite, celltypes_asap]
cell_ids = np.r_[cell_ids_dogma, cell_ids_cite, cell_ids_asap]

X = sp.sparse.csc_matrix(X)
Z = sp.sparse.csc_matrix(Z)

with h5py.File('data/dogma_cite_asap.h5', 'w') as hf:
    dt = h5py.special_dtype(vlen=str) 
    
    hf.create_dataset('ADT_names', data=ADT_names.astype(dt), compression="gzip", compression_opts=9)
    hf.create_dataset('gene_names', data=gene_names.astype(dt), compression="gzip", compression_opts=9)
    hf.create_dataset('peak_names', data=peak_names.astype(dt), compression="gzip", compression_opts=9)
    
    hf.create_dataset('cell_types', data=cell_types.astype(dt), compression="gzip", compression_opts=9)
    hf.create_dataset('cell_ids', data=cell_ids.astype(dt), compression="gzip", compression_opts=9)
    hf.create_dataset('batches', data=batches, compression="gzip", compression_opts=9)    
    
    hf.create_dataset('RNA.data', data=X.data, compression="gzip", compression_opts=9)
    hf.create_dataset('RNA.indices', data=X.indices, compression="gzip", compression_opts=9)
    hf.create_dataset('RNA.indptr', data=X.indptr, compression="gzip", compression_opts=9)
    hf.create_dataset('RNA.shape', data=X.shape, compression="gzip", compression_opts=9)
    
    hf.create_dataset('ADT', data=Y, compression="gzip", compression_opts=9)
    
    hf.create_dataset('peaks.data', data=Z.data, compression="gzip", compression_opts=9)
    hf.create_dataset('peaks.indices', data=Z.indices, compression="gzip", compression_opts=9)
    hf.create_dataset('peaks.indptr', data=Z.indptr, compression="gzip", compression_opts=9)
    hf.create_dataset('peaks.shape', data=Z.shape, compression="gzip", compression_opts=9)
    
    hf.create_dataset('id_X_dogma', data=id_X_dogma, compression="gzip", compression_opts=9)
    hf.create_dataset('id_X_cite', data=id_X_cite, compression="gzip", compression_opts=9)
    hf.create_dataset('id_Y_dogma', data=id_Y_dogma, compression="gzip", compression_opts=9)
    hf.create_dataset('id_Y_cite', data=id_Y_cite, compression="gzip", compression_opts=9)
    hf.create_dataset('id_Y_asap', data=id_Y_asap, compression="gzip", compression_opts=9)
    hf.create_dataset('id_Z_dogma', data=id_Z_dogma, compression="gzip", compression_opts=9)
    hf.create_dataset('id_Z_asap', data=id_Z_asap, compression="gzip", compression_opts=9)

    hf.create_dataset('sample_sizes', data=np.array([Y_dogma.shape[0],Y_cite.shape[0],Y_asap.shape[0]]), compression="gzip", compression_opts=9)
    