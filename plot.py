##############################################################################
#
# Ex1: Figure 2 and Figure S1
#
##############################################################################

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

from tensorflow.keras.layers import Layer, Dense, BatchNormalization, LeakyReLU
from tensorflow.keras.utils import Progbar

tfd = tfp.distributions

import matplotlib.pyplot as plt

import seaborn as sns
sns.set_theme()


df = pd.DataFrame()
for method in ['scVAEIT', 'Seurat', 'totalVI']:
    _df = pd.read_csv('result/ex1/Mono/{}res_{}_overall.csv'.format('' if method=='scVAEIT' else method+'/', method),
                     names=['Source', 'Target','Pearson r','Spearman r','MSE'], skiprows=1)
    _df['Method'] = method
    df = pd.concat([df,_df])
df = df.reset_index(drop=True)
df['RMSE'] = np.sqrt(df['MSE'])
df[df['Source']!=df['Target']]


'''
Fig2a table
'''
for source in ['RNA', 'ADT']:
    for target in ['RNA', 'ADT']:
        if source==target:
            continue
        for metric in ['Pearson r', 'Spearman r', 'RMSE']:
            _df = df[df['Source']!=df['Target']]
            print(' & '.join(
                [source, target, metric] + ['%.02f'%i for i in _df[(_df['Source']==source)&(_df['Target']==target)][metric].values
                 ]
            ))


'''
Fig2b violin plot
'''
import matplotlib as mpl
from matplotlib.colors import to_rgb
from matplotlib.legend_handler import HandlerTuple
target = 'ADT'
df = pd.DataFrame()
data_name = ['CITE-seq CBMC', 'REAP-seq PBMC']
for i,name in enumerate(['cbmc', 'reap']):    
    for method in ['scVAEIT', 'Seurat', 'totalVI']:
        _df = pd.read_csv('result/ex1/{}/res_{}_{}.csv'.format(
            name, method, target), index_col=0)
        _df['Method'] = method
        _df['Dataset'] = data_name[i]
        df = df.append(_df)
df = df.reset_index(drop=True)
df['RMSE'] = np.sqrt(df['MSE'])
df = df.fillna(0.)
df_overall = df.groupby('Method').median()


fig, axes = plt.subplots(1,3, figsize=(16,4), sharey=False)
plt.tight_layout()
axes[0] = sns.violinplot(x="Method", y="Pearson r", hue="Dataset",
                    data=df, cut=0., palette="muted", ax=axes[0])
axes[0].set_xticklabels(['scVAEIT', 'Seurat', 'totalVI'])
axes[0].tick_params(axis='y', which='major', labelsize=8)
colors = sns.color_palette()
handles= []
for i, violin in enumerate(axes[0].findobj(mpl.collections.PolyCollection)):
    rgb = to_rgb(colors[i // 2])
    if i % 2 == 0:
        rgb = 0.5 + 0.5 * np.array(rgb)
    violin.set_facecolor(rgb)
    handles.append(plt.Rectangle((0, 0), 0, 0, facecolor=rgb, edgecolor='black'))
axes[0].legend(handles=[tuple(handles[::2]), tuple(handles[1::2])], 
               labels=data_name, title="Dataset", 
               handlelength=4, handler_map={tuple: HandlerTuple(ndivide=None, pad=0)}, loc='lower left')
    
axes[1] = sns.violinplot(x="Method", y="Spearman r", hue="Dataset",
                    data=df, cut=0., palette="muted", ax=axes[1])
axes[1].set_xticklabels(['scVAEIT', 'Seurat', 'totalVI'])   
axes[1].tick_params(axis='y', which='major', labelsize=8)
handles= []
for i, violin in enumerate(axes[1].findobj(mpl.collections.PolyCollection)):
    rgb = to_rgb(colors[i // 2])
    if i % 2 == 0:
        rgb = 0.5 + 0.5 * np.array(rgb)
    violin.set_facecolor(rgb)
    handles.append(plt.Rectangle((0, 0), 0, 0, facecolor=rgb, edgecolor='black'))
axes[1].legend(handles=[tuple(handles[::2]), tuple(handles[1::2])], 
               labels=data_name, title="Dataset", 
               handlelength=4, handler_map={tuple: HandlerTuple(ndivide=None, pad=0)}, loc='lower left')
    
axes[2] = sns.violinplot(x="Method", y="RMSE", hue="Dataset",
                    data=df, cut=0., palette="muted", ax=axes[2])
axes[2].set_xticklabels(['scVAEIT', 'Seurat', 'totalVI'])    
axes[2].tick_params(axis='y', which='major', labelsize=8)   
handles= []
for i, violin in enumerate(axes[2].findobj(mpl.collections.PolyCollection)):
    rgb = to_rgb(colors[i // 2])
    if i % 2 == 0:
        rgb = 0.5 + 0.5 * np.array(rgb)
    violin.set_facecolor(rgb)
    handles.append(plt.Rectangle((0, 0), 0, 0, facecolor=rgb, edgecolor='black'))
axes[2].legend(handles=[tuple(handles[::2]), tuple(handles[1::2])], 
               labels=data_name, title="Dataset", 
               handlelength=4, handler_map={tuple: HandlerTuple(ndivide=None, pad=0)}, loc='upper left')
plt.savefig('result/Fig2b.png', dpi=300, bbox_inches='tight', pad_inches=0)


'''
Fig2c & FigS1 scatter plot
'''
data = 'cbmc'

df = pd.DataFrame()
for method in ['scVAEIT', 'Seurat', 'totalVI']:
    _df = pd.read_csv('result/ex1/{}/res_{}_ADT.csv'.format(data, method), index_col=0)
    _df['Method'] = method
    df = df.append(_df)

sns.set_theme()
fig, axes = plt.subplots(2, 1, figsize=(4,8), dpi=300, sharex=True)
_df = df.pivot(index='ADT', columns='Method', values=['Pearson r','Spearman r'])['Pearson r']
sns.scatterplot(data=_df, x="scVAEIT", y="Seurat", ax=axes[0], s=80)
axes[0].set_xlim([-0.25,1])
axes[0].set_ylim([-0.25,1])
axes[0].plot([-0.2, 1.], [-0.2, 1.], '--', c='gray', label='$y=x$')

for i, point in _df.iterrows():
    if i=='CD3-1':
        axes[0].text(point['scVAEIT'] - 0.02, point['Seurat']+.02,  str(point.name))
    elif i=='CD34':        
        axes[0].text( point['scVAEIT'] - 0.1, point['Seurat']-.1, str(point.name))
    elif i=='CD14':        
        axes[0].text(point['scVAEIT'], point['Seurat']+.02,  str(point.name))        
    else:
        axes[0].text(point['scVAEIT']+0.02, point['Seurat']+.02,  str(point.name))
        
        
sns.scatterplot(data=_df, x="scVAEIT",  y="totalVI", ax=axes[1], s=80)
axes[1].set_xlim([-0.25,1])
axes[1].set_ylim([-0.25,1])
axes[1].plot([-0.2, 1.], [-0.2, 1.], '--', c='gray', label='$y=x$')

for i, point in _df.iterrows():
    if i=='CD3-1':
        axes[1].text(point['scVAEIT'] - 0.03, point['totalVI']+.02, str(point.name))
    elif i=='CD34':        
        axes[1].text(point['scVAEIT'] + 0.05, point['totalVI']+.02, str(point.name))  
    elif i=='CD45RA':
        axes[1].text(point['scVAEIT'] - 0.05, point['totalVI']+.05, str(point.name))  
    else:
        axes[1].text(point['scVAEIT']+0.02, point['totalVI']+.02, str(point.name))
                        
plt.savefig('result/Fig2c.png', dpi=300, bbox_inches='tight', pad_inches=0)


data = 'reap'

df = pd.DataFrame()
for method in ['scVAEIT', 'Seurat', 'totalVI']:
    _df = pd.read_csv('result/ex1/{}/res_{}_ADT.csv'.format(data, method), index_col=0)
    _df['Method'] = method
    df = df.append(_df)

sns.set_theme()
fig, axes = plt.subplots(1,2, figsize=(10,4), dpi=300, sharey=True)
_df = df.pivot(index='ADT', columns='Method', values=['Pearson r','Spearman r'])['Pearson r']
sns.scatterplot(data=_df, x="Seurat", y="scVAEIT", ax=axes[0], s=80)
axes[0].set_xlim([-0.25,1])
axes[0].set_ylim([-0.25,1])
axes[0].plot([-0.2, 1.], [-0.2, 1.], '--', c='gray', label='$y=x$')
        
sns.scatterplot(data=_df, x="totalVI", y="scVAEIT", ax=axes[1], s=80)
axes[1].set_xlim([-0.25,1])
axes[1].set_ylim([-0.25,1])
axes[1].plot([-0.2, 1.], [-0.2, 1.], '--', c='gray', label='$y=x$')

plt.savefig('result/FigS1.png', dpi=300, bbox_inches='tight', pad_inches=0)



##############################################################################
#
# Ex2: Figure 3, Figure S2, and Figure S3
#
##############################################################################

'''
Fig3 imputation
'''
df = pd.DataFrame()
for celltype in ['CD4 T', 'CD8 T', 'B', 'NK', 'DC', 'Mono']:
    for method in ['scVAEIT', 'Seurat', 'totalVI', 'MultiVI']:
        _df = pd.read_csv('result/ex2/{}/{}res_{}_overall.csv'.format(
            celltype,
            '' if method=='scVAEIT' else method+'/', method),
                         names=['','Target','Batch','Metric 1','Metric 2','MSE'], 
                          skiprows=1, index_col=0, dtype={'MSE':np.float32}
                         )
        _df['Method'] = method
        _df['Celltype'] = celltype
        df = pd.concat([df,_df])
    df = df.reset_index(drop=True)
    df['RMSE'] = np.sqrt(df['MSE'])
df.sort_values(['Target','Batch'])



sns.set_theme()
sns.set(font_scale = 1.4)

name_dict = {j:i for i,j in enumerate(['scVAEIT', 'Seurat', 'totalVI', 'MultiVI'])}
celltype_list = ['CD4 T', 'CD8 T', 'B', 'NK', 'DC', 'Mono']
metric_list = ['Pearson r','Spearman r','RMSE']
batch_list = ['Control', 'Stimulation']
cmap_list = [sns.color_palette("Blues"), sns.color_palette("Greens"), sns.color_palette("YlOrBr")]

for target in ['RNA', 'ADT']:
    fig, axes = plt.subplots(2,3,
                             sharex=True, sharey=True, figsize=(18,6)#, gridspec_kw=dict(width_ratios=[4,1,0.2])
                            )
    for batch in [0,1]:
        for j,metric in enumerate(['Metric 1','Metric 2','RMSE']):
            _df = df[(df['Target']==target)&(df['Batch']==batch)]
            _df = _df[['Method','Celltype',metric]].pivot(
                index='Method',columns='Celltype').sort_index(
                key=lambda x: [name_dict[j] for j in x]
            )
            axes[batch,j] = sns.heatmap(_df, annot=True, ax=axes[batch,j], cmap=cmap_list[j])
            if j==0:
                axes[batch,j].set_ylabel(batch_list[batch])
            else:
                axes[batch,j].set_ylabel('')

            axes[batch,j].set_xlabel('')
            if batch==0:
                axes[batch,j].set_title(metric_list[j])
            else:
                if j==1:
                    axes[batch,j].set_xlabel('Cell Type')

                axes[batch,j].set_xticks(np.arange(len(celltype_list))+0.5, labels=celltype_list, rotation=0)

    plt.tight_layout()
    name = 'Fig3a' if target=='RNA' else 'FigS2'
    plt.savefig('result/{}.png'.format(name), dpi=300, bbox_inches='tight', pad_inches=0) 


metric_list = ['AUC','BCE','RMSE']
target = 'ATAC'
fig, axes = plt.subplots(2,3,
                         sharex=True, sharey=True, figsize=(18,6)#, gridspec_kw=dict(width_ratios=[4,1,0.2])
                        )
for batch in [0,1]:
    for j,metric in enumerate(['Metric 1','Metric 2','RMSE']):
        _df = df[(df['Target']==target)&(df['Batch']==batch)]
        _df = _df[['Method','Celltype',metric]].pivot(
            index='Method',columns='Celltype').sort_index(
            key=lambda x: [name_dict[j] for j in x]
        )
        axes[batch,j] = sns.heatmap(_df, annot=True, ax=axes[batch,j], cmap=cmap_list[j])
        if j==0:
            axes[batch,j].set_ylabel(batch_list[batch])
        else:
            axes[batch,j].set_ylabel('')

        axes[batch,j].set_xlabel('')
        if batch==0:
            axes[batch,j].set_title(metric_list[j])
        else:
            if j==1:
                axes[batch,j].set_xlabel('Cell Type')

            axes[batch,j].set_xticks(np.arange(len(celltype_list))+0.5, labels=celltype_list, rotation=0)
plt.tight_layout()
plt.savefig('result/Fig3b.png', dpi=300, bbox_inches='tight', pad_inches=0) 



##############################################################################
#
# Ex3: Figure 4, Figure S4, and Figure 5
#
##############################################################################

'''
Fig4 & FigS4 imputation
'''
df = pd.DataFrame()

method = 'scVAEIT'
_df = pd.read_csv('result/ex2/CD4 T/res_%s_masked_2.csv'%(method),
                  index_col=[0], skiprows=1,
                  names=['i','Batch','Target','Metric 1','Metric 2','MSE'])
_df['p'] = np.repeat(np.arange(1,9)/10, 6*10)
_df['Method'] = method
df = pd.concat([df, _df])
for method in ['Seurat', 'totalVI', 'MultiVI']:
    for p in np.arange(1,9)/10:
        _df = pd.read_csv('result/ex2/CD4 T/%s/res_%s_masked_%.01f.csv'%(
            method, method, p),
                          index_col=[0], skiprows=1,
                          names=['i','Batch','Target','Metric 1','Metric 2','MSE'])
        _df['p'] = p
        _df['Method'] = method
        df = pd.concat([df, _df])
df = df.reset_index(drop=True)    

df['Batch'] = df['Batch'].replace({0: 'Control', 1: 'Stimulation'})


sns.set_theme()
sns.set(font_scale = 1.3)


_df = pd.melt(df, id_vars=['i','Batch','Target','p','Method'], 
             value_vars=['Metric 1','Metric 2', 'MSE'], var_name='Metric')
_df['Metric'] = _df['Metric'].replace({'Metric 1': 'Pearson r', 'Metric 2': 'Spearman r'})
plot = sns.relplot(data=_df[(_df['Target']=='ADT')], x="p", y="value", 
             hue='Method', 
             markers=True, 
            row='Metric', col='Batch', kind="line",
             ci='sd', 
            facet_kws={'sharey': False, 'sharex': True},
                   height=3, aspect=8/4
            )
plot.set_titles("")
plot.axes[0,0].set_ylabel("Pearson r")
plot.axes[1,0].set_ylabel("Spearman r")
plot.axes[2,0].set_ylabel("RMSE")
plot.axes[2,0].set_xlabel("missing proportion")
plot.axes[2,1].set_xlabel("missing proportion")
plot.axes[0,0].set_title("Control")
plot.axes[0,1].set_title("Stimulation")

plt.savefig('result/Fig4a.png', dpi=300, bbox_inches='tight', pad_inches=0) 



_df = pd.melt(df, id_vars=['i','Batch', 'Target','p','Method'], 
             value_vars=['Metric 1','Metric 2', 'MSE'], var_name='Metric')
_df['Metric'] = _df['Metric'].replace({'Metric 1': 'AUC', 'Metric 2': 'BCE'})
plot = sns.relplot(data=_df[_df['Target']=='ATAC'], x="p", y="value", 
             hue='Method', 
             markers=True, 
            row='Metric', col='Batch', kind="line",
             ci='sd',
            facet_kws={'sharey':False , 'sharex': True}, 
            palette=np.array(sns.color_palette("muted"))[[0,1,3]].tolist(),
            height=3, aspect=8/4
            )
plot.set_titles("")
plot.axes[0,0].set_ylabel("AUROC")
plot.axes[1,0].set_ylabel("BCE")
plot.axes[2,0].set_ylabel("RMSE")
plot.axes[2,0].set_xlabel("missing proportion")
plot.axes[2,1].set_xlabel("missing proportion")
plot.axes[0,0].set_title("Control")
plot.axes[0,1].set_title("Stimulation")
plt.savefig('result/Fig4b.png', dpi=300, bbox_inches='tight', pad_inches=0) 


_df = pd.melt(df, id_vars=['i','Batch','Target','p','Method'], 
             value_vars=['Metric 1','Metric 2', 'MSE'], var_name='Metric')
_df['Metric'] = _df['Metric'].replace({'Metric 1': 'Pearson r', 'Metric 2': 'Spearman r'})
sns.relplot(data=_df[_df['Target']=='RNA'], x="p", y="value", 
             hue='Method', 
             markers=True, 
            row='Batch', col='Metric', kind="line",
             ci='sd',
            facet_kws={'sharey': False, 'sharex': True}
            )
plt.savefig('result/FigS4.png', dpi=300, bbox_inches='tight', pad_inches=0) 



'''
Fig5a imputation
'''
sns.set_theme()
sns.set(font_scale = 1.)
target = 'ADT'

df = pd.DataFrame()
for cell_type in  ['CD4 T', 'CD8 T', 'B', 'NK', 'DC', 'Mono']:        
    for method in ['scVAEIT', 'Seurat']:
        if cell_type in ['DC', 'Mono'] and method=='Seurat':
            continue
        _df = pd.read_csv(
            'result/ex3/{}/res_{}_overall.csv'.format(
                cell_type, method, data, target, batch),
                names=['Cell type', 'Name', 'Target', 'Batch','Pearson r','Spearman r','MSE'], skiprows=1)
        _df['Method'] = method

        df = pd.concat([df,_df])
df = df.reset_index(drop=True)
df['RMSE'] = np.sqrt(df['MSE'])
df['Batch'] = np.where(df['Batch']==0, 'Control','Stimulation')


import matplotlib as mpl
from matplotlib.colors import to_rgb
from matplotlib.legend_handler import HandlerTuple

sns.set_theme()
sns.set(font_scale = 1.)
fig, axes = plt.subplots(1,3, figsize=(16,4))
plt.tight_layout()
axes[0] = sns.violinplot(x='Method', y="Pearson r", hue="Batch",
                    data=df, cut=0., palette="muted", ax=axes[0])
axes[0].tick_params(axis='y', which='major', labelsize=10)
colors = sns.color_palette()
handles= []
for i, violin in enumerate(axes[0].findobj(mpl.collections.PolyCollection)):
    rgb = to_rgb(colors[i // 2])
    if i % 2 == 0:
        rgb = 0.5 + 0.5 * np.array(rgb)
    violin.set_facecolor(rgb)
    handles.append(plt.Rectangle((0, 0), 0, 0, facecolor=rgb, edgecolor='black'))
axes[0].legend(handles=[tuple(handles[::2]), tuple(handles[1::2])], 
              labels=['Control', 'Stimulation'], title="Condition", 
               handlelength=4, handler_map={tuple: HandlerTuple(ndivide=None, pad=0)})


axes[1] = sns.violinplot(x='Method', y="Spearman r", hue="Batch",
                    data=df, cut=0., palette="muted", ax=axes[1])
axes[1].tick_params(axis='y', which='major', labelsize=10)
colors = sns.color_palette()
handles= []
for i, violin in enumerate(axes[1].findobj(mpl.collections.PolyCollection)):
    rgb = to_rgb(colors[i // 2])
    if i % 2 == 0:
        rgb = 0.5 + 0.5 * np.array(rgb)
    violin.set_facecolor(rgb)
    handles.append(plt.Rectangle((0, 0), 0, 0, facecolor=rgb, edgecolor='black'))
axes[1].legend(handles=[tuple(handles[::2]), tuple(handles[1::2])], 
               labels=['Control', 'Stimulation'], title="Condition", 
               handlelength=4, handler_map={tuple: HandlerTuple(ndivide=None, pad=0)}, loc='lower left')


axes[2] = sns.violinplot(x='Method', y="RMSE", hue="Batch",
                    data=df, cut=0., palette="muted", ax=axes[2])
axes[2].tick_params(axis='y', which='major', labelsize=10)
colors = sns.color_palette()
handles= []
for i, violin in enumerate(axes[2].findobj(mpl.collections.PolyCollection)):
    rgb = to_rgb(colors[i // 2])
    if i % 2 == 0:
        rgb = 0.5 + 0.5 * np.array(rgb)
    violin.set_facecolor(rgb)
    handles.append(plt.Rectangle((0, 0), 0, 0, facecolor=rgb, edgecolor='black'))
axes[2].legend(handles=[tuple(handles[::2]), tuple(handles[1::2])], 
               labels=['Control', 'Stimulation'], title="Condition", 
               handlelength=4, handler_map={tuple: HandlerTuple(ndivide=None, pad=0)}, loc='lower right')

plt.savefig('result/Fig5a.png', dpi=300, bbox_inches='tight', pad_inches=0)



'''
Fig5b integration
'''

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

from tensorflow.keras.layers import Layer, Dense, BatchNormalization, LeakyReLU
from tensorflow.keras.utils import Progbar

tfd = tfp.distributions

import matplotlib.pyplot as plt

from time import time
from datetime import datetime

with h5py.File('data/dogma_cite_asap.h5', 'r') as f:
    print(f.keys())
    peak_names = np.array(f['peak_names'], dtype='S32').astype(str)
    ADT_names = np.array(f['ADT_names'], dtype='S32').astype(str)
    gene_names = np.array(f['gene_names'], dtype='S32').astype(str)
    X = sp.sparse.csc_matrix(
        (np.array(f['RNA.data'], dtype=np.float32), 
         np.array(f['RNA.indices'], dtype=np.int32),
         np.array(f['RNA.indptr'], dtype=np.int32)
        ), 
        shape = np.array(f['RNA.shape'], dtype=np.int32)).tocsc().astype(np.float32)
    Y = np.array(f['ADT'], dtype=np.float32)
    Z = sp.sparse.csc_matrix(
        (np.array(f['peaks.data'], dtype=np.float32),
         np.array(f['peaks.indices'], dtype=np.int32),
         np.array(f['peaks.indptr'], dtype=np.int32)
        ), 
        shape = np.array(f['peaks.shape'], dtype=np.int32)).tocsc().astype(np.float32)
    cell_types = np.array(f['cell_types'], dtype='S32').astype(str)
    batches = np.array(f['batches'], dtype=np.float32)
    
    id_X_dogma = np.array(f['id_X_dogma'], dtype=np.int32)
    id_Y_dogma = np.array(f['id_Y_dogma'], dtype=np.int32)
    id_Z_dogma = np.array(f['id_Z_dogma'], dtype=np.int32)    
    id_X_cite = np.array(f['id_X_cite'], dtype=np.int32)
    id_Y_cite = np.array(f['id_Y_cite'], dtype=np.int32)    
    id_Y_asap = np.array(f['id_Y_asap'], dtype=np.int32)
    id_Z_asap = np.array(f['id_Z_asap'], dtype=np.int32)
    
    sample_sizes = np.array(f['sample_sizes'], dtype=np.int32)
    
    
chunk_atac = np.array([
    np.sum(np.char.startswith(peak_names, 'chr%d-'%i)) for i in range(1,23)
    ], dtype=np.int32)

dim_input_arr = np.array([len(gene_names), len(ADT_names), len(peak_names)])
print(dim_input_arr)
X = X.toarray()
X[batches[:,-1]!=2,:] = np.log(X[batches[:,-1]!=2,:]/np.sum(X[batches[:,-1]!=2,:], axis=1, keepdims=True)*1e4+1.)
Y = np.log(Y/np.sum(Y, axis=1, keepdims=True)*1e4+1.)
Z[Z>0.] = 1.
Z = Z.toarray()
data = np.c_[X, Y, Z]

masks = - np.ones((3, np.sum(dim_input_arr)), dtype=np.float32)
masks[0,id_X_dogma] = 0.
masks[0,id_Y_dogma+dim_input_arr[0]] = 0.
masks[0,id_Z_dogma+np.sum(dim_input_arr[:2])] = 0.
masks[1,id_X_cite] = 0.
masks[1,id_Y_cite+dim_input_arr[0]] = 0.
masks[2,id_Y_asap+dim_input_arr[0]] = 0.
masks[2,id_Z_asap+np.sum(dim_input_arr[:2])] = 0.
masks = tf.convert_to_tensor(masks, dtype=tf.float32)


from types import SimpleNamespace

path_root = 'result/ex3/'
config = {
    'dim_input_arr': dim_input_arr,
    'dimensions':[256], 
    'dim_latent':32,
    'dim_block': np.append([len(gene_names),len(ADT_names)], chunk_atac), 
    'dist_block':['NB','NB'] + ['Bernoulli' for _ in chunk_atac], 
    'dim_block_enc':np.array([256, 128] + [16 for _ in chunk_atac]),
    'dim_block_dec':np.array([256, 128] + [16 for _ in chunk_atac]),
    'block_names':np.array(['rna', 'adt'] + ['atac' for _ in range(len(chunk_atac))]),
    'uni_block_names':np.array(['rna','adt','atac']),
    'dim_block_embed':np.array([16, 8] + [1 for _ in range(len(chunk_atac))])*2,

    'beta_kl':1.,
    'beta_unobs':2./3.,
    'beta_modal':np.array([0.14,0.85,0.01]),
    'beta_reverse':0.,

    "p_feat" : 0.2,
    "p_modal" : np.ones(3)/3,
    
}
config = SimpleNamespace(**config)
    
    

from scVAEIT.VAEIT import scVAEIT
model = scVAEIT(config, data, masks, batches)
checkpoint = tf.train.Checkpoint(net=model.vae)
epoch = 10
status = checkpoint.restore(path_root+'checkpoint/ckpt-{}'.format(epoch))
model.vae(tf.zeros((1,np.sum(model.vae.config.dim_input_arr))),
          tf.zeros((1,np.sum(model.vae.config.dim_input_arr))),
          tf.zeros((1,np.sum(model.batches.shape[1]))), 
          pre_train=True, L=1, training=False)
print(status)



map_dict = {0:'Control',1:'Stimulation'}
condition = np.array([map_dict[i] for i in batches[:,0]])
map_dict = {0:'DOGMA-seq',1:'CITE-seq',2:'ASAP-seq'}
dataset = np.array([map_dict[i] for i in batches[:,-1]])

model.update_z()


import scanpy as sc

z_mean = []
for x,b,id_data in model.dataset_full:
    m = tf.gather(model.vae.masks, id_data)
    m = tf.where(m==0., 0., 1.)
    m = m.numpy()
    m[:,:config.dim_input_arr[0]] = 1.
    m[:,-config.dim_input_arr[2]:] = 1.

    embed = model.vae.embed_layer(m)
    _z_mean, _, _ = model.vae.encoder(x, embed, b, 1, False)         
    z_mean.append(_z_mean.numpy())
z_mean = np.concatenate(z_mean)   

model.z = z_mean
model.adata = sc.AnnData(model.z)
sc.pp.neighbors(model.adata)
model.adata.obs['Condition'] = condition
model.adata.obs['Condition'] = model.adata.obs['Condition'].astype("category")
model.adata.obs['Dataset'] = dataset
model.adata.obs['Dataset'] = model.adata.obs['Dataset'].astype("category")
model.adata.obs['Cell Types'] = cell_types

adata = sc.AnnData(
    X=pd.DataFrame(data[:,config.dim_input_arr[0]:-config.dim_input_arr[2]], columns=ADT_names))

adata.uns = model.adata.uns
adata.obsm = model.adata.obsm
adata.obs = model.adata.obs

umap_seurat = pd.read_csv('result/ex3/full/Seurat_embedding.csv', index_col=0)

adata.obsm['X_umap_seurat'] = adata.obsm['X_umap'].copy()
adata.obsm['X_umap_seurat'] = umap_seurat.values


fig, axes = plt.subplots(1,2,figsize=(8,4))
sc.pl.embedding(adata, color='Condition', basis='X_umap_seurat', ax=axes[0], show=False)
sc.pl.embedding(adata, color='Dataset', basis='X_umap_seurat',  ax=axes[1], show=False)

# Hide the right and top spines
axes[0].spines.right.set_visible(False)
axes[0].spines.top.set_visible(False)

# Only show ticks on the left and bottom spines
axes[0].yaxis.set_ticks_position('left')
axes[0].xaxis.set_ticks_position('bottom')
axes[0].legend(loc='lower left')
axes[0].set_xlabel('UMAP1')
axes[0].set_ylabel('UMAP2')
axes[0].text(-.10, 0.5, 'Seurat',
        horizontalalignment='right',
        verticalalignment='center',
        rotation='vertical',
        transform=axes[0].transAxes, fontsize=14)

axes[1].spines.right.set_visible(False)
axes[1].spines.top.set_visible(False)
axes[1].yaxis.set_ticks_position('left')
axes[1].legend(loc='lower left')
axes[1].set_xlabel('UMAP1')
axes[1].set_ylabel('UMAP2')

plt.tight_layout()
plt.savefig('result/Fig5b-1.png', dpi=300, bbox_inches='tight', pad_inches=0) 



fig, axes = plt.subplots(1,2,figsize=(8,4))

model.visualize_latent(method = "UMAP", color = 'Condition', ax=axes[0], show=False)
model.visualize_latent(method = "UMAP", color = 'Dataset', ax=axes[1], show=False)

# Hide the right and top spines
axes[0].spines.right.set_visible(False)
axes[0].spines.top.set_visible(False)

# Only show ticks on the left and bottom spines
axes[0].yaxis.set_ticks_position('left')
axes[0].xaxis.set_ticks_position('bottom')
axes[0].legend(loc='lower left')
axes[0].set_xlabel('UMAP1')
axes[0].set_ylabel('UMAP2')
axes[0].text(-.10, 0.5, 'scVAEIT',
        horizontalalignment='right',
        verticalalignment='center',
        rotation='vertical',
        transform=axes[0].transAxes, fontsize=14)

axes[1].spines.right.set_visible(False)
axes[1].spines.top.set_visible(False)
axes[1].yaxis.set_ticks_position('left')
axes[1].legend(loc='lower left')
axes[1].set_xlabel('UMAP1')
axes[1].set_ylabel('UMAP2')

plt.tight_layout()
plt.savefig('result/Fig5b-2.png', dpi=300, bbox_inches='tight', pad_inches=0) 


fig, axes = plt.subplots(1,2,figsize=(9,4))
sc.pl.umap(adata[adata.obs['Condition']=='Control'], color='CD3-1', ax=axes[0], #colorbar_loc=None, 
           show=False)
sc.pl.umap(adata[adata.obs['Condition']=='Stimulation'], color='CD3-1', ax=axes[1], show=False)

# Hide the right and top spines
axes[0].spines.right.set_visible(False)
axes[0].spines.top.set_visible(False)

# Only show ticks on the left and bottom spines
axes[0].yaxis.set_ticks_position('left')
axes[0].xaxis.set_ticks_position('bottom')
# axes[0].legend(loc='lower left')
axes[0].set_xlabel('UMAP1')
axes[0].set_ylabel('UMAP2')
axes[0].set_title('CD3 Protein (Control)')

axes[1].spines.right.set_visible(False)
axes[1].spines.top.set_visible(False)
axes[1].yaxis.set_ticks_position('left')
axes[1].set_xlabel('UMAP1')
axes[1].set_ylabel('UMAP2')
axes[1].set_title('CD3 Protein (Stimulation)')

plt.tight_layout()
plt.savefig('result/Fig5b-3.png', dpi=300, bbox_inches='tight', pad_inches=0) 


fig, axes = plt.subplots(1,2,figsize=(9,4))
sc.pl.embedding(adata[adata.obs['Condition']=='Control'], color='CD3-1', ax=axes[0], show=False, basis='X_umap_seurat')
sc.pl.embedding(adata[adata.obs['Condition']=='Stimulation'], color='CD3-1', ax=axes[1], show=False, basis='X_umap_seurat')

# Hide the right and top spines
axes[0].spines.right.set_visible(False)
axes[0].spines.top.set_visible(False)

# Only show ticks on the left and bottom spines
axes[0].yaxis.set_ticks_position('left')
axes[0].xaxis.set_ticks_position('bottom')
# axes[0].legend(loc='lower left')
axes[0].set_xlabel('UMAP1')
axes[0].set_ylabel('UMAP2')
axes[0].set_title('CD3 Protein (Control)')

axes[1].spines.right.set_visible(False)
axes[1].spines.top.set_visible(False)
axes[1].yaxis.set_ticks_position('left')
axes[1].set_xlabel('UMAP1')
axes[1].set_ylabel('UMAP2')
axes[1].set_title('CD3 Protein (Stimulation)')


plt.tight_layout()
plt.savefig('result/Fig5b-4.png', dpi=300, bbox_inches='tight', pad_inches=0) 