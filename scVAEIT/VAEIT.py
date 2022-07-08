import warnings
from typing import Optional, Union
from types import SimpleNamespace

import scVAEIT.model as model 
import scVAEIT.train as train 
from scVAEIT.inference import Inferer
from scVAEIT.utils import get_igraph, leidenalg_igraph, \
   DE_test, _comp_dist, _get_smooth_curve
import tensorflow as tf

from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.cluster import AgglomerativeClustering
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from scipy import stats
import scanpy as sc
import matplotlib.patheffects as pe

class scVAEIT():
    """
    Variational Inference for Trajectory by AutoEncoder.
    """
    def __init__(self, config: SimpleNamespace, data, masks=None, batches=None,
        ):
        '''
        Get input data for model. Data need to be first processed using scancy and stored as an AnnData object
         The 'UMI' or 'non-UMI' model need the original count matrix, so the count matrix need to be saved in
         adata.layers in order to use these models.


        Parameters
        ----------
        config : SimpleNamespace
            Dict of config.

        Returns
        -------
        None.

        '''
        self.dict_method_scname = {
            'PCA' : 'X_pca',
            'UMAP' : 'X_umap',
            'TSNE' : 'X_tsne',
            'diffmap' : 'X_diffmap',
            'draw_graph' : 'X_draw_graph_fa'
        }

#         if model_type != 'Gaussian':
#             if adata_layer_counts is None:
#                 raise ValueError("need to provide the name in adata.layers that stores the raw count data")
#             if 'highly_variable' not in adata.var:
#                 raise ValueError("need to first select highly variable genes using scanpy")
#         if copy_adata:
#             self.adata = adata.copy()
#         else:
#             self.adata = adata

#         self._adata = sc.AnnData(X = self.adata.X, var = self.adata.var)
#         self._adata.obs = self.adata.obs
#         self._adata.uns = self.adata.uns

        self.data = data
        
        if batches is None:
            batches = np.zeros((data.shape[0],1), dtype=np.float32)
        if masks is None:
            masks = np.zeros((len(np.unique(batches[:,-1])), data.shape[1]), dtype=np.float32)
            
        self.cat_enc = OneHotEncoder().fit(batches)
        self.id_dataset = batches[:,-1].astype(np.int32)
        self.batches = self.cat_enc.transform(batches).toarray()        

        self.vae = model.VariationalAutoEncoder(config, masks)

        if hasattr(self, 'inferer'):
            delattr(self, 'inferer')
        

    def train(self, valid = False, stratify = False, test_size = 0.1, random_state: int = 0,
            learning_rate: float = 1e-3, batch_size: int = 256, batch_size_inference: int = 512, 
              L: int = 1, alpha: float = 0.10,
            num_epoch: int = 200, num_step_per_epoch: Optional[int] = None, save_every_epoch: Optional[int] = 25,
            early_stopping_patience: int = 10, early_stopping_tolerance: float = 1e-4, 
            early_stopping_relative: bool = True, verbose: bool = False,
            checkpoint_dir: Optional[str] = None, delete_existing: Optional[str] = True, eval_func=None): 
 #           path_to_weights: Optional[str] = None):
        '''Pretrain the model with specified learning rate.

        Parameters
        ----------
        test_size : float or int, optional
            The proportion or size of the test set.
        random_state : int, optional
            The random state for data splitting.
        learning_rate : float, optional
            The initial learning rate for the Adam optimizer.
        batch_size : int, optional 
            The batch size for pre-training.  Default is 256. Set to 32 if number of cells is small (less than 1000)
        L : int, optional 
            The number of MC samples.
        alpha : float, optional
            The value of alpha in [0,1] to encourage covariate adjustment. Not used if there is no covariates.
        num_epoch : int, optional 
            The maximum number of epochs.
        num_step_per_epoch : int, optional 
            The number of step per epoch, it will be inferred from number of cells and batch size if it is None.            
        early_stopping_patience : int, optional 
            The maximum number of epochs if there is no improvement.
        early_stopping_tolerance : float, optional 
            The minimum change of loss to be considered as an improvement.
        early_stopping_relative : bool, optional
            Whether monitor the relative change of loss as stopping criteria or not.
        path_to_weights : str, optional 
            The path of weight file to be saved; not saving weight if None.
        '''
        if valid:
            if stratify is False:
                stratify = None    

            id_train, id_valid = train_test_split(
                                    np.arange(self.data.shape[0]),
                                    test_size=test_size,
                                    stratify=stratify,
                                    random_state=random_state)
            
            self.dataset_train = tf.data.Dataset.from_tensor_slices((
                self.data[id_train].astype(tf.keras.backend.floatx()), 
                self.batches[id_train].astype(tf.keras.backend.floatx()),
                self.id_dataset[id_train]
                )).shuffle(buffer_size = len(id_train), seed=0,
                           reshuffle_each_iteration=True).batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)

            self.dataset_valid = tf.data.Dataset.from_tensor_slices((
                    self.data[id_valid].astype(tf.keras.backend.floatx()), 
                    self.batches[id_valid].astype(tf.keras.backend.floatx()),
                    self.id_dataset[id_valid]
                )).batch(batch_size_inference).prefetch(tf.data.experimental.AUTOTUNE)
        else:
            id_train = np.arange(self.data.shape[0])
            self.dataset_train = tf.data.Dataset.from_tensor_slices((
                self.data.astype(tf.keras.backend.floatx()), 
                self.batches.astype(tf.keras.backend.floatx()),
                self.id_dataset
                )).shuffle(buffer_size = len(id_train), seed=0,
                           reshuffle_each_iteration=True).batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)
            self.dataset_valid = None
            
        if num_step_per_epoch is None:
            num_step_per_epoch = len(id_train)//batch_size+1
            
        checkpoint_dir = 'checkpoint/pretrain/' if checkpoint_dir is None else checkpoint_dir
        if delete_existing and tf.io.gfile.exists(checkpoint_dir):
            print("Deleting old log directory at {}".format(checkpoint_dir))
            tf.io.gfile.rmtree(checkpoint_dir)
        tf.io.gfile.makedirs(checkpoint_dir)
        
        self.vae = train.train(
            self.dataset_train,
            self.dataset_valid,
            self.vae,
            checkpoint_dir,
            learning_rate,                        
            L, alpha,
            num_epoch,
            num_step_per_epoch,
            save_every_epoch,
            early_stopping_patience,
            early_stopping_tolerance,
            early_stopping_relative,
            verbose,
            eval_func
        )
        
#         self.update_z()
#         with open(checkpoint_dir+'Z.npy', 'wb') as f:
#             np.save(f, Z)
 #       if path_to_weights is not None:
 #           self.save_model(path_to_weights)
            
    def save_model(self, path_to_weights):
        checkpoint = tf.train.Checkpoint(net=self.vae)
        manager = tf.train.CheckpointManager(
            checkpoint, path_to_weights, max_to_keep=None
        )
        save_path = manager.save()        
        print("Saved checkpoint: {}".format(save_path))
        
    def load_model(self, path_to_weights):
        checkpoint = tf.train.Checkpoint(net=self.vae)
        status = checkpoint.restore(path_to_weights)
        print("Loaded checkpoint: {}".format(status))
    
        
    def update_z(self):
        self.z = self.get_latent_z()        
        self.adata = sc.AnnData(self.z)
        sc.pp.neighbors(self.adata)

            
    def get_latent_z(self, batch_size_inference=512):
        ''' get the posterier mean of current latent space z (encoder output)

        Returns
        ----------
        z : np.array
            \([N,d]\) The latent means.
        ''' 
        if not hasattr(self, 'dataset_full'):
            self.dataset_full = tf.data.Dataset.from_tensor_slices((
                    self.data.astype(tf.keras.backend.floatx()), 
                    self.batches.astype(tf.keras.backend.floatx()),
                    self.id_dataset
                )).batch(batch_size_inference).prefetch(tf.data.experimental.AUTOTUNE)

        return self.vae.get_z(self.dataset_full)
            
    
    def visualize_latent(self, method: str = "UMAP", 
                         color = None, **kwargs):
        '''
        visualize the current latent space z using the scanpy visualization tools

        Parameters
        ----------
        method : str, optional
            Visualization method to use. The default is "draw_graph" (the FA plot). Possible choices include "PCA", "UMAP", 
            "diffmap", "TSNE" and "draw_graph"
        color : TYPE, optional
            Keys for annotations of observations/cells or variables/genes, e.g., 'ann1' or ['ann1', 'ann2'].
            The default is None. Same as scanpy.
        **kwargs :  
            Extra key-value arguments that can be passed to scanpy plotting functions (scanpy.pl.XX).   

        Returns
        -------
        None.

        '''
          
        if method not in ['PCA', 'UMAP', 'TSNE', 'diffmap', 'draw_graph']:
            raise ValueError("visualization method should be one of 'PCA', 'UMAP', 'TSNE', 'diffmap' and 'draw_graph'")
        
        temp = list(self.adata.obsm.keys())
        if method == 'PCA' and not 'X_pca' in temp:
            print("Calculate PCs ...")
            sc.tl.pca(self.adata)
        elif method == 'UMAP' and not 'X_umap' in temp:  
            print("Calculate UMAP ...")
            sc.tl.umap(self.adata)
        elif method == 'TSNE' and not 'X_tsne' in temp:
            print("Calculate TSNE ...")
            sc.tl.tsne(self.adata)
        elif method == 'diffmap' and not 'X_diffmap' in temp:
            print("Calculate diffusion map ...")
            sc.tl.diffmap(self.adata)
        elif method == 'draw_graph' and not 'X_draw_graph_fa' in temp:
            print("Calculate FA ...")
            sc.tl.draw_graph(self.adata)
            

#         self._adata.obsp = self.adata.obsp
#        self._adata.uns = self.adata.uns
#         self._adata.obsm = self.adata.obsm
    
        if method == 'PCA':
            axes = sc.pl.pca(self.adata, color = color, **kwargs)
        elif method == 'UMAP':            
            axes = sc.pl.umap(self.adata, color = color, **kwargs)
        elif method == 'TSNE':
            axes = sc.pl.tsne(self.adata, color = color, **kwargs)
        elif method == 'diffmap':
            axes = sc.pl.diffmap(self.adata, color = color, **kwargs)
        elif method == 'draw_graph':
            axes = sc.pl.draw_graph(self.adata, color = color, **kwargs)
            
        return axes


    def init_latent_space(self, cluster_label = None, log_pi = None, res: float = 1.0, 
                          ratio_prune= None, dist_thres = 0.5):
        '''Initialize the latent space.

        Parameters
        ----------
        cluster_label : str, optional
            the name of vector of labels that can be found in self.adata.obs. 
            Default is None, which will perform leiden clustering on the pretrained z to get clusters
        mu : np.array, optional
            \([d,k]\) The value of initial \(\\mu\).
        log_pi : np.array, optional
            \([1,K]\) The value of initial \(\\log(\\pi)\).
        res: 
            The resolution of leiden clustering, which is a parameter value controlling the coarseness of the clustering. 
            Higher values lead to more clusters. Deafult is 1.
        ratio_prune : float, optional
            The ratio of edges to be removed before estimating.
        '''   
    
        
        if cluster_label is None:
            print("Perform leiden clustering on the latent space z ...")
            g = get_igraph(self.z)
            cluster_labels = leidenalg_igraph(g, res = res)
            cluster_labels = cluster_labels.astype(str) 
            uni_cluster_labels = np.unique(cluster_labels)
        else:
            cluster_labels = self.adata.obs[cluster_label].to_numpy()                   
            uni_cluster_labels = np.array(self.adata.obs[cluster_label].cat.categories)

        n_clusters = len(uni_cluster_labels)

        if not hasattr(self, 'z'):
            self.update_z()        
        z = self.z
        mu = np.zeros((z.shape[1], n_clusters))
        for i,l in enumerate(uni_cluster_labels):
            mu[:,i] = np.mean(z[cluster_labels==l], axis=0)
   #         mu[:,i] = z[cluster_labels==l][np.argmin(np.mean((z[cluster_labels==l] - mu[:,i])**2, axis=1)),:]
       
   ### update cluster centers if some cluster centers are too close
        clustering = AgglomerativeClustering(
            n_clusters=None,
            distance_threshold=dist_thres,
            linkage='complete'
            ).fit(mu.T/np.sqrt(mu.shape[0]))
        n_clusters_new = clustering.n_clusters_
        if n_clusters_new < n_clusters:
            print("Merge clusters for cluster centers that are too close ...")
            n_clusters = n_clusters_new
            for i in range(n_clusters):    
                temp = uni_cluster_labels[clustering.labels_ == i]
                idx = np.isin(cluster_labels, temp)
                cluster_labels[idx] = ','.join(temp)
                if np.sum(clustering.labels_==i)>1:
                    print('Merge %s'% ','.join(temp))
            uni_cluster_labels = np.unique(cluster_labels)
            mu = np.zeros((z.shape[1], n_clusters))
            for i,l in enumerate(uni_cluster_labels):
                mu[:,i] = np.mean(z[cluster_labels==l], axis=0)
            
        self.adata.obs['vitae_init_clustering'] = cluster_labels
        self.adata.obs['vitae_init_clustering'] = self.adata.obs['vitae_init_clustering'].astype('category')
        print("Initial clustering labels saved as 'vitae_init_clustering' in self.adata.obs.")

   
        if (log_pi is None) and (cluster_labels is not None) and (n_clusters>3):                         
            n_states = int((n_clusters+1)*n_clusters/2)
            d = _comp_dist(z, cluster_labels, mu.T)

            C = np.triu(np.ones(n_clusters))
            C[C>0] = np.arange(n_states)
            C = C.astype(int)

            log_pi = np.zeros((1,n_states))
            ## pruning to throw away edges for far-away clusters if there are too many clusters
            if ratio_prune is not None:
                log_pi[0, C[np.triu(d)>np.quantile(d[np.triu_indices(n_clusters, 1)], 1-ratio_prune)]] = - np.inf
            else:
                log_pi[0, C[np.triu(d)> np.quantile(d[np.triu_indices(n_clusters, 1)], 5/n_clusters) * 3]] = - np.inf

        self.n_states = n_clusters
        self.labels = cluster_labels
        # Not sure if storing the this will be useful
        # self.init_labels_name = cluster_label
        
        labels_map = pd.DataFrame.from_dict(
            {i:label for i,label in enumerate(uni_cluster_labels)}, 
            orient='index', columns=['label_names'], dtype=str
            )
        
        self.labels_map = labels_map
        self.vae.init_latent_space(self.n_states, mu, log_pi)
        self.inferer = Inferer(self.n_states)
        self.mu = self.vae.latent_space.mu.numpy()
        self.pi = np.triu(np.ones(self.n_states))
        self.pi[self.pi > 0] = tf.nn.softmax(self.vae.latent_space.pi).numpy()[0]

    def update_latent_space(self, dist_thres: float=0.5):
        pi = self.pi[np.triu_indices(self.n_states)]
        mu = self.mu    
        clustering = AgglomerativeClustering(
            n_clusters=None,
            distance_threshold=dist_thres,
            linkage='complete'
            ).fit(mu.T/np.sqrt(mu.shape[0]))
        n_clusters = clustering.n_clusters_   

        if n_clusters<self.n_states:      
            print("Merge clusters for cluster centers that are too close ...")
            mu_new = np.empty((self.dim_latent, n_clusters))
            C = np.zeros((self.n_states, self.n_states))
            C[np.triu_indices(self.n_states, 0)] = pi
            C = np.triu(C, 1) + C.T
            C_new = np.zeros((n_clusters, n_clusters))
            
            uni_cluster_labels = self.labels_map['label_names'].to_numpy()
            returned_order = {}
            cluster_labels = self.labels
            for i in range(n_clusters):
                temp = uni_cluster_labels[clustering.labels_ == i]
                idx = np.isin(cluster_labels, temp)
                cluster_labels[idx] = ','.join(temp)
                returned_order[i] = ','.join(temp)
                if np.sum(clustering.labels_==i)>1:
                    print('Merge %s'% ','.join(temp))
            uni_cluster_labels = np.unique(cluster_labels) 
            for i,l in enumerate(uni_cluster_labels):  ## reorder the merged clusters based on the cluster names
                k = np.where(returned_order == l)
                mu_new[:, i] = np.mean(mu[:,clustering.labels_==k], axis=-1)
                # sum of the aggregated pi's
                C_new[i, i] = np.sum(np.triu(C[clustering.labels_==k,:][:,clustering.labels_==k]))
                for j in range(i+1, n_clusters):
                    k1 = np.where(returned_order == uni_cluster_labels[j])
                    C_new[i, j] = np.sum(C[clustering.labels_== k, :][:, clustering.labels_==k1])

#            labels_map_new = {}
#            for i in range(n_clusters):                       
#                # update label map: int->str
#                labels_map_new[i] = self.labels_map.loc[clustering.labels_==i, 'label_names'].str.cat(sep=',')
#                if np.sum(clustering.labels_==i)>1:
#                    print('Merge %s'%labels_map_new[i])
#                # mean of the aggregated cluster means
#                mu_new[:, i] = np.mean(mu[:,clustering.labels_==i], axis=-1)
#                # sum of the aggregated pi's
#                C_new[i, i] = np.sum(np.triu(C[clustering.labels_==i,:][:,clustering.labels_==i]))
#                for j in range(i+1, n_clusters):
#                    C_new[i, j] = np.sum(C[clustering.labels_== i, :][:, clustering.labels_==j])
            C_new = np.triu(C_new,1) + C_new.T

            pi_new = C_new[np.triu_indices(n_clusters)]
            log_pi_new = np.log(pi_new, out=np.ones_like(pi_new)*(-np.inf), where=(pi_new!=0)).reshape((1,-1))
            self.n_states = n_clusters
            self.labels_map = pd.DataFrame.from_dict(
                {i:label for i,label in enumerate(uni_cluster_labels)},
                orient='index', columns=['label_names'], dtype=str
                )
            self.labels = cluster_labels
#            self.labels_map = pd.DataFrame.from_dict(
#                labels_map_new, orient='index', columns=['label_names'], dtype=str
#            )
            self.vae.init_latent_space(self.n_states, mu_new, log_pi_new)
            self.inferer = Inferer(self.n_states)
            self.mu = self.vae.latent_space.mu.numpy()
            self.pi = np.triu(np.ones(self.n_states))
            self.pi[self.pi > 0] = tf.nn.softmax(self.vae.latent_space.pi).numpy()[0]






 

    
