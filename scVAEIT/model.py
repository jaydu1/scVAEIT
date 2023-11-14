import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions

from scVAEIT.utils import ModalMaskGenerator
from scVAEIT.nn_utils import Encoder, Decoder
from tensorflow.keras.layers import Layer, Dense, BatchNormalization
from tensorflow.keras.utils import Progbar

            
            
class VariationalAutoEncoder(tf.keras.Model):
    """
    Combines the encoder, decoder into an end-to-end model for training and inference.
    """
    def __init__(self, config, masks, name = 'autoencoder', **kwargs):
        '''
        Parameters
        ----------
        dimensions : np.array
            The dimensions of hidden layers of the encoder.
        dim_latent : int
            The latent dimension of the encoder.
        
        dim_block : list of int
            (num_block,) The dimension of each input block.        
        dist_block : list of str
            (num_block,) `'NB'`, `'ZINB'`, `'Bernoulli'` or `'Gaussian'`.
        dim_block_enc : list of int
            (num_block,) The dimension of output of first layer of the encoder for each block.
        dim_block_dec : list of int
            (num_block,) The dimension of output of last layer of the decoder for each block.        
        block_names : list of str, optional
            (num_block,) The name of first layer for each block.
        name : str, optional
            The name of the layer.
        **kwargs : 
            Extra keyword arguments.
        '''
        super(VariationalAutoEncoder, self).__init__(name = name, **kwargs)
        self.config = config
        self.masks = masks
        self.embed_layer = Dense(np.sum(self.config.dim_block_embed), 
                                 activation = tf.nn.tanh, name = 'embed')
        self.encoder = Encoder(self.config.dimensions, self.config.dim_latent,
            self.config.dim_block, self.config.dim_block_enc, self.config.dim_block_embed, self.config.block_names)
        self.decoder = Decoder(self.config.dimensions[::-1], self.config.dim_block,
            self.config.dist_block, self.config.dim_block_dec, self.config.dim_block_embed, 
            self.config.max_vals, self.config.block_names)
        
        self.mask_generator = ModalMaskGenerator(
            config.dim_input_arr, config.p_feat, config.p_modal)
        
        
    def generate_mask(self, inputs, mask=None, p=None):
        '''
        Generate mask for inputs.

        Parameters
        ----------
        inputs : tf.Tensor of type tf.float32
            The input data.
        mask : tf.Tensor of type tf.float32
            The mask of input data: -1,0,1 indicate missing, observed, and masked.
        '''
        return self.mask_generator(inputs, mask, p)


    def call(self, x, masks, batches, L=1, training=True):
        '''Feed forward through encoder and decoder.

        Parameters
        ----------
        x : np.array, optional
            The input data.
        masks : np.array, optional
            The mask of input data: -1,0,1 indicate missing, observed, and masked.
        batches : np.array, optional
            The covariates of each cell.
        L : int, optional
            The number of MC samples.
        training : boolean, optional
            Whether in the training phase or not.

        Returns
        ----------
        losses : float
            the loss.
        '''
                            
        z_mean_obs, z_log_var_obs, log_probs_obs = self._get_reconstruction_loss(
            x, masks!=-1., masks!=-1., batches, L, training=training)
        z_mean_unobs_1, z_log_var_unobs_1, log_probs_unobs = self._get_reconstruction_loss(
            x, masks==0., masks==1., batches, L, training=training)
        z_mean_unobs_2, z_log_var_unobs_2, log_probs_ = self._get_reconstruction_loss(
            x, masks==1., masks==0., batches, L, training=training)
        log_probs_unobs = (1-self.config.beta_reverse) * log_probs_unobs + self.config.beta_reverse*log_probs_
        
        self.add_loss(
            [- (1-self.config.beta_unobs) * 
             tf.reduce_sum(tf.where(self.config.block_names==name, log_probs_obs, 0.)) * 
             self.config.beta_modal[i] for i,name in enumerate(self.config.uni_block_names)]
        )
        self.add_loss(
            [- self.config.beta_unobs * 
             tf.reduce_sum(tf.where(self.config.block_names==name, log_probs_unobs, 0.)) * 
             self.config.beta_modal[i] for i,name in enumerate(self.config.uni_block_names)]
        )
        
        kl = (1-self.config.beta_reverse) * self._get_kl_normal(
            z_mean_unobs_1, z_log_var_unobs_1, z_mean_obs, z_log_var_obs) + \
            self.config.beta_reverse * self._get_kl_normal(
            z_mean_unobs_2, z_log_var_unobs_2, z_mean_obs, z_log_var_obs)
        self.add_loss(self.config.beta_kl * kl)

        return self.losses
    

    @tf.function
    def _get_reconstruction_loss(self, x, bool_mask_in, bool_mask_out, batches, L, training=True):
        '''
        Parameters
        ----------
        x : tf.Tensor of type tf.float32
            The input data \(Y_i\).
        bool_mask_in : tf.Tensor of type tf.bool
            False indicates missing.
        bool_mask_out : tf.Tensor of type tf.bool
            Compute likelihood for entries with value True.
        batches : tf.Tensor of type tf.int32
            The batch index of each cell.
        L : int
            The number of MC samples.
        training : boolean, optional
            Whether in the training phase or not.
        '''
        _masks = tf.where(bool_mask_in, 0., 1.)
        _x = tf.where(bool_mask_in, x, 0.)
        embed = self.embed_layer(_masks)
        z_mean, z_log_var, z, x_embed = self.encoder(_x, embed, batches, L, training=training)
        if not self.config.skip_conn:
            x_embed = tf.zeros_like(x_embed)
        log_probs = tf.reduce_mean(
            self.decoder(x, embed, bool_mask_out, batches, z, x_embed, training=training), axis=0)
        return z_mean, z_log_var, log_probs
    
    
    @tf.function
    def _get_kl_normal(self, mu_0, log_var_0, mu_1, log_var_1):
        kl = 0.5 * (
            tf.exp(tf.clip_by_value(log_var_0-log_var_1, -6., 6.)) + 
            (mu_1 - mu_0)**2 / tf.exp(tf.clip_by_value(log_var_1, -6., 6.)) - 1.
             + log_var_1 - log_var_0)
        return tf.reduce_mean(tf.reduce_sum(kl, axis=-1))
    
    
    def get_recon(self, dataset_test, full_masks, masks=None, zero_out=True, 
                  return_mean=True, L=50):
        '''
        Compute the reconstruction of the input data.

        Parameters
        ----------
        dataset_test : tf.Dataset
            Dataset containing (x, batches).
        full_masks : boolean
            Whether to use full masks.
        masks : np.array, optional
            The mask of input data: -1,0,1 indicate missing, observed, and masked.
        zero_out : boolean, optional
            Whether to zero out the missing values.
        return_mean : boolean, optional
            Whether to return the mean of posterior samples.
        L : int, optional
            The number of MC samples.

        Returns
        ----------
        x_hat : np.array
            The reconstruction.
        '''
        if masks is None:
            masks = self.masks
        x_hat = []
        for x,b,m in dataset_test:
            if not full_masks:
                m = tf.gather(masks, m)
            _m = tf.where(m==0., 0., 1.)
            embed = self.embed_layer(_m)
            if zero_out:
                x = tf.where(m==0, x, 0.)
            _, _, z, x_embed = self.encoder(x, embed, b, L, False)
            if not self.config.skip_conn:
                x_embed = tf.zeros_like(x_embed)
            _x_hat = self.decoder(x, embed, tf.ones_like(m,dtype=tf.bool), 
                    b, z, x_embed, training=False, return_prob=False)
            if return_mean:
                _x_hat = tf.reduce_mean(_x_hat, axis=1)
            x_hat.append(_x_hat.numpy())
        x_hat = np.concatenate(x_hat)        

        return x_hat
    
    
    def get_z(self, dataset_test, full_masks=False, masks=None):
        '''Compute latent variables.

        Parameters
        ----------
        dataset_test : tf.Dataset
            Dataset containing (x, batches).
        full_masks : boolean
            Whether to use full masks.
        masks : np.array, optional
            The mask of input data: -1,0,1 indicate missing, observed, and masked.

        Returns
        ----------
        z_mean : np.array
            The latent mean.
        '''
        if masks is None:
            masks = self.masks
        z_mean = []
        for x,b,m in dataset_test:
            if not full_masks:
                m = tf.gather(masks, m)
            m = tf.where(m==0., 0., 1.)
            embed = self.embed_layer(m)
            _z_mean, _, _, _ = self.encoder(x, embed, b, 1, False)
            z_mean.append(_z_mean.numpy())
        z_mean = np.concatenate(z_mean)        

        return z_mean


    
