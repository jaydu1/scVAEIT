import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions

from tensorflow.keras.layers import Layer, Dense, BatchNormalization, LeakyReLU, Lambda
use_bias = True



###########################################################################
#
# Input and output blocks for multimodal datasets
#
###########################################################################
class InputBlock(tf.keras.layers.Layer):
    def __init__(self, dim_inputs, dim_latents, dim_embed, mean_vals, names=None, bn=False, **kwargs):
        '''
        Parameters
        ----------
        dim_inputs : list of int
            (B+1,) The dimension of each input block, where the last block 
            is assumed to be the batch effects.
        dim_latent : list of int
            (B,) The dimension of output of first layer for each block.
        dim_embed : int
            The dimension of the embedding layer.
        mean_vals : tf.Tensor
            The mean values of the outputs; only used for Gaussian distribution.
        names : list of str, optional
            (B,) The name of first layer for each block.
        **kwargs : 
            Extra keyword arguments.
        '''
        super(InputBlock, self).__init__()
        
        self.dim_inputs = tf.constant(dim_inputs, dtype=tf.int32)
        self.dim_embed = tf.constant(dim_embed, dtype=tf.int32)
        if names is None:
            names = ['Block_{}'.format(i) for i in range(len(dim_latents))]
        self.names = names
        self.dim_latents = dim_latents
        
        self.linear_layers = [
            Dense(d, use_bias=use_bias, activation = LeakyReLU(), name=names[i]) if d>0 else 
            tf.keras.layers.Lambda(lambda x,training: tf.identity(x))
                for i,d in enumerate(dim_latents)
        ]
        if bn:
            self.bn = BatchNormalization(center=False)
        else:
            self.bn = Lambda(lambda x,training: tf.identity(x))
        self.concat = tf.keras.layers.Concatenate()
        
        self.mean_vals = tf.expand_dims(mean_vals, 0)

    @tf.function(reduce_retracing=True)
    def call(self, x, mask, embed, batches, training=True):
        x = tf.where(mask, x - self.mean_vals, 0.)
        x_list = tf.split(x, self.dim_inputs, axis=1)

        embed_list = tf.split(embed, self.dim_embed, axis=1)
        outputs = self.concat([
            self.linear_layers[i](
                tf.concat([x_list[i], embed_list[i], batches], axis=1), training=training
                ) for i in range(len(self.dim_latents))])
        outputs = self.bn(outputs, training=training)
        
        return outputs




def get_dist(dist, mean_val, min_val, max_val):
    if dist=='NB':
        generative_dist = lambda x_hat, disp, mask, zi_prob: tfd.Independent(tfd.Masked(
                tfd.NegativeBinomial.experimental_from_mean_dispersion(
                    mean = min_val + x_hat * (max_val - min_val),
                    dispersion = disp, name='NB_rv'
                ), mask), reinterpreted_batch_ndims=1)
    elif dist=='Bernoulli':
        generative_dist = lambda x_hat, disp, mask, zi_prob: tfd.Independent(tfd.Masked(
            tfd.Bernoulli(
                probs = tfp.math.clip_by_value_preserve_gradient(x_hat, min_val, max_val),
                dtype=tf.float32, name='Bernoulli_rv'
            ), mask), reinterpreted_batch_ndims=1)
    elif dist=='Gaussian':
        # Calculate symmetric range around mean
        upper_range = max_val - mean_val
        lower_range = mean_val - min_val
        max_range = tf.maximum(upper_range, lower_range)
        
        # Create symmetric bounds around mean
        sym_min = mean_val - max_range
        sym_max = mean_val + max_range 
        
        generative_dist = lambda x_hat, disp, mask, zi_prob: tfd.Independent(tfd.Masked(
            tfd.Normal(
                loc = sym_min + x_hat * (sym_max - sym_min), 
                scale = disp, name='Gaussian_rv'
            ), mask), reinterpreted_batch_ndims=1)
    elif dist=='Poisson':
        log_min = tf.where(min_val > 0, tf.math.log(min_val), tf.constant(-np.inf))
        generative_dist = lambda x_hat, disp, mask, zi_prob: tfd.Independent(tfd.Masked(
            tfd.Poisson(
                log_rate = tfp.math.clip_by_value_preserve_gradient(x_hat, log_min, tf.math.log(max_val)), 
                name='Poisson_rv'
            ), mask), reinterpreted_batch_ndims=1)
    elif dist=='ZINB':
        generative_dist = lambda x_hat, disp, mask, zi_prob: tfd.Independent(tfd.Masked(
            tfd.Inflated(
                tfd.NegativeBinomial.experimental_from_mean_dispersion(
                    mean=min_val + x_hat * (max_val - min_val),
                    dispersion=disp, name='NB_rv'
                ), inflated_loc_probs=tf.clip_by_value(zi_prob, 1e-3, 1.-1e-3)), mask), reinterpreted_batch_ndims=1) 
    return generative_dist
    


class OutputBlock(tf.keras.layers.Layer):
    def __init__(self, dim_outputs, dist_outputs, dim_latents, dim_embed, 
        mean_vals, min_vals, max_vals, max_disp, max_zi_prob, names=None, bn=True, **kwargs):
        '''
        Parameters
        ----------
        dim_outputs : list of int
            (B,) The dimension of each output block.
        dist_outputs : list of str
            (B,) The distribution of each output block.
        dim_latents : list of int
            (B,) The dimension of output of last layer for each block.
        dim_embed : int
            The dimension of the embedding layer.
        mean_vals : tf.Tensor
            The mean values of the outputs; only used or Gaussian distribution.
        min_vals : tf.Tensor
            The minimum values of the outputs.
        max_vals : tf.Tensor
            The maximum values of the outputs.
        max_disp : float
            The maximum value of the dispersion parameter. `max_disp=6` by default.        
        names : list of str, optional
            (B,) The name of last layer for each block.
        bn : boolean
            Whether use batch normalization or not.
        
        **kwargs : 
            Extra keyword arguments.
        '''        
        super(OutputBlock, self).__init__()
        self.dim_inputs = tf.constant(dim_outputs, dtype=tf.int32)
        self.dim_embed = tf.constant(dim_embed, dtype=tf.int32)
        self.dim_outputs = tf.constant([d for i,d in enumerate(dim_outputs)], dtype=tf.int32)
        self.dist_outputs = dist_outputs#tf.constant(dist_outputs, dtype=tf.string)
        self.dim_latents = tf.constant(dim_latents, dtype=tf.int32)
        if names is None:
            names = ['Block_{}'.format(i) for i in range(len(dim_latents))]
        self.names = names        
        
        self.linear_layers = [
            Dense(d, use_bias=use_bias, activation = LeakyReLU(), name=names[i]) if d>0 else 
            (lambda k,d: Lambda(lambda x,training: tf.identity(x), name="identity_{}".format(names[k])))(i,d)
                for i,d in enumerate(self.dim_latents)
        ]
        if bn:
            self.bn = [BatchNormalization(center=False) for _ in range(len(dim_latents))]
        else:
            self.bn = [Lambda(lambda x,training: tf.identity(x)) for _ in range(len(dim_latents))]
        self.output_layers = [Dense(d, use_bias=use_bias, name=names[i]) for i,d in enumerate(self.dim_outputs)]
        self.out_act = [tf.identity if dist in ['Poisson'] else 
                        tf.nn.sigmoid for dist in self.dist_outputs]

        self.disp = [
            Dense(d, use_bias=use_bias, activation = tf.nn.softplus, name="disp_{}".format(names[i])) 
            if self.dist_outputs[i]!='Bernoulli' else 
            (lambda k,d: Lambda(lambda x,training: tf.zeros((1,1,tf.constant(d)), dtype=tf.float32), output_shape=(1,tf.constant(d)), name="lambda_{}".format(names[k])))(i,d)
                for i,d in enumerate(dim_outputs)
        ]
        self.max_disp = max_disp

        self.max_vals = tf.split(tf.expand_dims(max_vals, 0), self.dim_inputs, axis=-1)
        self.min_vals = tf.split(tf.expand_dims(min_vals, 0), self.dim_inputs, axis=-1)
        self.mean_vals = tf.split(tf.expand_dims(mean_vals, 0), self.dim_inputs, axis=-1)

        self.zi_prob = [
            (lambda k,d: Lambda(lambda x,training: tf.ones((512,1,tf.constant(d)), dtype=tf.float32)/2., output_shape=(1,tf.constant(d)), name="lambda_{}".format(names[k])))(i,d)
            # Dense(d, use_bias=use_bias, activation = tf.nn.sigmoid, name="zi_prob_{}".format(names[i])) 
            if self.dist_outputs[i].startswith('ZI') else 
            (lambda k,d: Lambda(lambda x,training: tf.zeros((1,1,tf.constant(d)), dtype=tf.float32), output_shape=(1,tf.constant(d)), name="lambda_{}".format(names[k])))(i,d)
                for i,d in enumerate(dim_outputs)
        ]

        self.dists = [get_dist(dist, mean_val, min_val, max_val) for dist, mean_val, min_val, max_val in zip(self.dist_outputs, mean_vals, min_vals, max_vals)]

        self.concat = tf.keras.layers.Concatenate()
        
        
    # @tf.function(reduce_retracing=True)
    def call(self, x, embed, masks, batches, z, x_embed, training=True):
        '''
        Parameters
        ----------
        x : tf.Tensor
            \([B, D]\) the observed \(x\).
        z : tf.Tensor
            \([B, L, d]\) the sampled \(z\).
        batches : tf.Tensor
            \([B, b]\) the sampled \(z\).
        masks : tf.Tensor
            \([B, D]\) the mask indicating feature missing.
        training : boolean, optional
            whether in the training or inference mode.
        '''

        m_list = tf.split(tf.expand_dims(masks,1), self.dim_inputs, axis=-1)
        x_list = tf.split(tf.expand_dims(x,1), self.dim_inputs, axis=-1)
        x_emded_list = tf.split(tf.expand_dims(x_embed,1), self.dim_latents, axis=-1)
        

        L = tf.shape(z)[1]
        batches = tf.tile(tf.expand_dims(batches, 1), (1, L, 1))

        probs = self.concat([
            self.dists[i](
                self.out_act[i](
                    self.output_layers[i](self.bn[i](
                        self.linear_layers[i](z,training=training) + x_emded_list[i], 
                        training=training), training=training)
                ),
                tfp.math.clip_by_value_preserve_gradient(
                    self.disp[i](batches, training=training), 0., self.max_disp),
                m_list[i],
                self.zi_prob[i](batches, training=training)
            ).log_prob(x_list[i]) for i in range(len(self.dim_latents))
        ])

        return probs

    
    @tf.function(reduce_retracing=True)
    def get_recon(self, embed, masks, batches, z, x_embed, training=True):
        m_list = tf.split(tf.expand_dims(masks,1), self.dim_inputs, axis=-1)
        x_emded_list = tf.split(tf.expand_dims(x_embed,1), self.dim_latents, axis=-1)

        L = tf.shape(z)[1]
        batches = tf.tile(tf.expand_dims(batches, 1), (1, L, 1))
        
        x_hat = self.concat([
            self.dists[i](
                self.out_act[i](
                    self.output_layers[i](self.bn[i](
                        self.linear_layers[i](z, training=training) + x_emded_list[i], 
                        training=training), training=training)
                ),                 
                tfp.math.clip_by_value_preserve_gradient(
                    self.disp[i](batches, training=training), 0., self.max_disp),
                m_list[i],
                self.zi_prob[i](batches, training=training)
            ).mean() for i in range(len(self.dim_latents))
        ])

        return x_hat


###########################################################################
#
# Sampling layers in the latent space
#
###########################################################################

class Sampling(Layer):
    """Sampling latent variable \(z\) from \(N(\\mu_z, \\log \\sigma_z^2\)).    
    Used in Encoder.
    """
    def __init__(self, seed=0, **kwargs):
        super(Sampling, self).__init__(**kwargs)
        self.seed = seed

    @tf.function(reduce_retracing=True)
    def call(self, z_mean, z_log_var):
        '''Return cdf(x) and pdf(x).

        Parameters
        ----------
        z_mean : tf.Tensor
            \([B, L, d]\) The mean of \(z\).
        z_log_var : tf.Tensor
            \([B, L, d]\) The log-variance of \(z\).

        Returns
        ----------
        z : tf.Tensor
            \([B, L, d]\) The sampled \(z\).
        '''   
        epsilon = tf.random.normal(shape = tf.shape(z_mean), dtype=tf.keras.backend.floatx())
        z = z_mean + tf.exp(0.5 * z_log_var) * epsilon
        z = tf.clip_by_value(z, -1e6, 1e6)
        return z



###########################################################################
#
# Encoder
# 
###########################################################################
class Encoder(Layer):
    '''
    Encoder, model \(p(Z_i|Y_i,X_i)\).
    '''
    def __init__(self, dimensions, dim_latent, 
        dim_block_inputs, dim_block_latents, dim_embed, 
        mean_vals, block_names=None, name='encoder', **kwargs):
        '''
        Parameters
        ----------
        dimensions : np.array
            The dimensions of hidden layers of the encoder.
        dim_latent : int
            The latent dimension of the encoder.
        dim_block_inputs : list of int
            (num_block,) The dimension of each input block, where the last block 
            is assumed to be the batch effects.
        dim_block_latents : list of int
            (num_block,) The dimension of output of first layer for each block.
        mean_vals : np.array
            The mean values of the outputs; only used for Gaussian distribution.
        dim_embed : int
            The dimension of the embedding layer.
        block_names : list of str, optional
            (num_block,) The name of first layer for each block.  
        name : str, optional
            The name of the layer.
        **kwargs : 
            Extra keyword arguments.
        ''' 
        super(Encoder, self).__init__(name = name, **kwargs)
        self.input_layer = InputBlock(dim_block_inputs, dim_block_latents, dim_embed, mean_vals, block_names, bn=False)
        self.dense_layers = [Dense(dim, activation = LeakyReLU(),
                                          name = 'encoder_%i'%(i+1)) \
                             for (i, dim) in enumerate(dimensions)]
        self.batch_norm_layers = [BatchNormalization(center=False) \
                                    for _ in range(len(dimensions))]
        self.batch_norm_layers.append(BatchNormalization(center=False))
        self.latent_mean = Dense(dim_latent, name = 'latent_mean')
        self.latent_log_var = Dense(dim_latent, name = 'latent_log_var')
        self.sampling = Sampling()
    
    
    @tf.function(reduce_retracing=True)
    def call(self, x, mask, embed, batches, L=1, training=True):
        '''Encode the inputs and get the latent variables.

        Parameters
        ----------
        x : tf.Tensor
            \([B, L, d]\) The input.
        mask : tf.Tensor
            \([B, L, d]\) The boolean mask indicating feature missing.
        embed : tf.Tensor
            \([B, L, d]\) The embedding of the input.
        batches : tf.Tensor
            \([B, b]\) The batch effects.
        L : int, optional
            The number of MC samples.
        training : boolean, optional
            Whether in the training or inference mode.
        
        Returns
        ----------
        z_mean : tf.Tensor
            \([B, L, d]\) The mean of \(z\).
        z_log_var : tf.Tensor
            \([B, L, d]\) The log-variance of \(z\).
        z : tf.Tensor
            \([B, L, d]\) The sampled \(z\).
        '''
        tmp = self.input_layer(x, mask, embed, batches, training=training)
        _z = tmp
        for dense, bn in zip(self.dense_layers, self.batch_norm_layers):
            _z = dense(_z, training=training)
            _z = bn(_z, training=training)
        z_mean = self.batch_norm_layers[-1](self.latent_mean(_z, training=training), training=training)
        z_log_var = self.latent_log_var(_z)
        _z_mean = tf.tile(tf.expand_dims(z_mean, 1), (1,L,1))
        _z_log_var = tf.tile(tf.expand_dims(z_log_var, 1), (1,L,1))
        z = self.sampling(_z_mean, _z_log_var)
        return z_mean, z_log_var, z, tmp



###########################################################################
#
# Decoder
# 
###########################################################################
class Decoder(Layer):
    '''
    Decoder, model \(p(Y_i|Z_i,X_i)\).
    '''
    def __init__(self, dimensions, dim_block_outputs, 
        dist_block_outputs, dim_block_latents, dim_embed, 
        mean_vals, min_vals, max_vals, max_disp, max_zi_prob,
        block_names=None, name = 'decoder', **kwargs):
        '''
        Parameters
        ----------
        dimensions : np.array
            The dimensions of hidden layers of the encoder.
        dim_block_outputs : list of int
            (B,) The dimension of each output block.
        dist_block_outputs : list of str
            (B,) `'NB'`, `'ZINB'`, `'Bernoulli'` or `'Gaussian'`.
        dim_block_latents : list of int
            (B,) The dimension of output of last layer for each block.
        dim_embed : int
            The dimension of the embedding layer.
        mean_vals : np.array
            The mean values of the outputs; only used for Gaussian distribution.
        min_vals : np.array
            The minimum values of the outputs.
        max_vals : np.array
            The maximum values of the outputs.
        max_disp : float
            The maximum value of the dispersion parameter.
        block_names : list of str, optional
            (B,) The name of last layer for each block.
        name : str, optional
            The name of the layer.
        '''
        super(Decoder, self).__init__(name = name, **kwargs)
        self.output_layer = OutputBlock(
            dim_block_outputs, dist_block_outputs, dim_block_latents, dim_embed, 
            mean_vals, min_vals, max_vals, max_disp, block_names, bn=False)

        self.dense_layers = [Dense(dim, activation = LeakyReLU(),
                                          name = 'decoder_%i'%(i+1)) \
                             for (i,dim) in enumerate(dimensions)]
        self.batch_norm_layers = [BatchNormalization(center=False) \
                                    for _ in range(len((dimensions)))]


    @tf.function(reduce_retracing=True)
    def call(self, x, embed, masks, batches, z, x_embed, training=True, return_prob=True):
        '''Decode the latent variables and get the reconstructions.

        Parameters
        ----------        
        x : tf.Tensor
            \([B, D]\) the input.
        embed : tf.Tensor
            \([B, D, d]\) the embedding of the input.
        masks : tf.Tensor
            \([B, D]\) the boolean mask indicating feature missing.
        batches : tf.Tensor
            \([B, b]\) the batch effects.
        z : tf.Tensor
            \([B, L, d]\) the sampled \(z\).
        x_embed : tf.Tensor
            \([B, D, d]\) the embedding of the input.
        return_prob : boolean, optional
            whether to return the log probability or the reconstruction.
        training : boolean, optional
            whether in the training or inference mode.

        Returns
        ----------
        log_probs : tf.Tensor
            \([B, block]\) The log probability.
        '''
        L = tf.shape(z)[1]
        _z = tf.concat([
            z, 
            tf.tile(tf.expand_dims(tf.concat([embed,batches], axis=-1), 1), (1,L,1))
        ], axis=-1)
        for dense, bn in zip(self.dense_layers, self.batch_norm_layers):
            _z = dense(_z, training=training)
            _z = bn(_z, training=training)

        if return_prob:
            log_probs = self.output_layer(x, embed, masks, batches, _z, x_embed, training=training)
            return log_probs
        else:
            x_hat = self.output_layer.get_recon(embed, masks, batches, _z, x_embed, training=training)
            return x_hat

