import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions

from scVAEIT.utils import BiModalMaskGenerator, BiModalandFixMaskGenerator
from scVAEIT.nn_utils import prior_network, proposal_network, generative_network
from tensorflow.keras.layers import Dense




class BiModalVAEIT(tf.keras.Model):
    def __init__(self, config, fixed_masks=None):
        super(BiModalVAEIT, self).__init__()
        self.config = config

        self.prior_net = prior_network(2*config.dim_latent)
        if self.config.rna_dist=='NB':
            self.generative_net = generative_network(config.dim_input,)
        else:
            self.generative_net = generative_network(config.dim_input + config.dim_input_rna,)
        self.mask_generator = BiModalandFixMaskGenerator(
            fixed_masks, config.dim_input_rna, config.dim_input_adt, config.p_feat, config.p_modal)
        # dispersion parameter
        self.log_r_rna = tf.Variable(
            tf.zeros([1, config.dim_input_rna], dtype=tf.keras.backend.floatx()), name = "log_r_rna")
        self.log_r_adt = tf.Variable(
            tf.zeros([1, config.dim_input_adt], dtype=tf.keras.backend.floatx()), name = "log_r_adt")

    def generate_mask(self, inputs, p=None):
        return self.mask_generator(inputs, p)

    @staticmethod
    def make_observed_inputs(inputs, masks):
        """
        compute x_observed
        :param inputs:
        :param masks:
        :return:
        """
        return tf.where(tf.cast(masks, tf.bool), tf.zeros_like(inputs), inputs)

    @tf.function
    def get_latent(self, inputs, masks, training=False):
        batch_size = inputs.shape[0]
        
        observed_inputs_with_masks = tf.concat([inputs, tf.zeros_like(inputs)], axis=-1)
        prior_params = self.prior_net(observed_inputs_with_masks, training=training)

        # (batch_size, dim_latent)
        prior_distribution = tfd.Normal(
            loc=prior_params[..., :self.config.dim_latent],
            scale=tf.clip_by_value(
            tf.nn.softplus(prior_params[..., self.config.dim_latent:]),
            1e-3,
            tf.float32.max),
            name="priors")

        # (batch_size, dim_latent)
        latent = prior_distribution.sample()
        
        # (batch_size, 2*dim_input)
        observed_inputs = self.make_observed_inputs(inputs, masks)
        observed_inputs_with_masks = tf.concat([observed_inputs, masks], axis=-1)
        prior_params = self.prior_net(observed_inputs_with_masks, training=training)

        # (batch_size, dim_latent)
        prior_distribution_1 = tfd.Normal(
            loc=prior_params[..., :self.config.dim_latent],
            scale=tf.clip_by_value(
            tf.nn.softplus(prior_params[..., self.config.dim_latent:]),
            1e-3,
            tf.float32.max),
            name="priors")

        # (batch_size, dim_latent)
        latent_1 = prior_distribution_1.sample()
        
        
        # (batch_size, 2*dim_input)
        observed_inputs = self.make_observed_inputs(inputs, 1-masks)
        observed_inputs_with_masks = tf.concat([observed_inputs, 1-masks], axis=-1)
        prior_params = self.prior_net(observed_inputs_with_masks, training=training)

        # (batch_size, dim_latent)
        prior_distribution_2 = tfd.Normal(
            loc=prior_params[..., :self.config.dim_latent],
            scale=tf.clip_by_value(
            tf.nn.softplus(prior_params[..., self.config.dim_latent:]),
            1e-3,
            tf.float32.max),
            name="priors_2")

        # (batch_size, dim_latent)
        latent_2 = prior_distribution_2.sample()
        
        # (batch_size, )
        divergence = tf.reduce_sum(
            tfd.kl_divergence(prior_distribution_1, prior_distribution), -1
        ) + tf.reduce_sum(
                tfd.kl_divergence(prior_distribution_2, prior_distribution), -1)
        
        out = tf.nn.sigmoid(self.generative_net(latent, training=training))
        out_1 = tf.nn.sigmoid(self.generative_net(latent_1, training=training))
        out_2 = tf.nn.sigmoid(self.generative_net(latent_2, training=training))
        
        return out, out_1, out_2, -tf.reduce_mean(divergence)
    
    
    @tf.function
    def get_probs(self, inputs, out,
                  disp_rna, disp_adt, training=False):
        if self.config.rna_dist=='NB':
            lambda_z = out * np.log(10**4+1)
            generative_dist_rna = tfd.NegativeBinomial.experimental_from_mean_dispersion(
                mean = lambda_z[..., :self.config.dim_input_rna], 
                dispersion = disp_rna,
                name='generative_RNA'
            )
        else:
            phi_rna = tf.clip_by_value(out[..., self.config.dim_input:], 1e-5, 1.-1e-5)
            lambda_z = out[..., :self.config.dim_input] * np.log(10**4+1)
        
            generative_dist_rna = tfd.Mixture(
                cat=tfd.Categorical(
                    probs=tf.stack([phi_rna, 1.0 - phi_rna], axis=-1)),
                components=[tfd.Deterministic(loc=tf.zeros_like(phi_rna)), 
                            tfd.NegativeBinomial.experimental_from_mean_dispersion(
                                mean = lambda_z[..., :self.config.dim_input_rna], 
                                dispersion = disp_rna
                            )],
                name='generative_RNA'
            )
        

        generative_dist_adt = tfd.NegativeBinomial.experimental_from_mean_dispersion(
            mean = lambda_z[..., self.config.dim_input_rna:],
            dispersion=disp_adt,
            name='generative_ADT'
        )
        
        # (batch_size, )
        probs = tf.concat(
            [self.config.beta * generative_dist_rna.log_prob(inputs[..., :self.config.dim_input_rna]),
            (1 - self.config.beta) * generative_dist_adt.log_prob(inputs[..., self.config.dim_input_rna:])], axis=-1
            )
        
        return probs
    
        
    def compute_loss(self, inputs, masks, training=False):
        """
        :param inputs: (batch_size, dim_input)
        :param masks: (batch_size, dim_input)
        :return:
        """
        
        out, out_1, out_2, neg_kl = self.get_latent(inputs, masks, training)
        
        disp_rna = tfp.math.clip_by_value_preserve_gradient(
            tf.nn.softplus(
                self.log_r_rna), 0., 6.)
        disp_adt = tfp.math.clip_by_value_preserve_gradient(
            tf.nn.softplus(
                self.log_r_adt), 0., 6.)
        
        
        probs = self.get_probs(inputs, out,
                               disp_rna, disp_adt, training)
        likelihood_observed_rna = tf.reduce_sum(probs[..., :self.config.dim_input_rna], -1)
        likelihood_observed_adt = tf.reduce_sum(probs[..., self.config.dim_input_rna:], -1)
        
        probs = self.get_probs(inputs, out_1, 
                               disp_rna, disp_adt, training)
        likelihood_unobserved = tf.multiply(probs, masks)
        likelihood_unobserved_rna = tf.reduce_sum(likelihood_unobserved[..., :self.config.dim_input_rna], -1)
        likelihood_unobserved_adt = tf.reduce_sum(likelihood_unobserved[..., self.config.dim_input_rna:], -1)
        
        probs = self.get_probs(inputs, out_2, 
                               disp_rna, disp_adt, training)
        likelihood_unobserved = tf.multiply(probs, 1 - masks)
        likelihood_unobserved_rna += tf.reduce_sum(likelihood_unobserved[..., :self.config.dim_input_rna], -1)
        likelihood_unobserved_adt += tf.reduce_sum(likelihood_unobserved[..., self.config.dim_input_rna:], -1)

        '''
        tf.print(
            tf.reduce_sum(
                probs[..., :self.config.dim_input_rna]
            ),
            tf.reduce_sum(
                probs[..., self.config.dim_input_rna:]
            )
        )
        '''

        
        return  neg_kl, tf.reduce_mean(likelihood_unobserved_rna), \
    tf.reduce_mean(likelihood_observed_rna),\
    tf.reduce_mean(likelihood_unobserved_adt),\
    tf.reduce_mean(likelihood_observed_adt)

    
    def prior_regularizer(self, prior):
        # (batch_size, -1)
        mu = prior.mean() # tf.reshape(prior.mean(), (self.config.batch_size, -1))
        sigma = prior.scale # tf.reshape(prior.scale, (self.config.batch_size, -1))

        # (batch_size, )
        mu_regularizer = -tf.reduce_sum(tf.square(mu), -1) / (2 * self.config.sigma_mu ** 2)
        sigma_regularizer = tf.reduce_sum((tf.math.log(sigma) - sigma), -1) * self.config.sigma_sigma
        return mu_regularizer + sigma_regularizer

    
    @tf.function
    def compute_apply_gradients(self, optimizer, inputs, masks, train_loss, train_loss_list):
        with tf.GradientTape() as tape:
            losses = self.compute_loss(inputs, masks, True)
            loss = - (
                (losses[0] + losses[1] + losses[3]) * self.config.alpha
                      + (losses[2] + losses[4]) * (1-self.config.alpha)  
            )/ self.config.scale_factor
        train_loss(loss)
        for i, l in enumerate(train_loss_list):
            l(losses[i])

        gradients = tape.gradient(loss, self.trainable_variables,
                                  unconnected_gradients=tf.UnconnectedGradients.ZERO)
        optimizer.apply_gradients(zip(gradients, self.trainable_variables))

        
    def generate_samples(self, inputs, masks, n_samples=10):
        # (batch_size, width, height, channels)
        observed_inputs = self.make_observed_inputs(inputs, masks)
        # (batch_size, width, height, 2*channels)
        observed_inputs_with_masks = tf.concat([observed_inputs, masks], axis=-1)

        prior_params = self.prior_net(observed_inputs_with_masks, training=False)
        prior_distribution = tfd.Normal(
            loc=prior_params[..., :self.config.dim_latent],
            scale=tf.clip_by_value(
            tf.nn.softplus(prior_params[..., self.config.dim_latent:]),
            1e-3,
            tf.float32.max),
            name="priors")
    
        x_hat = []
        y_hat = []
        for _ in range(n_samples):
            latent = prior_distribution.sample()
            out = tf.nn.sigmoid(self.generative_net(latent, training=False))
            if self.config.rna_dist=='NB':
                lambda_z = out * np.log(10**4+1)
                x_hat.append(lambda_z[..., :self.config.dim_input_rna])
            else:
                phi_rna = tf.clip_by_value(out[..., self.config.dim_input:], 1e-5, 1.-1e-5)
                lambda_z = out[..., :self.config.dim_input] * np.log(10**4+1)
                x_hat.append(lambda_z[..., :self.config.dim_input_rna] * (1-phi_rna))
                
            y_hat.append(lambda_z[..., self.config.dim_input_rna:])


        x_hat = tf.reduce_mean(tf.stack(x_hat, axis=1), axis=1)
        y_hat = tf.reduce_mean(tf.stack(y_hat, axis=1), axis=1)        

        return x_hat, y_hat
