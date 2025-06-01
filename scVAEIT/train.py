# -*- coding: utf-8 -*-
from typing import Optional

from scVAEIT.utils import Early_Stopping

import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import Progbar

import sys
from time import time



def clear_session():
    '''Clear Tensorflow sessions.
    '''
    tf.keras.backend.clear_session()
    return None


class loss_metric(object):
    def __init__(self, uni_block_names, evaluate=False, verbose=False):
        n_modal = len(uni_block_names)
        self.loss_names = ['total', ] +\
            ['obs_{}'.format(i) for i in range(n_modal)] +\
            ['unobs_{}'.format(i) for i in range(n_modal)] +\
            ['kl','mmd']
        self.loss_names_print = ['total', ] +\
            ['obs_{}'.format(i) for i in uni_block_names] +\
            ['unobs_{}'.format(i) for i in uni_block_names] +\
            ['kl','mmd']
        self.verbose = verbose
        names = ['train'] if evaluate is False else ['train', 'val']
        self.losses = {}
        self.hist = {}
        for name in names:
            self.losses[name] = {loss_name:tf.metrics.Mean(name+'_loss'+loss_name, dtype=tf.float32) 
                                       for loss_name in self.loss_names}
            self.hist[name] = {loss_name:[] for loss_name in self.loss_names}
        
    def __call__(self, losses, name):
        for i,loss_name in enumerate(self.losses[name]):
            if i==0:
                self.losses[name][loss_name](tf.reduce_sum(losses))
            else:
                self.losses[name][loss_name](losses[i-1])
        
    def reset_state(self):
        for name in self.losses:
            for loss_name in self.losses[name]:
                self.losses[name][loss_name].reset_state()
        # TF 2.4.1 reset_states 
        # TF 2.5.0 reset_state
        
    def on_epoch_end(self, header):
        if self.verbose:
            print(header)
            for name in self.losses:
                print('{:<5s}   '.format('')+', '.join('{:>7s}'.format(i) for i in self.loss_names_print), flush = True)
                print('{:<5s} : '.format(name)+', '.join('{:>7.02f}'.format(self.losses[name][l].result()) 
                                                       for l in self.losses[name]), flush = True)
        for name in self.hist:
            for loss_name in self.losses[name]:
                self.hist[name][loss_name].append(self.losses[name][loss_name].result().numpy())


def train(dataset_train, dataset_valid, vae, checkpoint_dir, 
              learning_rate: float, L: int,
              num_epoch: int, num_step_per_epoch: int, save_every_epoch: int, init_epoch: int = 1,
              es_patience: int = 10, es_tolerance: int = 1e-4, es_relative: bool = True,
              full_masks: bool = False, verbose: bool = True, eval_func = None):
    '''Train a Variational Autoencoder (VAE) model.

    Parameters
    ----------
    dataset_train : tf.data.Dataset
        The TensorFlow Dataset object for training data.
    dataset_valid : tf.data.Dataset
        The TensorFlow Dataset object for validation data.
    vae : VariationalAutoEncoder
        The VAE model to be trained.
    checkpoint_dir : str
        Directory to save model checkpoints.
    learning_rate : float
        The initial learning rate for the AdamW optimizer.
    L : int
        The number of Monte Carlo samples.
    num_epoch : int
        The maximum number of epochs for training.
    num_step_per_epoch : int
        The number of steps per epoch. If None, it will be inferred from the number of cells and batch size.
    save_every_epoch : int
        Frequency (in epochs) to save model checkpoints.
    init_epoch : int, optional
        The initial epoch number, default is 1.
    es_patience : int, optional
        The number of epochs to wait for improvement before early stopping, default is 10.
    es_tolerance : float, optional
        The minimum change in loss to be considered as an improvement, default is 1e-4.
    es_relative : bool, optional
        Whether to monitor the relative change in loss for early stopping, default is True.
    full_masks : bool, optional
        Whether to use full masks, default is False.
    verbose : bool, optional
        Whether to print the training process, default is True.
    eval_func : function, optional
        A function to evaluate the model, which takes the VAE as an input.

    Returns
    -------
    vae : VariationalAutoEncoder
        The trained VAE model.
    hist : dict
        A dictionary containing the history of training and validation losses.
    '''
    
    optimizer =  tf.keras.optimizers.AdamW(learning_rate=learning_rate, weight_decay=1e-4)

    if checkpoint_dir is not None:
        checkpoint = tf.train.Checkpoint(optimizer=optimizer, net=vae, step=tf.Variable(1),)
        manager = tf.train.CheckpointManager(checkpoint, checkpoint_dir, 
                                            max_to_keep=None if dataset_valid is None else es_patience+2)

        # Restore from the latest checkpoint if available
        if manager.latest_checkpoint:
            checkpoint.restore(manager.latest_checkpoint)
            print(f"Restored from {manager.latest_checkpoint}", flush = True)
            checkpoint.step.assign_add(1)

    evaluate = dataset_valid is not None
    loss_monitor = loss_metric(vae.config.uni_block_names, evaluate, verbose)        
    early_stopping = Early_Stopping(patience=es_patience, tolerance=es_tolerance, relative=es_relative)

    if not verbose:
        progbar = Progbar(num_epoch)
    
    start_time = time()
    for epoch in range(init_epoch, num_epoch+1):

        if verbose:
            progbar = Progbar(num_step_per_epoch)
            print('Train - Start of epoch %d' % (epoch,), flush = True)
        else:
            if (epoch+1)%2==0 or epoch+1==num_epoch:
                progbar.update(epoch+1)
                sys.stdout.flush()

        gamma = vae.config.gamma if np.isscalar(vae.config.gamma) else vae.config.gamma(epoch)

        # Iterate over the batches of the dataset.
        for step, (x, b, m, c) in enumerate(dataset_train):
            if not full_masks:
                m = tf.gather(vae.masks, m)
            m = vae.generate_mask(x, m)
            with tf.GradientTape() as tape:
                losses = vae(x, m, b, c, gamma=gamma, L=L)
                # Compute reconstruction loss
                loss = tf.reduce_sum(losses)
            grads = tape.gradient(loss, vae.trainable_weights,
                        unconnected_gradients=tf.UnconnectedGradients.ZERO)
            optimizer.apply_gradients(zip(grads, vae.trainable_weights))
            
            loss_monitor(losses, 'train')

            if verbose:
                if (step+1)%10==0 or step+1==num_step_per_epoch:
                    progbar.update(step+1, [('Reconstructed Loss', float(loss))])
        
        if evaluate:
            for step, (x, b, m, c) in enumerate(dataset_valid):
                if not full_masks:
                    m = tf.gather(vae.masks, m)
                m = vae.generate_mask(x, m, p=0.)
                losses = vae(x, m, b, c, L=L, training=False)
                loss_monitor(losses, 'val')

        loss_monitor.on_epoch_end(
            'Epoch {}, Time elapsed: {} minutes'.format(epoch, round((time() - start_time) / 60, 2))
        )
                
        if checkpoint_dir is not None:
            # When validation set is available, save the model within latest es_patience+1 epochs.
            if dataset_valid is not None:
                save_path = manager.save()
                print("Saved checkpoint for epoch {}: {}".format(epoch, save_path), flush = True)

                if early_stopping(float(loss_monitor['val'][0].result())):
                    print('Early stopping.', flush = True)
                    break
            else:
                if int(checkpoint.step) % save_every_epoch == 0:
                    print(checkpoint.step)
                    save_path = manager.save()
                    print("Saved checkpoint for epoch {}: {}".format(epoch, save_path), flush = True)

                    if eval_func is not None:
                        eval_func(vae)
            checkpoint.step.assign_add(1)
        
        loss_monitor.reset_state()
        
    print('Train Done.', flush = True)
    return vae, loss_monitor.hist