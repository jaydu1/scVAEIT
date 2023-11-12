# -*- coding: utf-8 -*-
from typing import Optional

from scVAEIT.utils import Early_Stopping

import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import Progbar

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
            ['kl']
        self.loss_names_print = ['total', ] +\
            ['obs_{}'.format(i) for i in uni_block_names] +\
            ['unobs_{}'.format(i) for i in uni_block_names] +\
            ['kl']
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
        
        
    def reset_states(self):
        for name in self.losses:
            for loss_name in self.losses[name]:
                self.losses[name][loss_name].reset_states()
        
        
    def on_epoch_end(self, header):
        if self.verbose:
            print(header)
            for name in self.losses:
                print('{:<5s}   '.format('')+', '.join('{:>7s}'.format(i) for i in self.loss_names_print))
                print('{:<5s} : '.format(name)+', '.join('{:>7.02f}'.format(self.losses[name][l].result()) 
                                                       for l in self.losses[name]))
        for name in self.hist:
            for loss_name in self.losses[name]:
                self.hist[name][loss_name].append(self.losses[name][loss_name].result().numpy())


def train(dataset_train, dataset_valid, vae, checkpoint_dir, 
              learning_rate: float, L: int,
              num_epoch: int, num_step_per_epoch: int, save_every_epoch: int,
              es_patience: int, es_tolerance: int, es_relative: bool,
              full_masks: bool = False, verbose: bool = True, eval_func=None):
    '''Pretraining.

    Parameters
    ----------
    dataset_train : tf.Dataset
        The Tensorflow Dataset object.
    dataset_valid : tf.Dataset
        The Tensorflow Dataset object.
    vae : VariationalAutoEncoder
        The model.
    learning_rate : float
        The initial learning rate for the Adam optimizer.
    L : int
        The number of MC samples.
    num_epoch : int
        The maximum number of epoches.
    num_step_per_epoch : int
        The number of step per epoch, it will be inferred from number of cells and batch size if it is None.            
    es_patience : int
        The maximum number of epoches if there is no improvement.
    es_tolerance : float
        The minimum change of loss to be considered as an improvement.
    es_relative : bool, optional
        Whether monitor the relative change of loss or not.        
    es_warmup : int, optional
        The number of warmup epoches.
    full_masks : bool, optional
        Whether use full masks or not.
    verbose : bool, optional
        Whether print the training process or not.
    eval_func : function, optional
        The function to evaluate the model, which takes the vae as an input.

    Returns
    ----------
    vae : VariationalAutoEncoder
        The pretrained model.
    hist : dict
        The history of loss.
    '''
    
    optimizer =  tf.keras.optimizers.AdamW(learning_rate=learning_rate, weight_decay=1e-4)

    if checkpoint_dir is not None:
        checkpoint = tf.train.Checkpoint(optimizer=optimizer, net=vae, step=tf.Variable(1),)
        manager = tf.train.CheckpointManager(checkpoint, checkpoint_dir, 
                                            max_to_keep=None if dataset_valid is None else es_patience+2)
    
    evaluate = dataset_valid is not None
    loss_monitor = loss_metric(vae.config.uni_block_names, evaluate, verbose)        
    early_stopping = Early_Stopping(patience=es_patience, tolerance=es_tolerance, relative=es_relative)

    if not verbose:
        progbar = Progbar(num_epoch)
    
    start_time = time()
    for epoch in range(1,num_epoch+1):

        if verbose:
            progbar = Progbar(num_step_per_epoch)
            print('Train - Start of epoch %d' % (epoch,))
        else:
            if (epoch+1)%2==0 or epoch+1==num_epoch:
                progbar.update(epoch+1)

        # Iterate over the batches of the dataset.
        for step, (x, b, m) in enumerate(dataset_train):
            if not full_masks:
                m = tf.gather(vae.masks, m)
            m = vae.generate_mask(x, m)
            with tf.GradientTape() as tape:
                losses = vae(x, m, b, L=L)
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
            for step, (x, b, m) in enumerate(dataset_valid):
                if not full_masks:
                    m = tf.gather(vae.masks, m)
                m = vae.generate_mask(x, m, p=0.)
                losses = vae(x, m, b, L=L, training=False)
                loss_monitor(losses, 'val')

        loss_monitor.on_epoch_end(
            'Epoch {}, Time elapsed: {} minutes'.format(epoch, round((time() - start_time) / 60, 2))
        )
                
        if checkpoint_dir is not None:
            # When validation set is available, save the model winthin latest es_patience+1 epoches.        
            if dataset_valid is not None:
                save_path = manager.save()
                print("Saved checkpoint for epoch {}: {}".format(epoch, save_path))

                if early_stopping(float(loss_monitor['val'][0].result())):
                    print('Early stopping.')
                    break
            else:
                if int(checkpoint.step) % save_every_epoch == 0:
                    print(checkpoint.step)
                    save_path = manager.save()
                    print("Saved checkpoint for epoch {}: {}".format(epoch, save_path))

                    if eval_func is not None:
                        eval_func(vae)
            checkpoint.step.assign_add(1)
        
        loss_monitor.reset_states()

    print('Train Done.')
    return vae, loss_monitor.hist

