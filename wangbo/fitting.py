#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 30 20:00:01 2020

@author: wang
"""
import tensorflow as tf
import numpy as np

def one_layer(inputs, 
              outputs_size, 
              activation_fn=tf.nn.relu, 
              precision = tf.float32, 
              stddev=1.0,
              bavg=0.0,
              name='linear', 
              reuse=tf.AUTO_REUSE,
              seed=None, 
              use_timestep = False, 
              trainable = True,
              useBN = False):
    with tf.variable_scope(name, reuse=reuse):
        shape = inputs.get_shape().as_list()
        w = tf.get_variable('matrix', 
                            [shape[1], outputs_size], 
                            precision,
                            tf.random_normal_initializer(stddev=stddev/np.sqrt(shape[1]+outputs_size), seed = seed), 
                            trainable = trainable)
        b = tf.get_variable('bias', 
                            [outputs_size], 
                            precision,
                            tf.random_normal_initializer(stddev=stddev, mean = bavg, seed = seed), 
                            trainable = trainable)
        hidden = tf.matmul(inputs, w) + b
        if activation_fn != None and use_timestep :
            idt = tf.get_variable('idt',
                                  [outputs_size],
                                  precision,
                                  tf.random_normal_initializer(stddev=0.001, mean = 0.1, seed = seed), 
                                  trainable = trainable)
        if activation_fn != None:
            if useBN:
                None
                # hidden_bn = self._batch_norm(hidden, name=name+'_normalization', reuse=reuse)   
                # return activation_fn(hidden_bn)
            else:
                if use_timestep :
                    return tf.reshape(activation_fn(hidden), [-1, outputs_size]) * idt
                else :
                    return tf.reshape(activation_fn(hidden), [-1, outputs_size])                    
        else:
            if useBN:
                None
                # return self._batch_norm(hidden, name=name+'_normalization', reuse=reuse)
            else:
                return hidden
            
class fitting():
    def __init__(self,fitting_param,jdata,descrpt):
        self.descrpt = descrpt
        self.fitting_param = fitting_param
        self.fitting_precision = tf.float32
        self.fitting_activation_fn = tf.nn.relu
        self.seed = 32423342

    def get_dim_descrpt(self):
        return self.dim_descrpt

    def get_descrpt(self):
        Results=self.build()
        return Results

    def loss(self,fit_eners):
        self.energy=self.energy/10000.0
        l2_ener_loss = tf.reduce_mean( tf.square(self.energy - fit_eners), name='l2_energy')
        return l2_ener_loss

    def build_training(self,loss):
        optimizer = tf.train.AdamOptimizer(learning_rate = self.learning_rate)         
        train = optimizer.minimize(loss)
        train_loss=[];epochs=[]
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            for epoch in range(100):
                epochs.append(epoch)
                epoch_loss=0.0
                _,c=sess.run([train,loss])
                epoch_loss+=c
                train_loss.append(epoch_loss)
                print('Epoch %02d, Loss = %.6f' % (epoch, epoch_loss))
            #saver = tf.train.Saver(tf.global_variables(), max_to_keep=5)
            #path = saver.save(sess, "./lstm-attention.ckpt")
            #print("Saved model checkpoint to {}\n".format(path))
            
      #  meta_path = './lstm-attention.ckpt.meta'
      #  ckpt_path = './lstm-attention.ckpt'

      #  with tf.Session() as sess: 
      #      saver=tf.train.import_meta_graph(meta_path)
      #      saver.restore(sess, ckpt_path)
      #      sess.run(fit_eners, feed_dict=self.Results)
    
    def make(self,inputs, atom_types, reuse = None,suffix = ''):       
        #inputs=self.get_descrpt()
        self.dim_descrpt = self.descrpt.filter_neuron[-1] * self.descrpt.n_axis_neuron
        natoms=self.descrpt.frames_natoms[1]
        inputs = tf.cast(tf.reshape(inputs, [-1, self.dim_descrpt * natoms]), self.fitting_precision)
        fit_atom_eners=[]
        for i in range(inputs.shape[0]):
            atom_ener=self.one_frame(inputs[0], atom_types)
            fit_atom_eners.append(atom_ener)
        #print('fit_atom_energys:',fit_atom_eners)
        fit_atom_eners=tf.reshape(fit_atom_eners,[-1,natoms])
        print('fit_atom_energys:',fit_atom_eners)

        fit_eners=tf.reduce_sum(fit_atom_eners, 1)
        print('fit_eners:',fit_eners.shape)
        #loss_ener=self.loss(fit_eners)
        #self.build_training(loss_ener)
        return fit_atom_eners

    def one_frame(self,
                  input,
                  atom_types,
                  reuse = tf.AUTO_REUSE,
                  suffix = ''):
        
        input = tf.reshape(input,[self.descrpt.frames_natoms[1],self.dim_descrpt])
        #print('input_shape:',input.shape)
        atom_ener=[]
        for at in range(self.descrpt.types.shape[0]):
            inputs_i = input[at]   
            type_i = atom_types[at]
            inputs_i = tf.reshape(inputs_i, [-1, self.dim_descrpt])
            layer = inputs_i

            for ii in range(0,len(self.descrpt.filter_neuron)) :
                if ii >= 1 and self.descrpt.filter_neuron[ii] == self.descrpt.filter_neuron[ii-1] :
                    layer+= one_layer(layer, self.descrpt.filter_neuron[ii], name='layer_'+str(ii)+'_type_'+str(type_i)+suffix, reuse=reuse, seed = self.seed,  activation_fn = self.fitting_activation_fn, precision = self.fitting_precision)
                else :
                    layer = one_layer(layer, self.descrpt.filter_neuron[ii], name='layer_'+str(ii)+'_type_'+str(type_i)+suffix, reuse=reuse, seed = self.seed, activation_fn = self.fitting_activation_fn, precision = self.fitting_precision)
            final_layer = one_layer(layer, 1, activation_fn = None, name='final_layer_type_'+str(type_i)+suffix, reuse=reuse, seed = self.seed, precision = self.fitting_precision)
            
            atom_ener.append(final_layer)
        atom_ener=tf.reshape(atom_ener,[-1])
        print('atom_ener:',atom_ener)
        return atom_ener
                      