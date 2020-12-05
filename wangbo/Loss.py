#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 30 11:26:52 2020

@author: wang
"""

import numpy as np
import tensorflow as tf
from common import ClassArg, add_data_requirement
from RunOptions import global_cvt_2_tf_float
from RunOptions import global_cvt_2_ener_float

class EnerStdLoss () :
    def __init__ (self, jdata, **kwarg) :
        self.starter_learning_rate = kwarg['starter_learning_rate']
        args = ClassArg()\
            .add('start_pref_e',        float,  default = 0.02)\
            .add('limit_pref_e',        float,  default = 1.00)\
            .add('start_pref_f',        float,  default = 1000)\
            .add('limit_pref_f',        float,  default = 1.00)\
            .add('start_pref_v',        float,  default = 0)\
            .add('limit_pref_v',        float,  default = 0)\
            .add('start_pref_ae',       float,  default = 0)\
            .add('limit_pref_ae',       float,  default = 0)\
            .add('start_pref_pf',       float,  default = 0)\
            .add('limit_pref_pf',       float,  default = 0)\
            .add('relative_f',          float)
        class_data = args.parse(jdata)
        self.start_pref_e = class_data['start_pref_e']
        self.limit_pref_e = class_data['limit_pref_e']
        self.start_pref_f = class_data['start_pref_f']
        self.limit_pref_f = class_data['limit_pref_f']
        self.start_pref_v = class_data['start_pref_v']
        self.limit_pref_v = class_data['limit_pref_v']
        self.start_pref_ae = class_data['start_pref_ae']
        self.limit_pref_ae = class_data['limit_pref_ae']
        self.start_pref_pf = class_data['start_pref_pf']
        self.limit_pref_pf = class_data['limit_pref_pf']
        self.relative_f = class_data['relative_f']
        self.has_e = (self.start_pref_e != 0 or self.limit_pref_e != 0)
        self.has_f = (self.start_pref_f != 0 or self.limit_pref_f != 0)
        self.has_v = (self.start_pref_v != 0 or self.limit_pref_v != 0)
        self.has_ae = (self.start_pref_ae != 0 or self.limit_pref_ae != 0)
        self.has_pf = (self.start_pref_pf != 0 or self.limit_pref_pf != 0)
        # data required
        add_data_requirement('energy', 1, atomic=False, must=self.has_e, high_prec=True)
        add_data_requirement('force',  3, atomic=True,  must=self.has_f, high_prec=False)
        add_data_requirement('virial', 9, atomic=False, must=self.has_v, high_prec=False)
        add_data_requirement('atom_ener', 1, atomic=True, must=self.has_ae, high_prec=False)
        add_data_requirement('atom_pref', 1, atomic=True, must=self.has_pf, high_prec=False, repeat=3)

    def build (self, 
               learning_rate,
               natoms,
               model_dict,
               label_dict,
               suffix):        
        energy = model_dict['energy']
       # force = model_dict['force']
       # virial = model_dict['virial']
       # atom_ener = model_dict['atom_ener']
        energy_hat = label_dict['energy:0']
       # force_hat = label_dict['force']
       # virial_hat = label_dict['virial']
       # atom_ener_hat = label_dict['atom_ener']
       # atom_pref = label_dict['atom_pref']
       # find_energy = label_dict['find_energy']
       # find_force = label_dict['find_force']
       # find_virial = label_dict['find_virial']
       # find_atom_ener = label_dict['find_atom_ener']                
       # find_atom_pref = label_dict['find_atom_pref']                

        l2_ener_loss = tf.reduce_mean( tf.square(energy - energy_hat), name='l2_'+suffix)
        #with tf.Session() as sess:
        #    sess.run(tf.global_variables_initializer())
        #    print(sess.run(l2_ener_loss))

   #     force_reshape = tf.reshape (force, [-1])
   #     force_hat_reshape = tf.reshape (force_hat, [-1])
   #     atom_pref_reshape = tf.reshape (atom_pref, [-1])
   #     diff_f = force_hat_reshape - force_reshape
   #     if self.relative_f is not None:            
   #         force_hat_3 = tf.reshape(force_hat, [-1, 3])
   #         norm_f = tf.reshape(tf.norm(force_hat_3, axis = 1), [-1, 1]) + self.relative_f
   #         diff_f_3 = tf.reshape(diff_f, [-1, 3])
   #         diff_f_3 = diff_f_3 / norm_f
   #         diff_f = tf.reshape(diff_f_3, [-1])
   #     l2_force_loss = tf.reduce_mean(tf.square(diff_f), name = "l2_force_" + suffix)
   #     l2_pref_force_loss = tf.reduce_mean(tf.multiply(tf.square(diff_f), atom_pref_reshape), name = "l2_pref_force_" + suffix)

   #     virial_reshape = tf.reshape (virial, [-1])
   #     virial_hat_reshape = tf.reshape (virial_hat, [-1])
   #     l2_virial_loss = tf.reduce_mean (tf.square(virial_hat_reshape - virial_reshape), name = "l2_virial_" + suffix)

   #     atom_ener_reshape = tf.reshape (atom_ener, [-1])
   #     atom_ener_hat_reshape = tf.reshape (atom_ener_hat, [-1])
   #     l2_atom_ener_loss = tf.reduce_mean (tf.square(atom_ener_hat_reshape - atom_ener_reshape), name = "l2_atom_ener_" + suffix)

   #     atom_norm  = 1./ global_cvt_2_tf_float(natoms[0]) 
   #     atom_norm_ener  = 1./ global_cvt_2_ener_float(natoms[0]) 
   #     pref_e = global_cvt_2_ener_float(find_energy * (self.limit_pref_e + (self.start_pref_e - self.limit_pref_e) * learning_rate / self.starter_learning_rate) )
   #     pref_f = global_cvt_2_tf_float(find_force * (self.limit_pref_f + (self.start_pref_f - self.limit_pref_f) * learning_rate / self.starter_learning_rate) )
   #     pref_v = global_cvt_2_tf_float(find_virial * (self.limit_pref_v + (self.start_pref_v - self.limit_pref_v) * learning_rate / self.starter_learning_rate) )
   #     pref_ae= global_cvt_2_tf_float(find_atom_ener * (self.limit_pref_ae+ (self.start_pref_ae-self.limit_pref_ae) * learning_rate / self.starter_learning_rate) )
   #     pref_pf= global_cvt_2_tf_float(find_atom_pref * (self.limit_pref_pf+ (self.start_pref_pf-self.limit_pref_pf) * learning_rate / self.starter_learning_rate) )

   #     l2_loss = 0
   #     more_loss = {}
   #     if self.has_e :
   #         l2_loss += atom_norm_ener * (pref_e * l2_ener_loss)
   #     more_loss['l2_ener_loss'] = l2_ener_loss
   #     if self.has_f :
   #         l2_loss += global_cvt_2_ener_float(pref_f * l2_force_loss)
   #     more_loss['l2_force_loss'] = l2_force_loss
   #     if self.has_v :
   #         l2_loss += global_cvt_2_ener_float(atom_norm * (pref_v * l2_virial_loss))
   #     more_loss['l2_virial_loss'] = l2_virial_loss
   #     if self.has_ae :
   #         l2_loss += global_cvt_2_ener_float(pref_ae * l2_atom_ener_loss)
   #     more_loss['l2_atom_ener_loss'] = l2_atom_ener_loss
   #     if self.has_pf :
   #         l2_loss += global_cvt_2_ener_float(pref_pf * l2_pref_force_loss)
   #     more_loss['l2_pref_force_loss'] = l2_pref_force_loss

   #     self.l2_l = l2_loss
   #     self.l2_more = more_loss
   #     return l2_loss, more_loss
        return l2_ener_loss