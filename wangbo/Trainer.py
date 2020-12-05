#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 30 11:23:02 2020

@author: wang
"""
import json,os
import numpy as np
import tensorflow as tf
from Desea_fit import descrpt
from fitting import fitting
from Model import Model
from Loss import EnerStdLoss
from common import j_must_have, ClassArg
from LearningRate import LearningRateExp

class NNPTrainer(object):
    def __init__(self,
                 jdata):
        self._init_param(jdata)
        self.data_path = './set.000'

    def _init_param(self,jdata):
        model_param = j_must_have(jdata, 'model')
        descrpt_param = j_must_have(model_param, 'descriptor')
        fitting_param = j_must_have(model_param, 'fitting_net')
        
        descrpt_type = j_must_have(descrpt_param, 'type')
        if descrpt_type == 'se_a':
            self.descrpt = descrpt(descrpt_param)
        else:
            raise RuntimeError('unknow model type '+ descrpt_type)
            
        try:
            fitting_type = fitting_param['type']
        except:
            fitting_type = 'ener'
        if fitting_type == 'ener':
            self.fitting = fitting(fitting_param, jdata, self.descrpt)
        else:
            raise RuntimeError('unknow fitting type' +fitting_type)
            
        if fitting_type == Model.model_type:
            self.model = Model(model_param, self.descrpt, self.fitting)
        else:
            raise RuntimeError('get unknown fitting type when building model')
        
        lr_param = j_must_have(jdata, 'learning_rate')
        try:
            lr_type = lr_param['type']
        except:
            lr_type = 'exp'
        if lr_type == 'exp':
            self.lr = LearningRateExp(lr_param)
        else:
            raise RuntimeError('unkown learning_rate type '+lr_type)
        
        try:
            loss_param = jdata['loss']
            loss_type = loss_param.get('type','std')
        except:
            loss_param = None
            loss_type = 'std'
        
        if fitting_type == 'ener':
            if loss_type == 'std':
                self.loss = EnerStdLoss(loss_param, starter_learning_rate = self.lr.start_lr())
            else:
                raise RuntimeError('unknow loss type')
        else:
            raise RuntimeError('get unknow fitting type when building loss function')
            
        training_param =j_must_have(jdata, 'training')
        
        tr_args = ClassArg()\
                  .add('numb_test', int, default = 1)\
                  .add('disp_file', str, default = 'lcurve.out')\
                  .add('disp_freq', int, default = 100)\
                  .add('save_freq', int, default = 1000)\
                  .add('save_ckpt', str, default = 'model.ckpt')\
                  .add('display_in_traning', bool, default = True)\
                  .add('timing_in_traning', bool, default = True)\
                  .add('profiling', bool, default = False)\
                  .add('profiling_file', str, default = 'timeline.json')\
                  .add('sys_probs', list)\
                  .add('auto_prob_style', str, default = 'prob_sys_size')
        tr_data = tr_args.parse(training_param)
        print('tr_data:',tr_data)
        self.numb_test = tr_data['numb_test']
        self.disp_file = tr_data['disp_file']
        self.disp_freq = tr_data['disp_freq']
        self.save_freq = tr_data['save_freq']
        self.saveckpt = tr_data['save_ckpt']
        self.timing_in_traning = tr_data['timing_in_traning']
        self.profiling = tr_data['profiling']
        self.profiling_file = tr_data['profiling_file']
        self.sys_probs = tr_data['sys_probs']
        self.auto_prob_style = tr_data['auto_prob_style']
        self.userBN = False
            
    def _message(self,information):
        print(information)
        
    def building(self,stop_batch=None):
        self.stop_batch = stop_batch
        self.ntypes = self.model.get_ntypes()            
        self._build_lr()
        self._build_network()
        self._build_training()
 
    def _build_lr(self):
        self._extra_train_ops = []
        self.global_step = tf.train.get_or_create_global_step()
        self.learning_rate = self.lr.build(self.global_step, self.stop_batch)
        self._message("built lr")

    def _make_default_mesh(self):
        self.default_mesh = []
        

    def get_data(self):
        data_dict = {}
        coord_ = np.array(np.load(os.path.join(self.data_path,'coord.npy')),dtype=float)
        force_ = np.array(np.load(os.path.join(self.data_path,'force.npy')),dtype=float)
        energy_ = np.array(np.load(os.path.join(self.data_path,'energy.npy')),dtype=float)/10000
        box_ = np.array(np.load(os.path.join(self.data_path,'box.npy')),dtype=float)
        atype_ = []; natoms=[]
        natoms.append(force_.shape[0]);natoms.append(int(force_.shape[1]/3))
        self.frames = force_.shape[0]
        print('natoms:',natoms)
        lines=open(os.path.join(self.data_path,'type.raw'),'r').readlines()
        for line in lines:
            line=line.strip().split()
            atype_.extend(line)
            atype_ = list(map(int, atype_))
            self.natoms=len(atype_)
        self.atype = atype_
        self.energy_ = energy_
        self.force_ = force_
        
        data_dict['energy:0'] = energy_
        data_dict['force:0'] = force_
        data_dict['coord:0'] = coord_
        data_dict['box:0'] = box_
        data_dict['type:0'] = atype_
        data_dict['natoms_vec:0'] = natoms
        data_dict['default_mesh:0'] = []
        data_dict['is_training:0'] = True
        return data_dict
       
    def _build_network(self):
        data_dict = self.get_data()
        
        self.place_holders = {}
        self.place_holders['coord'] = tf.placeholder(tf.float64, [self.frames, self.natoms*3], name = 'coord')
        self.place_holders['force'] = tf.placeholder(tf.float64, [self.frames, self.natoms*3], name = 'force')
        self.place_holders['energy'] = tf.placeholder(tf.float64, [self.frames], name = 'energy')
        self.place_holders['box'] = tf.placeholder(tf.float64, [self.frames, 9], name = 'box')
        self.place_holders['type'] = tf.placeholder(tf.int32,   [self.natoms], name='type')
        self.place_holders['natoms_vec'] = tf.placeholder(tf.int32, [2] , name='natoms_vec')
        self.place_holders['default_mesh'] = tf.placeholder(tf.int32,   [None], name='default_mesh')
        self.place_holders['is_training'] = tf.placeholder(tf.bool, name='is_training')
            
        self.model_pred, self.model_exp\
        = self.model.build (self.place_holders['coord'],
                            self.place_holders['type'],
                            self.place_holders['natoms_vec'],
                            self.place_holders['box'],
                            self.place_holders['default_mesh'],
                            self.place_holders,
                            self.atype,
                            suffix = "", 
                            reuse = False)

        self.l2_l\
        = self.loss.build (self.learning_rate,
                          self.descrpt.frames_natoms, 
                          self.model_pred,
                          data_dict,
                          suffix = "test")
        self._message("built network")

    def _build_training(self):
        trainable_variables = tf.trainable_variables()
        optimizer = tf.train.AdamOptimizer(learning_rate = self.learning_rate)
        grads = tf.gradients(self.l2_l, trainable_variables)
        apply_op = optimizer.apply_gradients (zip(grads, trainable_variables),
                                              global_step = self.global_step)
        train_ops = [apply_op] + self._extra_train_ops
        self.train_op = tf.group(*train_ops)
        self._message("built training")

    def train(self):
        data_dict = self.get_data()
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            curbatch = sess.run(self.global_step)
            self.curbatch = curbatch
            for epoch in range(100):
                _, ener_loss=sess.run([self.train_op, self.l2_l], feed_dict = data_dict)
                print('ener_loss:',ener_loss)


if __name__ == '__main__':
    tf.reset_default_graph()
    with open('input.json', 'r') as fp:
        jdata = json.load(fp)
    model = NNPTrainer(jdata)
    model.building()
    model.train()