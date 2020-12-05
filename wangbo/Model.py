#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 30 11:26:20 2020

@author: wang
"""
import os
import numpy as np
import tensorflow as tf
from common import ClassArg

class Model():
    model_type = 'ener'   
    def __init__(self, jdata, descrpt, fitting):
        self.descrpt = descrpt
        self.rcut = self.descrpt.get_rcut()
        self.ntypes = self.descrpt.get_ntypes()
        self.fitting = fitting
        
        args = ClassArg()\
               .add('type_map', list , default = [])\
               .add('data_stat_nbatch', int , default = 10)\
               .add('data_stat_protect', float, default = 1e-2)\
               .add('use_srtab', str)
        class_data = args.parse(jdata)
        self.type_map = class_data['type_map']
        self.srtab_name = class_data['use_srtab']
        self.data_stat_nbatch = class_data['data_stat_nbatch']
        self.data_stat_protect = class_data['data_stat_protect']

    def get_rcut (self) :
        return self.rcut

    def get_ntypes (self) :
        return self.ntypes

    def get_type_map (self) :
        return self.type_map

    def global_cvt_2_ener_float(xx) :
        return tf.cast(xx, np.float64)
        
    def build(self,
              coord_,
              atype_,
              natoms,
              box,
              mesh,
              input_dict,
              atom_types,
              suffix = '',
              reuse = None):

        coord = tf.reshape(coord_, [-1, natoms[1] , 3])
        atype = tf.reshape(atype_, [natoms[1]])
        
        dout = self.descrpt.build(coord, atype, natoms, box, atom_types)
        
        atom_ener = self.fitting.make(dout, atom_types)
        energy_raw = atom_ener
        energy_raw = tf.reshape(energy_raw, [-1, natoms[1]], name = 'o_atom_energy'+suffix)
        energy = tf.reduce_sum(energy_raw, axis=1, name='o_energy'+suffix)
        print('energy_fit:',energy)
        
        self.descrpt.prod_force_virial(atom_ener, natoms) ####
        
        
        model_dict = {}
        model_dict['energy'] = energy
        model_dict['coord'] = coord
        model_dict['atype'] = atype
        #print('model_dict:',model_dict)
        
        model_exp = {}
        model_exp['energy'] = energy
        model_exp['coord'] = coord
        model_exp['atype'] = atype
        #print('model_exp:',model_exp)
        return model_dict, model_exp