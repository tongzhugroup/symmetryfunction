#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 30 11:46:44 2020

@author: wang
"""
import tensorflow as tf
import os

def j_must_have (jdata, key) :
    if not key in jdata.keys() :
        raise RuntimeError ("json database must provide key " + key )
    else :
        return jdata[key]
global_tf_float_precision=tf.float64
data_requirement = {}
activation_fn_dict = {
    "relu": tf.nn.relu,
    "relu6": tf.nn.relu6,
    "softplus": tf.nn.softplus,
    "sigmoid": tf.sigmoid,
    "tanh": tf.nn.tanh,
}

def get_activation_func(activation_fn):
    if activation_fn not in activation_fn_dict:
        raise RuntimeError(activation_fn+" is not a valid activation function")
    return activation_fn_dict[activation_fn]

#def expand_sys_str(root_dir):
#    matches = []
#    for root, dirnames, filenames in os.walk(root_dir, followlinks=True):
#        for filename in fnmatch.filter(filenames, 'type.raw'):
#            matches.append(root)
#    return matches

def get_precision(precision):
    if precision == "default":
        return  global_tf_float_precision
    elif precision == "float16":
        return tf.float16
    elif precision == "float32":
        return tf.float32
    elif precision == "float64":
        return tf.float64
    else:
        raise RuntimeError("%d is not a valid precision" % precision)
def add_data_requirement(key, 
                         ndof, 
                         atomic = False, 
                         must = False, 
                         high_prec = False,
                         type_sel = None,
                         repeat = 1) :
    data_requirement[key] = {'ndof': ndof, 
                             'atomic': atomic,
                             'must': must, 
                             'high_prec': high_prec,
                             'type_sel': type_sel,
                             'repeat': repeat,
    }    
class ClassArg () : 
    def __init__ (self) :
        self.arg_dict = {}
        self.alias_map = {}
    
    def add (self, 
             key,
             types_,
             alias = None,
             default = None, 
             must = False) :
        if type(types_) is not list :
            types = [types_]
        else :
            types = types_
        if alias is not None :
            if type(alias) is not list :
                alias_ = [alias]
            else:
                alias_ = alias
        else :
            alias_ = []

        self.arg_dict[key] = {'types' : types,
                              'alias' : alias_,
                              'value' : default, 
                              'must': must}
        for ii in alias_ :
            self.alias_map[ii] = key

        return self


    def _add_single(self, key, data) :
        vtype = type(data)
        if not(vtype in self.arg_dict[key]['types']) :
            # try the type convertion to the first listed type
            try :
                vv = (self.arg_dict[key]['types'][0])(data)
            except TypeError:
                raise TypeError ("cannot convert provided key \"%s\" to type %s " % (key, str(self.arg_dict[key]['types'][0])) )
        else :
            vv = data
        self.arg_dict[key]['value'] = vv

    
    def _check_must(self) :
        for kk in self.arg_dict:
            if self.arg_dict[kk]['must'] and self.arg_dict[kk]['value'] is None:
                raise RuntimeError('key \"%s\" must be provided' % kk)


    def parse(self, jdata) :
        print('jdata:',jdata)
        for kk in jdata.keys() :
            if kk in self.arg_dict :
                key = kk
                self._add_single(key, jdata[kk])
            else:
                if kk in self.alias_map: 
                    key = self.alias_map[kk]
                    self._add_single(key, jdata[kk])
        self._check_must()
        return self.get_dict()

    def get_dict(self) :
        ret = {}
        for kk in self.arg_dict.keys() :
            ret[kk] = self.arg_dict[kk]['value']
        return ret