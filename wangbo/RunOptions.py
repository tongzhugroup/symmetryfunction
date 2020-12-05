#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 30 14:55:39 2020

@author: wang
"""
import tensorflow as tf
import numpy as np


global_tf_float_precision = tf.float64
global_np_float_precision = np.float64
global_ener_float_precision = np.float64
global_float_prec = 'double'

def global_cvt_2_tf_float(xx) :
    return tf.cast(xx, global_tf_float_precision)
def global_cvt_2_ener_float(xx) :
    return tf.cast(xx, global_ener_float_precision)

