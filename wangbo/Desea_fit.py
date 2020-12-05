import numpy as np
import os
import matplotlib.pyplot as plt
import tensorflow as tf
tf.compat.v1.disable_v2_behavior()
from common import ClassArg, get_activation_func, get_precision

class descrpt(object):
    def __init__(self, jdata):
        args = ClassArg()\
               .add('sel',      list,   must = True) \
               .add('rcut',     float,  default = 6.0) \
               .add('rcut_smth',float,  default = 5.5) \
               .add('neuron',   list,   default = [10, 20, 40]) \
               .add('axis_neuron', int, default = 4, alias = 'n_axis_neuron') \
               .add('resnet_dt',bool,   default = False) \
               .add('trainable',bool,   default = True) \
               .add('seed',     int) \
               .add('type_one_side', bool, default = False) \
               .add('exclude_types', list, default = []) \
               .add('set_davg_zero', bool, default = False) \
               .add('activation_function', str,    default = 'tanh') \
               .add('precision', str, default = "default")
        class_data = args.parse(jdata)
        #self.prod_force_module = tf.load_op_library('prod_force_se_a.so')
        self.sel_a = class_data['sel']
        self.rcut_r = class_data['rcut']
        self.rcut_r_smth = class_data['rcut_smth']
        self.filter_neuron = class_data['neuron']
        self.n_axis_neuron = class_data['axis_neuron']
        self.filter_resnet_dt = class_data['resnet_dt']
        self.seed = class_data['seed']
        self.trainable = class_data['trainable']
        self.filter_activation_fn = get_activation_func(class_data['activation_function'])
        self.filter_precision = get_precision(class_data['precision'])
        exclude_types = class_data['exclude_types']
        self.exclude_types = set()
        for tt in exclude_types:
            assert(len(tt) == 2)
            self.exclude_types.add((tt[0], tt[1]))
            self.exclude_types.add((tt[1], tt[0]))
        self.set_davg_zero = class_data['set_davg_zero']
        self.type_one_side = class_data['type_one_side']
        if self.type_one_side and len(exclude_types) != 0:
            raise RuntimeError('"type_one_side" is not compatible with "exclude_types"')
         # descrpt config
        self.sel_r = [ 0 for ii in range(len(self.sel_a)) ]
        self.ntypes = len(self.sel_a)
        assert(self.ntypes == len(self.sel_r))
        self.rcut_a = -1
        # numb of neighbors and numb of descrptors
        self.nnei_a = np.cumsum(self.sel_a)[-1]
        self.nnei_r = np.cumsum(self.sel_r)[-1]
        self.nnei = self.nnei_a + self.nnei_r
        self.ndescrpt_a = self.nnei_a * 4
        self.ndescrpt_r = self.nnei_r * 1
        self.ndescrpt = self.ndescrpt_a + self.ndescrpt_r
        self.useBN = False
        self.dstd = None
        self.davg = None

    def get_rcut (self) :
        return self.rcut_r

    def get_ntypes (self) :
        return self.ntypes

    def get_dim_out (self) :
        return self.filter_neuron[-1] * self.n_axis_neuron

    def get_dim_rot_mat_1 (self) :
        return self.filter_neuron[-1]

    def get_nlist (self) :
        return self.nlist, self.rij, self.sel_a, self.sel_r

    def r(self,rij):
        if rij < self.rcut_r_smth:
            return 1/rij
        elif rij > self.rcut_r_smth and rij<self.rcut_r:
            return (1/rij)*(0.5*np.cos(np.pi*(rij-self.rcut_r_smth)/(self.rcut_r-self.rcut_r_smth))+0.5)
        elif rij > self.rcut_r:
            return 0
        tf.case({tf.less(rij, self.rcut_r_smth): f1, tf.greater(rij, self.rcut_r): f2},default=f3, exclusive=True)
    
    def Rrow(self,rij_):
        row=[]
        xij=rij_[0];yij=rij_[1];zij=rij_[2]
        rij=tf.sqrt(xij**2+yij**2+zij**2)
        s=self.r(rij)
        print('s:',s)
        row.append(s/rij);row.append(s*xij/rij);row.append(s*yij/rij);row.append(s*zij/rij)
        return s/rij, s*xij/rij, s*yij/rij, s*zij/rij
 
    def Rrowf(self,rij_):
        return 0, 0, 0, 0
    
    def get_dim_out (self) :
        return self.filter_neuron[-1] * self.n_axis_neuron

    def get_natoms(self):
        return self.natoms
    
    def build(self, coord, types, frames_natoms, box, atom_types, reuse=None):
        print('types:',types)
        frames_natoms = [2,21] #########temp addion
        self.types = tf.reshape(types,[frames_natoms[1]])
        natoms = frames_natoms[1]
        coord = tf.reshape(coord,[frames_natoms[0],frames_natoms[1],3])
        box = tf.reshape(box, [frames_natoms[0],9])
        self.frames_natoms = frames_natoms
        
        #type_maxs = self.sel_a
        print('coord:',coord)
        R_total = []
        Rij_total = []
        for fr in range(coord.shape[0]):
            for i in range(coord.shape[1]):
                Ri_total = []
                R_i = np.array([[0,0,0,0]])
                #type_i = types[i];
                ri_ = coord[fr][i]
                for type_a in range(self.ntypes):
                    print('typea:',type_a)
                    max_types = self.sel_a[type_a]
                    type_a_matric = np.zeros((max_types, 4))
                    index = -1
                    for j in range(coord.shape[1]):
                        rj_ = coord[fr][j]
                        type_j = types[j]
                        print('type_j:',type_j)
                        if i != j and type_j == type_a:
                            index += 1
                            rij_ = ri_-rj_
                            Ri_total.append([rij_[0], rij_[1], rij_[2]])
                            rij = tf.sqrt(tf.square(rij_[0])+tf.square(rij_[1])+tf.square(rij_[2]))                            
                            if rij < self.rcut_r:
                                s1, s2, s3, s4 = self.Rrow(rij_)
                                type_a_matric[index][0] = s1
                                type_a_matric[index][1] = s2
                                type_a_matric[index][2] = s3
                                type_a_matric[index][3] = s4
                    R_i = np.append(R_i, type_a_matric, axis=0)
                R_ii = R_i[1:]
                R_total.append(R_ii)
                Rij_total.append(Ri_total)
        print('Rij_total:',np.array(Rij_total))
            
        R_total = np.array(R_total).reshape(-1,natoms, self.nnei*4)
        self.descrpt_reshape = R_total
        #nframes*natoms*(nnei*4)
        print('R_total:',R_total.shape)
        self.R_total = R_total
        self.Total_inputs = []
        Results = []
        for i in range(R_total.shape[0]):
            t = tf.convert_to_tensor(R_total[i], tf.float64, name='t')
            result,qmat = self.filter(t, natoms)
            Results.append(result)
        Results = tf.reshape(Results,[-1,natoms,self.filter_neuron[-1] * self.n_axis_neuron])
        print('Results:',Results.shape)
        return Results
          
    def filter(self, 
                   inputs,
                   natoms,
                   activation_fn=tf.nn.tanh, 
                   stddev=1.0,
                   bavg=0.0,
                   name='linear', 
                   reuse=tf.AUTO_REUSE,
                   seed=None, 
                   trainable = True):
        self.inputs = inputs
        self.Total_inputs.append(self.inputs)
        shape = inputs.get_shape().as_list()
        outputs_size = [1] + self.filter_neuron
        outputs_size_2 = self.n_axis_neuron
        with tf.variable_scope(name, reuse=reuse):
          start_index = 0
          xyz_scatter_total = []
          for type_i in range(self.ntypes):
            print('self.ntypes:',self.ntypes)
            nei = self.nnei
            nei_type_i = self.sel_a[type_i]
            print('nei_type:',nei_type_i) 
            inputs_i = tf.slice (inputs,
                                 [ 0, start_index*      4],
                                 [-1, nei_type_i* 4] )
            start_index += nei_type_i
            inputs_reshape = tf.reshape(inputs_i, [-1, 4])
            xyz_scatter = tf.reshape(tf.slice(inputs_reshape, [0,0],[-1,1]),[-1,1])
            for ii in range(1, len(outputs_size)):
                w = tf.get_variable('matrix_'+str(ii)+'_'+str(type_i), 
                                  [outputs_size[ii - 1], outputs_size[ii]], 
                                  self.filter_precision,
                                  tf.random_normal_initializer(stddev=stddev/np.sqrt(outputs_size[ii]+outputs_size[ii-1]), seed = seed), 
                                  trainable = trainable)
                b = tf.get_variable('bias_'+str(ii)+'_'+str(type_i), 
                                  [1, outputs_size[ii]], 
                                  self.filter_precision,
                                  tf.random_normal_initializer(stddev=stddev, mean = bavg, seed = seed), 
                                  trainable = trainable)
                hidden = tf.reshape(activation_fn(tf.matmul(xyz_scatter, w) + b), [-1, outputs_size[ii]])
                if outputs_size[ii] == outputs_size[ii-1]:
                        xyz_scatter += hidden
                elif outputs_size[ii] == outputs_size[ii-1] * 2: 
                        xyz_scatter = tf.concat([xyz_scatter,xyz_scatter], 1) + hidden
                else:
                    xyz_scatter = hidden
            xyz_scatter = tf.reshape(xyz_scatter, (-1, nei_type_i, outputs_size[-1]))
            print('xyz_scatter_one:',xyz_scatter.shape)
            xyz_scatter_total.append(xyz_scatter)

          xyz_scatter = tf.concat(xyz_scatter_total, axis=1)
          print('xyz_scatter:',xyz_scatter.shape)
          inputs_reshape = tf.reshape(inputs, [-1, nei, 4])
          print('inputs_reshape:',inputs_reshape.shape)
          xyz_scatter_1 = tf.matmul(inputs_reshape, xyz_scatter, transpose_a = True)
          xyz_scatter_1 = xyz_scatter_1 * (4.0 / shape[1])
          xyz_scatter_2 = tf.slice(xyz_scatter_1, [0,0,0],[-1,-1,outputs_size_2])
          qmat = tf.slice(xyz_scatter_1, [0,1,0], [-1, 3, -1])
          qmat = tf.transpose(qmat, perm = [0, 2, 1])
          result = tf.matmul(xyz_scatter_1, xyz_scatter_2, transpose_a = True)
          result = tf.reshape(result, [-1, outputs_size_2 * outputs_size[-1]])
          print(outputs_size_2,outputs_size[-1])
          print(result)
        return result, qmat

    def prod_force_virial(self, atom_ener, natoms) :
        #self.descrpt_reshape = tf.convert_to_tensor(self.inputs)
        print('atom_ener:',atom_ener)
        print('inputs:',self.R_total)
        grads = []
        for i in range(len(self.Total_inputs)):
        #[net_deriv] = tf.gradients (atom_ener[0], self.inputs)
            [net_deriv] = tf.gradients (atom_ener[i], self.Total_inputs[i])
            grads.append([net_deriv])
            print('net_deriv:', [net_deriv])
        grads = tf.reshape(grads, [-1, self.frames_natoms[1], self.nnei*4])
        print('grad:', grads)
        #net_deriv_reshape = tf.reshape (net_deriv, [-1, self.frames_natoms[0] * self.nnei])
       # with tf.Session() as sess:
       #     sess.run(tf.global_variables_initializer())
            #print('atom_ener:',sess.run(atom_ener))
            #print('descrpt_reshape:',sess.run(self.descrpt_reshape))
       #     print('net_deriv:',sess.run(list([net_deriv])))
      #  force \
      #      = self.prod_force_module(net_deriv_reshape,
      #                               self.descrpt_deriv,
      #                               self.nlist,
      #                               self.frames_natoms,
      #                               n_a_sel = self.nnei_a,
      #                               n_r_sel = self.nnei_r)