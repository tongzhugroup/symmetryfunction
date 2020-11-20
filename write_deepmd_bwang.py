import numpy as np
import os
import matplotlib.pyplot as plt
import tensorflow as tf
tf.compat.v1.disable_v2_behavior()
            
class descrpt(object):
    def __init__(self):
        self.data_path='./set.000'
        self.rcut_r=6.0
        self.rcut_smth=5.5
        self.ntypes=2
        self.filter_neuron=[20,30,40]
        self.n_neuron=[30,50,50]
        self.n_axis_neuron=3
        self.filter_precision=tf.float32
        self.seed=1
        self.learning_rate=0.001

    def r(self,rij):
        if rij<self.rcut_smth:
            return 1/rij
        elif rij>self.rcut_smth and rij<self.rcut_r:
            return (1/rij)*(0.5*np.cos(np.pi*(rij-self.rcut_smth)/(self.rcut_r-self.rcut_smth))+0.5)
        elif rij>self.rcut_r:
            return 0

    def Rrow(self,rij_):
        row=[]
        xij=rij_[0];yij=rij_[1];zij=rij_[2]
        rij=np.sqrt(xij**2+yij**2+zij**2)
        s=self.r(rij)
        row.append(s/rij);row.append(s*xij/rij);row.append(s*yij/rij);row.append(s*zij/rij)
        return row
 
    def get_dim_out (self) :
        return self.filter_neuron[-1] * self.n_axis_neuron
    
    def get_data(self):
        coord=np.array(np.load(os.path.join(self.data_path,'coord.npy')),dtype=float)
        force=np.array(np.load(os.path.join(self.data_path,'force.npy')),dtype=float)
        energy=np.array(np.load(os.path.join(self.data_path,'energy.npy')),dtype=float)
        box=np.array(np.load(os.path.join(self.data_path,'box.npy')),dtype=float)
        types=[];print('energy:',energy.shape)
        self.types=types
        lines=open(os.path.join(self.data_path,'../type.txt'),'r').readlines()
        for line in lines:
            line=line.strip().split()
            types.extend(line)
            types=list(map(int,types))
            self.natoms=len(types)
        return coord,energy,force,box,types

    def get_natoms(self):
        return self.natoms
    
    def build(self,reuse=None):
        coord,energy,force,box,types=self.get_data()
        self.energy=energy
        natoms=self.get_natoms()
        coord=np.reshape(coord,[-1,natoms,3])
        box=np.reshape(box,[-1,9])
        nnei=natoms-1
        R_total=[]
        for fr in range(coord.shape[0]):
            Rfr=[]
            for i in range(coord.shape[1]):
                ri_=coord[fr][i]
                for j in range(coord.shape[1]):
                    rj_=coord[fr][j]
                    if i!=j:
                        rij_=ri_-rj_
                        Rrowl=self.Rrow(rij_)
                        Rfr.append(Rrowl)
            R_total.append(Rfr)

        R_total=tf.reshape(R_total,[-1,natoms,nnei*4])
        print('R_total:',R_total.shape)
        Results=[]
        for i in range(R_total.shape[0]):
            result,qmat=self.filter(R_total[i],self.natoms)
            Results.append(result)
        Results=tf.reshape(Results,[-1,natoms,self.filter_neuron[-1] * self.n_axis_neuron])
        print('Results:',Results.shape)
       # with tf.Session() as sess:
       #     sess.run(tf.global_variables_initializer())
       #     print(sess.run(Results))
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
        shape = inputs.get_shape().as_list()
        outputs_size = [1] + self.filter_neuron
        outputs_size_2 = self.n_axis_neuron
        with tf.variable_scope(name, reuse=reuse):
          start_index = 0
          xyz_scatter_total = []
          for type_i in range(self.ntypes):
            print('self.types:',self.types)
            nei=len(self.types)-1
            if str(type_i)=='0':
                nei_type_i=self.types.count(str(type_i))-1
            else:
                nei_type_i=self.types.count(str(type_i))
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
          print(result)
        return result, qmat

def one_layer(inputs, 
              outputs_size, 
              activation_fn=tf.nn.tanh, 
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
            
class fitting(descrpt):
    def __init__(self):
        descrpt.__init__(self)
        self.fitting_precision=tf.float32
        self.dim_descrpt = self.get_dim_out()
        self.fitting_activation_fn=tf.nn.tanh

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
            
            plt.plot(epochs,train_loss)  
            plt.ylabel('Loss')
            plt.xlabel('Epoch')
            plt.title('Train Energy RMSE')
            plt.show()
            
      #  meta_path = './lstm-attention.ckpt.meta'
      #  ckpt_path = './lstm-attention.ckpt'

      #  with tf.Session() as sess: 
      #      saver=tf.train.import_meta_graph(meta_path)
      #      saver.restore(sess, ckpt_path)
      #      sess.run(fit_eners, feed_dict=self.Results)
    
    def make(self, 
               reuse = None,
               suffix = '') :       
        inputs=self.get_descrpt()
        natoms=self.natoms
        inputs = tf.cast(tf.reshape(inputs, [-1, self.dim_descrpt * natoms]), self.fitting_precision)
        fit_atom_eners=[]
        for i in range(inputs.shape[0]):
            atom_ener=self.one_frame(inputs[0])
            fit_atom_eners.append(atom_ener)
        print('fit_atom_energys:',fit_atom_eners)
        fit_atom_eners=tf.reshape(fit_atom_eners,[-1,natoms])
        print('fit_atom_energys:',fit_atom_eners)

        fit_eners=tf.reduce_sum(fit_atom_eners, 1)
        print('fit_eners:',fit_eners.shape)
        loss_ener=self.loss(fit_eners)
        self.build_training(loss_ener)

    def one_frame(self,
                  input,
                  reuse = tf.AUTO_REUSE,
                  suffix = ''):
        
        input = tf.reshape(input,[self.natoms,self.dim_descrpt])
        #print('input_shape:',input.shape)
        atom_ener=[]
        for at in range(len(self.types)):
            type_i=self.types[at]

            inputs_i=input[at]   
            
            inputs_i = tf.reshape(inputs_i, [-1, self.dim_descrpt])
            layer = inputs_i

            for ii in range(0,len(self.n_neuron)) :
                if ii >= 1 and self.n_neuron[ii] == self.n_neuron[ii-1] :
                    layer+= one_layer(layer, self.n_neuron[ii], name='layer_'+str(ii)+'_type_'+str(type_i)+suffix, reuse=reuse, seed = self.seed,  activation_fn = self.fitting_activation_fn, precision = self.fitting_precision)
                else :
                    layer = one_layer(layer, self.n_neuron[ii], name='layer_'+str(ii)+'_type_'+str(type_i)+suffix, reuse=reuse, seed = self.seed, activation_fn = self.fitting_activation_fn, precision = self.fitting_precision)
            final_layer = one_layer(layer, 1, activation_fn = None, name='final_layer_type_'+str(type_i)+suffix, reuse=reuse, seed = self.seed, precision = self.fitting_precision)
            
            atom_ener.append(final_layer)
        atom_ener=tf.reshape(atom_ener,[-1])
        print('atom_ener:',atom_ener)
        return atom_ener
                      
if __name__ == '__main__':
    tf.reset_default_graph()
    Des=descrpt()
    Des.build()
    Fitting=fitting()
    Fitting.make()