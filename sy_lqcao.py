import numpy as np
import math
import pandas as pd
from ase.db import connect
from ase import Atom
import tensorflow as tf
#import matplotlib.pyplot  as plt
#import tensorflow.compat.v1 as tf
#tf.compat.v1.disable_eager_execution()
#tf.compat.v1.Session()
#tf.compat.v1.Variable()
####dataset
a=[]
x=[]
y=[]
z=[]
E_reference=[]
F_reference=[]
path_to_db='/home/lqcao/work/symmetry-function/iso17/reference.db'
with connect(path_to_db) as conn:
       for row in conn.select(limit=3):
           for  i in row.toatoms():
               atoms=i
               x.append(atoms.position[0])
               y.append(atoms.position[1])
               z.append(atoms.position[2])
               a.append(atoms.symbol)
           E_reference.append(row['total_energy'])
x=np.reshape(z,(-1,19))
y=np.reshape(y,(-1,19))
z=np.reshape(z,(-1,19))
a=np.reshape(a,(-1,19))

####symmetry functions
elta=[5.00]
Rs=[2.0,3.0]
zeta=[1,3]

def G1(rc):
    g11=[]
    for ii in range(len(a)):
        for i in range(len(a[ii])):
            fc=[]
            for j in range(len(a[ii])):
                if i!= j:
                    Rij=((x[ii][i]-x[ii][j])**2+(y[ii][i]-y[ii][j])**2+(z[ii][i]-z[ii][j])**2)**0.5
                    if Rij <= float(rc):
                        fc.append(0.5*(math.cos(math.pi*Rij/float(rc))+1))
                    else:
                        fc.append(float(0.0))
            f1=sum(fc)
            g11.append(f1)
    g11=np.reshape(g11,(-1,19))
    return g11

#print((G1(6.0)))
def G2(rc,elta,Rs):
    g22=[]
    for ii in range(len(a)):
        for i in range(len(a[ii])):
            g2=[]
            for j in range(len(a[ii])):
                if i!= j:
                    Rij=((x[ii][i]-x[ii][j])**2+(y[ii][i]-y[ii][j])**2+(z[ii][i]-z[ii][j])**2)**0.5
                    if Rij <= float(rc):
                        fc=(0.5*(math.cos(math.pi*Rij/float(rc))+1))
                        g2.append(math.exp(-elta*((i-Rs)**2))*fc)
                    else:
                        fc=float(0.0)
                        g2.append(math.exp(-elta*((i-Rs)**2))*fc)
            g2_value=sum(g2)
            g22.append(g2_value) 
    g22=np.reshape(g22,(-1,19))
    return g22
#print(G2(6.0,1,3))
def G4(rc,elta,lam,zeta):
    g44=[]
    the=[]
    for ii in range(len(a)):
        for i in range(len(a[ii])):
            g4=[]
            for j in range(len(a[ii])):
                for k in range(len(a[ii])):
                    if i != j and i != k and j != k:
                        Rij=((x[ii][i]-x[ii][j])**2+(y[ii][i]-y[ii][j])**2+(z[ii][i]-z[ii][j])**2)**0.5
                        Rik=((x[ii][i]-x[ii][k])**2+(y[ii][i]-y[ii][k])**2+(z[ii][i]-z[ii][k])**2)**0.5
                        Rjk=((x[ii][j]-x[ii][k])**2+(y[ii][j]-y[ii][k])**2+(z[ii][j]-z[ii][k])**2)**0.5
                        if Rij > float(rc) or Rik > float(rc)  or Rjk  > float(rc):
                            fc=float(0.0)
                            g4.append(fc)
                        else:
                            fc1=0.5*(math.cos(math.pi*Rij/float(rc))+1)
                            fc2=0.5*(math.cos(math.pi*Rik/float(rc))+1)
                            fc3=0.5*(math.cos(math.pi*Rjk/float(rc))+1)
                            d=((x[ii][j]-x[ii][i])*(x[ii][k]-x[ii][i])+(y[ii][j]-y[ii][i])*(y[ii][k]-y[ii][i])+(z[ii][j]-z[ii][i])*(z[ii][k]-z[ii][i]))
                            #print(d)
                            theta=(math.acos(d/(Rij*Rik)))/math.pi*180
                            the.append(theta)
                            g4.append(((1+lam*(d/(Rij*Rik)))**zeta)*(math.exp(-elta*(Rij**2+Rik**2+Rjk**2)))*fc1*fc2*fc3)
                #print(g4)
            g4_value=(2**(1-zeta))*sum(g4)
            g44.append(g4_value)
    g44=np.reshape(g44,(-1,19))
    return g44
#print(G4(4.0,1,1,1))

G=[]
for i in elta:
    for j in Rs:
        G.append(G2(5.0,i,j))
for i in elta:
    for j in zeta:
        G.append(G4(5.0,i,1,j))
#print(len(G))
GG=[]
for k in range(19):
    g=[]
    for i in range(len(G)):
        for j in range(len(G[i])):
            g.append(G[i][j][k])
    g=np.reshape(g,(len(G),-1))
    g_mat=np.matrix(g)
    g_mat = np.transpose(g_mat)
    g_mat = g_mat.tolist()
    GG.append(g_mat)
#print((GG))
data_x=[]
for j in range(len(GG[0])):
    for i in range(len(GG)):
        data_x.append(GG[i][j])
#print(data_x)
data_g=np.reshape(data_x,(len(GG[0]),19,len(G)))
#print(data_x)

####train
def add_layer(inputs, in_size,out_size,activation_function=None):
    Weights=tf.Variable(tf.random_normal([in_size,out_size]))
    biases=tf.Variable(tf.zeros([1,out_size]) + 0.1)
    #print(Weights)
    Ws_plus_b=tf.matmul(inputs,Weights) +  biases
    if activation_function is None:
        outputs = Ws_plus_b
    else:
        outputs = activation_function(Ws_plus_b)
    return outputs

for i in range(len(data_g)):
    y_data=E_reference[i]
    for j in range(len(data_g[i])):
        pre=[]
        #print(data_g[i][j])
        x_data=data_g[i][j]
        xs=tf.placeholder(tf.float32, [None,1])
        #ys=tf.placeholder(tf.float32, [None,1])
        l1=add_layer(xs,1,25,activation_function=tf.nn.relu)
        l2=add_layer(l1,25,25,activation_function=tf.nn.relu)
        prediction=add_layer(l2,25,1,activation_function=None)
        pre.append(prediction)
    ys=tf.placeholder(tf.float32, [None,1])
    E_pre=sum(pre)
    print(E_pre)
    loss=tf.reduce_mean(tf.reduce_sum(tf.square(ys-E_pre),reduction_indices=[1]))

    train_step=tf.train.GradientDescentOptimizer(0.1).minimize(loss)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        #sess.run(tf.initialize_all_variables())
        for i in range(1000):
            sess.run(train_step,feed_dict={xs:x_data,ys:y_data})
            if i% 50== 0:
                print(sess.run(loss,feed_dict={xs:x_data, ys:y_data}))
#print(len(G[0]))
#print(row['atomic_coordinates'])
#print(row['total_energy'])
#print(row.data['atomic_forces'])
