import keras
from keras import activations
from keras.callbacks.callbacks import History
import matplotlib.pyplot as plt
from keras.layers import Dense, Activation
from keras.models import Sequential
import numpy as np
from descriptors import FC
from descriptors import G2
from data import radiusG

L=radiusG("./reference.db")

#print(L)
#print(len(L))
#prepare input data
etaL1=[0.001,0.01,0.03,0.06] #G2 for rs=0.0,Rc=12
etaL2=[0.15,0.3,0.6,1.5]#G2 for rs=0.9

En=[]
for J in L:
    En.append(J[3])
Ee=np.array(En)/27.21

#for carbon
sumG2C=[]
for eta in etaL1:
    tet=[]
    for J in L:
        Y=sum([G2(eta,i,0.0,12.0) for i in J[0]])
        tet.append(Y)
    sumG2C.append(tet)
sumG2C=np.transpose(np.array(sumG2C))
#print(len(sumG2C[0]))
#for hydrogen
sumG2H=[]
for eta in etaL1:
    tet=[]
    for J in L:
        Y=sum([G2(eta,i,0.0,12.0) for i in J[1]])
        tet.append(Y)
    sumG2H.append(tet)
sumG2H=np.transpose(np.array(sumG2H))
#for oygen
sumG2O=[]
for eta in etaL1:
    tet=[]
    for J in L:
        Y=sum([G2(eta,i,0.0,12.0) for i in J[2]])
        tet.append(Y)
    sumG2O.append(tet)
sumG2O=np.transpose(np.array(sumG2O))

#carbon
CM=keras.layers.Input(shape=(4,))
CM1=Dense(25,activation="sigmoid")(CM)
CM2=Dense(25,activation="sigmoid")(CM1)
CM3=Dense(1,input_shape=(25,))(CM2)

HM=keras.layers.Input(shape=(4,))
HM1=Dense(25,activation="sigmoid")(HM)
HM2=Dense(25,activation="sigmoid")(HM1)
HM3=Dense(1,input_shape=(25,))(HM2)

OM=keras.layers.Input(shape=(4,))
OM1=Dense(25,activation="sigmoid")(OM)
OM2=Dense(25,activation="sigmoid")(OM1)
OM3=Dense(1,input_shape=(25,))(OM2)

added=keras.layers.add([CM3,HM3,OM3])
out= keras.layers.Dense(1)(added)
model=keras.models.Model(inputs=[CM,HM,OM],outputs=out)

#print(model.summary())
model.compile(loss='mean_squared_error',optimizer=keras.optimizers.Adam(lr=0.001,beta_1=0.9,beta_2=0.999,epsilon=1e-08,decay=0.0), metrics=['mae'])

history=model.fit([sumG2C,sumG2H,sumG2O],Ee, epochs=1000,batch_size=100,verbose=2,shuffle=True)
#print(sumG2C.shape,sumG2H.shape,sumG2O.shape)


A=model.predict([sumG2C,sumG2H,sumG2O])

#for i in range(len(Ee)):
#    print(A[i],Ee[i])

plt.figure(figsize=(8,6))
plt.plot(range(0,len(history.history["loss"])),history.history["loss"])
plt.legend(["loss"])
plt.text(800,10000,history.history["loss"][-1])
plt.savefig("training.png")
prerr=A-Ee
MAE=np.mean(np.absolute(A-Ee))
plt.figure(figsize=(8,6))
plt.scatter(A,prerr,marker='o',s=2,c=prerr,cmap='hsv')
plt.savefig("MAE.png")

#print(len(A))
#for i in range(len(Ee)):
#   print(A*27.21)
"""
De=np.array(sumG2)
Ae=np.transpose(De)
Ee=np.array([Example_E(i) for i in np.linspace(1,8,71)])
model = Sequential([
    Dense(25, input_shape=(3,),use_bias=True,activation="sigmoid"),
    Dense(25, input_shape=(25,),use_bias=True,activation="sigmoid"),
    Dense(1)
])
model.compile(loss='mean_squared_error',optimizer=keras.optimizers.Adam(lr=0.001,beta_1=0.9,beta_2=0.999,epsilon=1e-08,decay=0.0), metrics=['mae'])
history=model.fit(Ae,Ee, epochs=50000,
batch_size=100,verbose=2,shuffle=True)
"""
