#import keras
#from keras.callbacks.callbacks import History
#from matplotlib.pyplot import axes, axis
#from keras.layers import Dense, Activation
#from keras.models import Sequential
import numpy as np
from descriptors import FC
from descriptors import G2
from data import radiusG

L=radiusG("./iso17/reference.db")
#print(len(L))
#prepare input data
etaL1=[0.001,0.01,0.03,0.06] #G2 for rs=0.0,Rc=12
etaL2=[0.15,0.3,0.6,1.5]#G2 for rs=0.9


#for carbon
sumG2C=[]
for eta in etaL1:
    tet=[]
    for J in L:
        Y=sum([G2(eta,i,0.0,12.0) for i in J[0]])
        tet.append(Y)
    sumG2C.append(tet)
print(len(sumG2[0]))
#for hydrogen
sumG2H=[]
for eta in etaL1:
    tet=[]
    for J in L:
        Y=sum([G2(eta,i,0.0,12.0) for i in J[0]])
        tet.append(Y)
    sumG2H.append(tet)
#for oygen
sumG2O=[]
for eta in etaL1:
    tet=[]
    for J in L:
        Y=sum([G2(eta,i,0.0,12.0) for i in J[0]])
        tet.append(Y)
    sumGO.append(tet)

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
