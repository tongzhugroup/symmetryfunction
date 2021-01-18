import keras
import numpy as np
#from keras.models import Sequantial
from keras.layers import Dense,Activation,Input
from keras.models import Model 
from keras.layers.normalization import BatchNormalization
#from read_crd import *

Ri=RN("xyz.npy")[1]
Li=RN("xyz.npy")[0].reshape(18,1)
X_train=
En=np.load("energy.npy")
#Input=Input(shape=RN[1].shape,name="input")
#----block1---#
input_x=Input(shape=Li.shape,name="input")
M=Dense(40,activation="sigmoid",name="block1_L1")(input_x)
M=BatchNormalization()(M)
M=Dense(60,activation="sigmoid",name="block1_L2")(M)
M=BatchNormalization()(M)
G1=Dense(80,activation="sigmoid",name="block1_L3")(M)

G2=G1[:,0:40]

Dsr=G1*np.transform(Ri)*Ri*np.transform(G2)#descriptor

#---block2--#
M2=Dense(100,activation="sigmoid",name="block2_L1")(Dsr)
M2=BatchNormalization()(M2)
M2=Dense(60,activation="sigmoid",name="block2_L1")(M2)
M2=BatchNormalization()(M2)
Energy=Dense(1,activation="sigmoid",name="block2_L1")(M2)

picmodel=Model(inputs=input_x,outputs=Energy,name="two block NN")
picmodel.compile(loss='mean_squared_error',optimizer=keras.optimizers.Adam(lr=0.001,beta_1=0.9,beta_2=0.999,epsilon=1e-08,decay=0.0), metrics=['mae'])

history=picmodel.fit(X_train,En,epochs=1000,batch_size=80,verbose=2,shuffle=True)
print(history)
