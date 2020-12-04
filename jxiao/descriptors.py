#radiaal symmetry function
#import keras
#from keras.callbacks.callbacks import History
from matplotlib.pyplot import axes, axis
import numpy as np
#from keras.layers import Dense, Activation
#from keras.models import Sequential
def FC(Rij,Rc):
    if Rij<=Rc:
        return 0.5*(np.cos(np.pi*(Rij/Rc))+1)
    else:
        return 0
def G2(eta,Rij,Rs,Rc):
    return np.exp(-eta*np.square(Rij-Rs))*FC(Rij,Rc) #radiaal symmetry function

def G4(theta,Rij,Rc):
    return np.power((1+lbd*np.cos((theta/180)*np.pi)),zeta)*(1/np.power(2,zeta-1))*Fc(Rij,Rc) #radial symmetry fcuntion

def Example_E(Rij):
    return (np.cos(5*Rij)+np.square(Rij-4.5))/5-1
    


 
