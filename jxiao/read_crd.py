import numpy as np

def smoothfunc(rij,rc,rcs):
    if rij <rcs:
        return 1/rij
    elif rcs<rij<rc:
        return (1/rij)*(1/2*(np.cos(np.pi*(rij-rcs)/(rc-rcs)))+1/2)
    else:
        return 0
def Distance(xyz):
    C=[]
    for i in range(len(xyz)-1):
        dis=list(np.abs(xyz[i+1]-xyz[0]))
        rij=np.sqrt(np.sum(np.square(dis)))
        sij=smoothfunc(rij,6.5,2)
        dis.insert(0,sij)
        C.append(dis)
    return C
def RN(npy):
    X=np.load(npy)
    F=[]
    for j in X:
        F.append(Distance(j))
    return F
print(RN("xyz.npy")[-1])
np.save("RI.npy",RN("xyz.npy"))


