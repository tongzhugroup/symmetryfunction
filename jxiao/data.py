from ase.db import connect
import numpy as np
import operator
from collections import Counter
#distance matrix
def dismatrix(xyz):
    atomnum=len(xyz)
    DM=np.empty((atomnum,atomnum))
    for A in range(atomnum):
        for B in range(atomnum):
            if A!=B:
                DM[A,B]=np.linalg.norm(xyz[A]-xyz[B])
            else:
                DM[A,B]=np.infty
    return DM 
#T=np.array([[0,0,0],[0,0,1.5],[1,0,0],[0,0.7,1]])
#print(dismatrix(T))

def radiusG(path):
    with connect(path) as conn:
        totradius=[]
        for row in conn.select(limit=500):
            A=[]
            for atom in row.toatoms():
                B=[]
                B.append(atom.symbol)
                B.append(atom.position)
                A.append(B)
            A.sort(key=operator.itemgetter(0))
            atomlabel=[j[0] for j in A]
            Cindex=[i for i,z in enumerate(atomlabel) if z=="C"]
            Hindex=[i for i,z in enumerate(atomlabel) if z=="H"]
            Oindex=[i for i,z in enumerate(atomlabel) if z=="O"]
            Col=np.array([i[1] for i in A])
            diss=np.tril(dismatrix(Col),-1)
            C=diss[Cindex[0]:Cindex[-1]+1].flatten()
            H=diss[Hindex[0]:Hindex[-1]+1].flatten()
            O=diss[Oindex[0]:Oindex[-1]+1].flatten()
            Cradius=[i for i in C if i !=0]
            Hradius=[i for i in H if i !=0]
            Oradius=[i for i in O if i !=0]
            totradius.append([Cradius,Hradius,Oradius])
        return totradius
#print(radiusG("./reference_eq.db"))
        #print(dismatrix(C))
        #print(row['total_energy'])
        #prin t(row.data['atomic_forces'])
 
