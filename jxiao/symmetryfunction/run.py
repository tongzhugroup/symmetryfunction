import numpy as np
import itertools
from math import pi
from function_name import *
from typing import List, Tuple, Set,Dict





def symm_f2_mol(mol,atom_types, eta_list, R_s_list, R_c:float=None):
    symm_f_params = get_symm2_params(eta_list, R_s_list)

    pairwise_permutations = np.array(list(itertools.permutations(mol,2))) # get coord pair
    
    atom_list = [{} for i in range(len(mol))]
    for  i in range(len(atom_list)):
        for j in range(len(pairwise_permutations)):
            if (pairwise_permutations[j][0] == mol[i]).all(): #'center atom vs. atom in xyz'
                for p1,p2 in symm_f_params:
                    atom_dict = atom_list[i]
                    f_c = cutoff(pairwise_permutations[j][0][1:],pairwise_permutations[j][1][1:])
                    sf = symm_f2(pairwise_permutations[j][0][1:],
                                pairwise_permutations[j][1][1:],
                                eta=p1,R_s=p2) * f_c
                    key = "-".join(pairwise_permutations[j][:,0])+": eta="+str(p1)+" R_s="+str(p2) #'long key name to check where values come from'
                    atom_dict[key] = atom_dict.setdefault(key,0) + sf

            else:
                continue
    return atom_list
mol = [['O', '-0.005703', '0.385159', '-0.000000'],
        ['H', '-0.796078', '-0.194675', '-0.000000'], 
        ['H', '0.801781', '-0.190484', '0.000000']]

def symm_f4_mol(mol, atom_types, eta_list, lamdb_list, zeta_list, R_s_list):
    threewise_permutations = np.array(list(itertools.permutations(mol,3)))
    symm_f_params = get_symm4_params(eta_list, lamdb_list, zeta_list, R_s_list)
    atom_list = [{} for i in range(len(mol))]
    for i in range(len(atom_list)):
        for j in range(len(threewise_permutations)):
            if (threewise_permutations[j][0] == mol[i]).all():
                for p1,p2,p3,p4 in symm_f_params:
                    atom_dict = atom_list[i]
                    f_c = cutoff(threewise_permutations[j][0][1:],threewise_permutations[j][1][1:])
                    #print(threewise_permutations[j][0][1:],threewise_permutations[j][1][1:],threewise_permutations[j][2][1:])
                    sf = symm_f4(threewise_permutations[j][0][1:],
                                threewise_permutations[j][1][1:],
                                threewise_permutations[j][2][1:],
                                eta=p1, lamdb=p2, zeta=p3) * f_c

                    key = "-".join(threewise_permutations[j][:,0])+": eta="+str(p1)+" lamda="+str(p2)+" zeta="+str(p3)+" R_s="+str(p4)
                    atom_dict[key] = atom_dict.setdefault(key,0) + sf
            else:
                continue
    return atom_list
            
R_c = 10
eta=[1,2,3]
lamda=[-1,1]
zeta=[4,5,6]
R_s=[7,8]
at_types=['C','H','N','O']

mol = [['O', '-0.005703', '0.385159', '-0.000000'],
        ['H', '-0.796078', '-0.194675', '-0.000000'], 
        ['H', '0.801781', '-0.190484', '0.000000']]

for i in symm_f2_mol(mol,at_types,eta,R_s,R_c):
    for j in i:
        print(j, i[j])

for i in symm_f4_mol(mol,at_types,eta,lamda,zeta,R_s):
    for j in i:
        print(j,i[j])

#print(*symm_f2_mol(mol, at_types, eta, R_s), sep ="\n")
#print(*symm_f4_mol(mol,at_types, eta, lamda, zeta, R_s),sep ="\n")

