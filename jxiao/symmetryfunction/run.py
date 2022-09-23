import numpy as np
import itertools
from math import pi
from function_name import *
from typing import List, Tuple, Set,Dict


R_c = 10
eta=[1,2,3]
R_s=[7,8]
at_types=['C','H','N','O']

mol = [['O', '-0.005703', '0.385159', '-0.000000'], 
        ['H', '-0.796078', '-0.194675', '-0.000000'], 
        ['H', '0.801781', '-0.190484', '0.000000']]

def symm_f_mol(mol,atom_types, eta_list, R_s_list,R_c:float=None):
    symm_f_params = get_symm_params(eta_list, R_s_list)

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
                    print(sf)
                    key = "-".join(pairwise_permutations[j][:,0])+": eta="+str(p1)+" R_s="+str(p2) #'long key name to check where values come from'
                    atom_dict[key] = atom_dict.setdefault(key,0) + sf

            else:
                continue
    return atom_list

print(*symm_f_mol(mol, at_types, eta, R_s), sep ="\n")

