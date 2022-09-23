import numpy as np
import math as pi
import itertools
from typing import List, Tuple, Dict


def symm_f2(atom_i: np.ndarray, atom_j: np.ndarray, eta: float = 1, R_s: float = 2):
    
    a=np.array([float(i) for i in atom_i])
    b=np.array([float(j) for j in atom_j])
    R_ij = np.linalg.norm(a-b)
    return np.exp(-eta * (R_ij-R_s)**2)

def cutoff(atom_i: np.ndarray, atom_j: np.ndarray, R_c: float=None) -> float:
    a=np.array([float(i) for i in atom_i])
    b=np.array([float(j) for j in atom_j])
    
    r_ij = np.linalg.norm(a-b)
    if not R_c:
        f_c=1
    elif r_ij <= R_c:
        f_c = 0.5 * (np.cos(pi * r_ij / R_c) + 1)
    elif r_ij > R_c:
        f_c = 0
    return f_c




#  get eta and R_s couple
def get_symm_params(eta_list: List = None , R_s_list: List=None) -> List[Tuple[float]]:
    if not eta_list:
        eta_list=[1]
    if not R_s_list:
        R_s_list=[1]
    return list(itertools.product(eta_list, R_s_list))
