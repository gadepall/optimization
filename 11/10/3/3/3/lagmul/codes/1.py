import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.insert(0,'/home/jaswanth/6th-Sem/EE2802/CoordGeo')
from line.funcs import *
from conics.funcs import *

P = np.array([0,0])
n = np.array([1,-1])
c = 4

lamda = -2* (n.T@P - c)/np.linalg.norm(n)**2
Q = P + (lamda/2)*n

print(lamda, Q)