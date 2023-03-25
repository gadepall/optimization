import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.insert(0,'/home/jaswanth/6th-Sem/EE2802/CoordGeo')
from line.funcs import *
from conics.funcs import *

def V(x):
    return x*(18-2*x)**2

a = 18
x = a/6
print  (x, V(x))