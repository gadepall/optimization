import numpy as np
import matplotlib.pyplot as plt
import math

#Generate line points
def line_gen_vector(n, c, x):
    y = (c - n[0]*x)/n[1]
    return y

def line_gen(A,B):
  len =10
  dim = A.shape[0]
  x_AB = np.zeros((dim,len))
  lam_1 = np.linspace(0,1,len)
  for i in range(len):
    temp1 = A + lam_1[i]*(B-A)
    x_AB[:,i]= temp1.T
  return x_AB

def lamda(c, n, O):
    return (c - (n@O))/(n@n)

def func_derivative(x):
    return 260*x - 260/7  # f'(x)

def f(x) :
    return (130*x**2 -(260/7)*x + math.sqrt(36010)/63) #Objective function

#Given points
O = np.array([22/9, 3])
A = np.array([19/7, 0])

#Given line
n = np.array([7, -9])
c = 19


#finding the point P
P = O + lamda(c, n, O)*n

x_OP = line_gen(O, P)
plt.plot(x_OP[0,:],x_OP[1,:],label='$Perpendicular$')

#Plot the given line
x = np.linspace(-7, 12, 100)
y = line_gen_vector(n, c, x)
plt.plot(x, y, label='$(7  -9)x = 19$')

#Plot the points
plt.plot(O[0], O[1], 'o')
plt.text(O[0] * (1 + 0.1), O[1] * (1 - 0.1) , 'O')
plt.plot(P[0], P[1], 'o')
plt.text(P[0] * (1 + 0.1), P[1] * (1 - 0.1) , 'P')

#legend
plt.xlabel('$x$')
plt.ylabel('$y$')
plt.legend(loc='best')
plt.grid() # minor
plt.axis('equal')
plt.savefig('/home/lokesh/EE2802/EE2802-Machine_learning/11.10.3.10/lagmul/figs/lines.jpg')