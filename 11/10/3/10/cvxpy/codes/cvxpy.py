import numpy as np
import matplotlib.pyplot as plt
from numpy import linalg as LA
import math
import cvxpy as cp

def line_gen_vector(n, c, x):
    y = (c - n[0]*x)/n[1]
    return y

omat = np.array([[0,1],[-1,0]])

n =  np.array([7,-9]).reshape(2,-1)
c = 19

A = np.array([22/9,3]).reshape(2,-1)

x = cp.Variable((2,1))

#Cost function
f =  cp.quad_form(x-A, np.eye(2))
obj = cp.Minimize(f)

#Constraints
constraints = [n.T@x == c]

#solution
prob  = cp.Problem(obj, constraints)
solution = prob.solve()

print("Foot of perpendicular:", x.value)


#Plotting
B = np.array([x.value[0],x.value[1]]).reshape(2)
A = A.reshape(2)

x = np.linspace(-7, 12, 100)

plt.plot(x, line_gen_vector((B-A)@omat, ((B-A)@omat)@A, x), label='$(2  14/9)x = 86/9$')
plt.plot(x, line_gen_vector(n, c, x), label='$(7  -9)x = 19$')

#Plot the points
plt.plot(A[0], A[1], 'o')
plt.text(A[0] * (1 + 0.1), A[1] * (1 - 0.1) , 'A')
plt.plot(B[0], B[1], 'o')
plt.text(B[0] * (1 + 0.1), B[1] * (1 - 0.1) , 'B')

plt.xlabel('$x$')
plt.ylabel('$y$')
plt.legend(loc='best')
plt.grid()
plt.axis('equal')
plt.savefig('/home/lokesh/EE2802/EE2802-Machine_learning/11.10.3.10/cvxpy/figs/lines.png')
