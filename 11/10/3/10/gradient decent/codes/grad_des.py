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

lamda_n = -5 # Start value at lamda_n= -5
alpha = 0.001 # step size multiplier
precision = 0.0000001
previous_step_size = 1 
max_count = 10000000 # maximum number of iterations
count = 0 # counter

def func_derivative(x):
    return 260*x - 260/7  # f'(x)

while (previous_step_size > precision) & (count < max_count):
    lamda_n_minus1 = lamda_n
    lamda_n -= alpha * func_derivative(lamda_n_minus1)
    previous_step_size = abs(lamda_n - lamda_n_minus1)
    count+=1

print("The minimum value of function is at", lamda_n)

def f(x) :
    return (130*x**2 -(260/7)*x + math.sqrt(36010)/63) #Objective function

x = np.linspace(-1,1,100)#points on the x axis
y=  f(x) #Objective function
plt.figure(1)
plt.plot(x,y, label = '$f(x)$')
plt.plot([lamda_n],[f(lamda_n)],marker='o',label='$\lambda_{Min}$')
plt.grid()
plt.xlabel('$x-Axis$')
plt.ylabel('$y-Axis$')
plt.title('Minimum Value of Function')
plt.legend(loc = 'best')
plt.savefig('/home/lokesh/EE2802/EE2802-Machine_learning/11.10.3.10/gradient decent/figs/plot.jpg')

plt.close()

#Given points
O = np.array([22/9, 3])
A = np.array([19/7, 0])

P = np.array([4, 1])

#Given line
n = np.array([7, -9])
c = 19

x = np.linspace(-7, 12, 100)

#Plot line formed by A and B
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
plt.savefig('/home/lokesh/EE2802/EE2802-Machine_learning/11.10.3.10/gradient decent/figs/lines.jpg')