import numpy as np
import matplotlib.pyplot as plt
import cvxpy as cp

A = np.array([[1,2],
              [3,2],
              [-1,0],
              [0,-1]])

B = np.array([8,12,0,0])

# Define variables
x = cp.Variable(2)

# Define constraints
constraints = [A@x <= B]

n = np.array([-3,4])

# Define objective
objective = cp.Minimize(n.T@x)

# Solve problem
prob = cp.Problem(objective, constraints)
result = prob.solve()

# Print solution
print(round(result,4))
print("x =", np.round(x.value,4))

#Drawing lines
x1 = np.linspace(-1,7,400) 
t = np.zeros(len(x1))
y1 = (8-x1)/2
y2 = (12-3*x1)/2
y3 = np.linspace(-1,7,400)

plt.plot(x1,y1,label = '$x+2y=8 $')
plt.plot(x1,y2,label = '$3x+2y=12$')
plt.plot(x1,t,label = '$x=0$')
plt.plot(t,y3,label = '$y=0$')

plt.grid()
plt.xlabel('$x$')
plt.ylabel('$y$')
plt.title('Corner point method')
plt.ylim(-0.5,8)

#Corner Points
P = np.array([0, 0])
Q = np.array([0, 4])
R = np.array([2, 3])
S = np.array([4, 0])

#Filling the feasible region
fill_cords = np.vstack((P,Q,R,S))
plt.fill(fill_cords[:,0], fill_cords[:,1],'plum',label =  "Feasible Region")


#Labeling the coordinates
tri_coords = np.vstack((P,Q,R,S)).T
plt.scatter(tri_coords[0,:], tri_coords[1,:])
vert_labels = ['P','Q','R','S']
for i, txt in enumerate(vert_labels):
    label = "{}({},{})".format(txt, int(tri_coords[0,i]),int(tri_coords[1,i])) #Form label as A(x,y)
    plt.annotate(label, # this is the text
                 (tri_coords[0,i], tri_coords[1,i]), # this is the point to label
                 textcoords="offset points", # how to position the text
                 xytext=(20,5), # distance from text to points (x,y)
                 ha='center') # horizontal alignment can be left, right or center

plt.legend(loc='best')
plt.grid('on')
plt.savefig('../figs/fig.png')
plt.show()
