#Code by GVV Sharma, 
#Jan 13, 2020
#Revised
#March 9, 2023
#Released under GNU GPL
#Solving a quadratic program using cvx
from cvxpy import *
import numpy as np

#Line parameters
n =  np.array([1,1]).reshape(2,-1)
c = 9
P = np.array([8,6]).reshape(2,-1)

x = Variable((2,1))
#Cost function
f =  quad_form(x-P, np.eye(2))
obj = Minimize(f)
#Constraints
constraints = [n.T@x == c]

#solution
Problem(obj, constraints).solve()

print(np.sqrt(f.value),x.value)


