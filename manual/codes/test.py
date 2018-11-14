#Code by GVV Sharma, Novermber 12, 2018
#Released under GNU GPL
#Semi definite programming example
from cvxpy import *
from numpy import matrix

x = Variable((2,2),PSD = True)
u = matrix([[1],[0]])
v = matrix([[1],[1]])
#f = x[0][0] + x[0][1]
f =  u.transpose()*x*v
obj = Minimize(f)
#constraints = [x[0][0] + x[1][1] == 1]
constraints = [x[0][0] + x[1][1] == 1]

Problem(obj, constraints).solve()
print ("Minimum of f(x) is ",round(f.value,2), " at  \
(",round(x[0][0].value,2),",",round(x[0][1].value,2),")") 

