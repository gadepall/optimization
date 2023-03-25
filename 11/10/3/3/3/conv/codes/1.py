import numpy as np
import cvxpy as cp

P = np.array([0,0])
n = np.array([1,-1])
c = 4

# Define variables
x = cp.Variable(2)

# Define constraints
constraints = [n.T@x == c]

# Define objective
objective = cp.Minimize(cp.norm(x-P))

# Solve problem
prob = cp.Problem(objective, constraints)
result = prob.solve()

# Print solution
print(result)
print("x =", x.value)
