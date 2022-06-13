import pdb
import numpy as np
from pyopt.NonlinearProgram import NonlinearProgram, make_variables
from pyopt.Costs import QuadraticCost
from pyopt.Constraints import LinearConstraint
from pyopt.snopt_wrapper import solve_snopt

# Create a nonlinear program and a couple of variables.
prog = NonlinearProgram()
x = prog.new_variables('x', (2, ))
y = prog.new_variables('y', (4, ))

# Add an arbitrary PSD cost.
L = np.random.random((4, 4))
Q = L.transpose().dot(L) + 1e-2 * np.eye(4)
cost = QuadraticCost('my_cost', [y], Q)
prog.add_cost(cost)

# Add an arbitrary linear inequality constraint.
A = np.random.random((2, 6))
lb = np.random.random(2)
ub = lb + 2
con = LinearConstraint('my_constraint', [np.concatenate([x, y])], lb, ub, A)
prog.add_constraint(con)

# Solve the optimization problem.
result = solve_snopt(prog)

# Extract the variable values.
x_val = prog.get_solution_value(x, result.x)
y_val = prog.get_solution_value(y, result.x)

# Verify that the cost matches the result objective.
cost_val = cost.eval(y_val)
np.testing.assert_almost_equal(result.objective, cost_val)

# Verify that the constraints are satisfied.
eps = 1e-6  # epsilon for solver tolerance
con_val = con.eval(np.concatenate([x_val, y_val]))
assert (np.all(con_val <= ub + eps))
assert (np.all(con_val >= lb - eps))
