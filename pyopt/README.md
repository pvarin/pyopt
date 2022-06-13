# Optimization Framework
This module adds a framework for nonlinear optimization and a wrapper for SNOPT.

## Example
Here is a short example that creates and solved an optimization problem. We start by constructing a `NonlinearProgram` object as well as two vector-values variables `x` and `y`:
```python
# Create a nonlinear program and a couple of variables.
prog = NonlinearProgram()
x = prog.new_variables('x', (2,))
y = prog.new_variables('y', (4,))
```

We can then add any costs and constraints that we want:
```python
# Add an arbitrary PSD cost.
L = np.random.random((4,4))
Q = L.transpose().dot(L)
cost = QuadraticCost('my_cost', [y], Q)
prog.add_cost(cost)

# Add an arbitrary linear inequality constraint.
A = np.random.random((2, 6))
lb = np.random.random(2)
ub = lb + 2
con = LinearConstraint('my_constraint', [np.concatenate([x,y])], lb, ub, A)
prog.add_constraint(con)
```

Finally we solve the problem and extract the values of the variables that we defined earlier.
```python
# Solve the optimization problem.
result = solve_snopt(prog)

# Extract the variable values.
x_val = prog.get_solution_value(x, result.x)
y_val = prog.get_solution_value(y, result.x)
```
