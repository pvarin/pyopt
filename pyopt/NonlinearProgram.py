import numpy as np


class Variable:
    next_id = 0

    def __init__(self, name, initial_guess=0.0, lb=-np.inf, ub=np.inf):
        self.id = Variable.next_id
        Variable.next_id += 1
        self.name = name
        self.initial_guess = initial_guess
        self.lb = lb
        self.ub = ub

    def __repr__(self):
        return f"Variable_{self.name}"


def make_variables(name, shape, initial_guess=None, lb=None, ub=None):
    num_vars = np.prod(shape)

    variables = np.empty(shape, dtype=Variable)
    for i in range(num_vars):
        multi_index = np.unravel_index(i, shape)
        name_i = name + '_' + '_'.join([str(s) for s in multi_index])
        variables[multi_index] = Variable(
            name_i,
            initial_guess=0
            if initial_guess is None else initial_guess[multi_index],
            lb=-np.inf if lb is None else lb[multi_index],
            ub=np.inf if ub is None else ub[multi_index])

    return variables


class NonlinearProgram:

    def __init__(self):
        self.variables = []
        self.costs = []
        self.constraints = []

    def new_variables(self, *args, **kwargs):
        self.variables.append(make_variables(*args, **kwargs))
        return self.variables[-1]

    def add_cost(self, cost):
        self.costs.append(cost)

    def add_constraint(self, constraint):
        self.constraints.append(constraint)

    def num_vars(self):
        n_vars = 0
        for variable in self.variables:
            if isinstance(variable, Variable):
                n_vars += 1
            else:
                n_vars += variable.size
        return n_vars

    def get_flattened_variable_names(self):
        names = []
        for variable in self.variables:
            if isinstance(variable, Variable):
                names.append(variable.name)
            else:
                for v in variable.flatten():
                    names.append(v.name)
        return names

    def get_flattened_variable_ids(self):
        ids = []
        for variable in self.variables:
            if isinstance(variable, Variable):
                ids.append(variable.id)
            else:
                for v in variable.flatten():
                    ids.append(v.id)
        return ids

    def get_variable_indices(self, var):
        var_ids = self.get_flattened_variable_ids()
        return [var_ids.index(v.id) for v in var]

    def get_flattened_initial_guess(self):
        x = []
        for variable in self.variables:
            if isinstance(variable, Variable):
                x.append(variable.initial_guess)
            else:
                for v in variable.flatten():
                    x.append(v.initial_guess)
        return np.array(x)

    def get_flattened_variable_bounds(self):
        lb = []
        ub = []
        for variable in self.variables:
            if isinstance(variable, Variable):
                lb.append(variable.lb)
                ub.append(variable.ub)
            else:
                for v in variable.flatten():
                    lb.append(v.lb)
                    ub.append(v.ub)
        return np.array(lb), np.array(ub)

    def num_constraints(self):
        n_con = 0
        for con in self.constraints:
            n_con += con.num_constraints()
        return n_con

    def get_constraint_bounds(self):
        lb = np.concatenate([c.lb for c in self.constraints])
        ub = np.concatenate([c.ub for c in self.constraints])
        return lb, ub

    def get_constraint_names(self):
        names = []
        for con in self.constraints:
            num_con = con.num_constraints()
            if num_con == 1:
                names.append(con.name)
            else:
                names.extend([f"{con}_{i}" for i in range(num_con)])
        return names

    def get_solution_value(self, var, solution):
        return solution[self.get_variable_indices(var)]


class Cost:

    def __init__(self, name, variables, coeff=1.0):
        self.name = name
        self.variables = variables
        self.coeff = coeff

    def num_vars(self):
        return self.variables.size

    def eval(self):
        pass

    def evalGrad(self):
        pass

    def __repr__(self):
        return f"Cost_{self.name}"


class Constraint:

    def __init__(self, name, variables, lb, ub):
        self.name = name
        self.variables = variables
        self.lb = lb
        self.ub = ub

    def num_vars(self):
        return self.variables.size

    def num_constraints(self):
        return self.lb.size

    def eval(self, *args):
        pass

    def evalJac(self, *args):
        pass

    def __repr__(self):
        return f"Constraint_{self.name}"


if __name__ == '__main__':
    print(make_variables("test", (3, 3)))
