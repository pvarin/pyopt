import numpy as np
from optimize import snopta, SNOPT_options


def get_snopt_variable_name(name):
    return " " * 6 + name


def get_snopt_objective_name(name):
    return " " * 6 + name


def get_snopt_constraint_name(name):
    return " " * 6 + name


def solve_snopt(prog, options=None, run_name=None):
    if options is None:
        options = SNOPT_options()
    if run_name is None:
        run_name = 'snopt_problem'

    # Variable information.
    n_vars = prog.num_vars()
    xids = prog.get_flattened_variable_ids()
    initial_guess = prog.get_flattened_initial_guess()
    xlow, xupp = prog.get_flattened_variable_bounds()
    var_names = np.array([
        get_snopt_variable_name(name)
        for name in prog.get_flattened_variable_names()
    ],
                         dtype="|S8")

    # Constraint information.
    n_con = prog.num_constraints()
    lb, ub = prog.get_constraint_bounds()
    constraint_names = np.array([
        get_snopt_constraint_name(name)
        for name in prog.get_constraint_names()
    ],
                                dtype="|S8")

    ObjRow = 1

    cost_indices = [[prog.get_variable_indices(v) for v in cost.variables]
                    for cost in prog.costs]
    constraint_indices = [[
        prog.get_variable_indices(v) for v in con.variables
    ] for con in prog.constraints]

    def snopt_callback(status, x, needF, F, needG, G):
        # Accumulate the costs and
        cost_val = 0
        for cost, idxs in zip(prog.costs, cost_indices):
            vars = [x[idx] for idx in idxs]
            cost_val += cost.eval(*vars)

        constraint_start_idx = 1
        for i, constraint in enumerate(prog.constraints):
            vars = [x[idx] for idx in constraint_indices[i]]
            num_con = constraint.num_constraints()
            F[constraint_start_idx:constraint_start_idx +
              num_con] = constraint.eval(*vars)
            constraint_start_idx += num_con

        return status, F

    # Build the data how SNOPT wants it. The objective data is in the first row.
    Fnames = np.concatenate([
        np.array([get_snopt_objective_name("objective")], dtype="|S8"),
        constraint_names
    ])
    Flow = np.concatenate([[-np.inf], lb])
    Fupp = np.concatenate([[np.inf], ub])

    # Solve the snopt problem.
    return snopta(snopt_callback,
                  n_vars,
                  1 + n_con,
                  x0=initial_guess,
                  xlow=xlow,
                  xupp=xupp,
                  Flow=Flow,
                  Fupp=Fupp,
                  ObjRow=ObjRow,
                  xnames=var_names,
                  Fnames=Fnames,
                  name=run_name,
                  options=options)
