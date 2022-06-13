from pyopt.NonlinearProgram import Constraint


class LinearConstraint(Constraint):
    def __init__(self, name, variables, lb, ub, A):
        super().__init__(name, variables, lb, ub)
        self.A = A

    def eval(self, x):
        return self.A.dot(x)

    def evalJac(self, x):
        return self.A
