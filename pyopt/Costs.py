import numpy as np
from pyopt.NonlinearProgram import Cost


def wrap_angle(theta):
    return np.mod(theta + np.pi, 2 * np.pi) - np.pi


class QuadraticCost(Cost):
    def __init__(self, name, variables, Q):
        super().__init__(name, variables)
        self.Q = Q

    def eval(self, x):
        return 0.5 * x.dot(self.Q).dot(x)

    def evalGrad(self, x):
        return self.Q.dot(x)


class SquaredNormCost(Cost):
    def __init__(self, name, variables, coeff):
        super().__init__(name, variables)
        self.coeff = coeff

    def eval(self, x):
        return 0.5 * self.coeff * x.dot(x)

    def evalGrad(self, x):
        return self.coeff * x


class TipPoseErrorCost(Cost):
    def __init__(self, name, variables, coeff, model, desired_pose):
        super().__init__(name, variables)
        self.coeff = coeff
        self.model = model
        self.desired_pose = desired_pose

    def eval(self, x):
        tip_pose = self.model.tip_pose(q=x)
        pos_err = tip_pose[:2] - self.desired_pose[:2]
        orientation_err = wrap_angle(tip_pose[2] - self.desired_pose[2])
        return self.coeff * (pos_err.dot(pos_err) + np.square(orientation_err))

    def evalGrad(self, x):
        # TODO: eval with numerical gradient or compute analytical gradient with jacobian
        raise NotImplementedError
