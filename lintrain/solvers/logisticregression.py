from solver import Solver
import numpy as np


class LogisticRegression(Solver):
    """
    This uses the generalized linear model with gradient descent to calculate linear coefficients (parameters). This
    class is abstract and must be extended by the underlying distribution to determine the exact gradient and method
    for applying parameters.

    Inspired by Matlab code:
    http://www.cs.cmu.edu/~ggordon/IRLS-example/
    """
    epsilon = 1e-10
    max_iterations = 500
    ridge = 1e-5

    def __init__(self):
        Solver.__init__(self)

    def calculate_parameters(self, x, y):
        # dimensions
        n, m = np.shape(x)
        # use standard least squares algorithm from numpy
        i = 0
        params = np.zeros((m, 1))
        old_exp_y = - np.ones(np.shape(y))
        while i < self.max_iterations:
            # count iteration
            i += 1

            # calculate
            adj_y = np.dot(x, params)
            exp_y = 1 / (1 + np.exp(-adj_y))
            deriv = exp_y * (1 - exp_y)
            w_adj_y = (deriv * adj_y + (y - exp_y))  # * w
            weights = np.diag(deriv.flatten(1))  # * w
            try:
                params = np.dot(np.dot(np.linalg.inv(np.dot(np.dot(x.T, weights), x) + self.ridge), x.T), w_adj_y)
            except np.linalg.linalg.LinAlgError as err:
                print "Warning: Singular matrix"
                return params

            if np.sum(np.abs(exp_y - old_exp_y)) < n * self.epsilon:
                return params

            old_exp_y = exp_y

        # todo
        print "Warning: Does not converge"

        return params

    def apply_parameters(self, x, params):
        return 1 / (1 + np.exp(-np.dot(x, params)))
