from solver import Solver
import numpy as np


class RidgeRegression(Solver):
    """
    Analytically performs ridge regression, where coefficients are regularized by learning rate alpha. This constrains
    coefficients and can be effective in situations where over- or under-fitting arise. Parameters `alpha`
    is the regularization constant (alpha of 0 is least squares, as alpha increases, coefficients are increasingly
    constrained).  Parameter `intercept` (defaults to true) causes an intercept column (all ones) to automatically be
    detected and excluded from regularization.

    Based off of:
    https://gist.github.com/diogojc/1519756
    """

    def __init__(self, alpha=0.1, intercept=True):
        Solver.__init__(self)

        # parameters
        self.alpha = alpha # regularization constant
        self.intercept = intercept # automatically gues intercept and do not regularize

    def calculate_parameters(self, x, y):
        g = self.alpha * np.eye(x.shape[1])

        # cancel out intercept term (do not regularize intercept
        if self.intercept:
            idx = np.all(x == 1, axis=0)
            g[idx, idx] = 0

        fit = np.dot(np.linalg.inv(np.dot(x.T, x) + np.dot(g.T, g)),
                     np.dot(x.T, y))
        return fit

    def apply_parameters(self, x, params):
        return np.dot(x, params)
