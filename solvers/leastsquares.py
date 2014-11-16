from solver import Solver
import numpy as np


class LeastSquares(Solver):
    """
    One of the most common tools used for regression, as it has a nice analytic solution that minimizes least square error.
    This is ideal suited to predicting y where y is a function of the predictors plus a normal noise term.
    Y | X ~ Normal(x, v)
    """

    def __init__(self):
        Solver.__init__(self)

    def calculate_parameters(self, x, y):
        # use standard least squares algorithm from numpy
        fit = np.linalg.lstsq(x, y)[0]
        return fit

    def apply_parameters(self, x, params):
        return np.dot(x, params)
