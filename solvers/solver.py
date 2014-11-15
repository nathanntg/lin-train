import abc


class Solver:
    def __init__(self):
        pass

    @abc.abstractmethod
    def calculate_parameters(self, x, y):
        """
        Takes a matrix and a column vector. Returns a column vector of coefficients that best correspond to y = f(x, b).
        """
        return [0.]

    @abc.abstractmethod
    def apply_parameters(self, x, params):
        return 0.
