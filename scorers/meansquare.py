from scorer import Scorer
import numpy as np


class MeanSquare(Scorer):
    def __init__(self):
        Scorer.__init__(self)

    def score(self, predictions):
        return np.sum(np.absolute(predictions[:, 1] - predictions[:, 0])) / np.shape(predictions)[0]

    @staticmethod
    def sort():
        """
        If score get larger as fit improves, then this should return 1.
        If score gets smaller as fit improves, then this should return -1.
        """
        return -1
