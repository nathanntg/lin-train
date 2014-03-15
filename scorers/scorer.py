import abc


class Scorer:
    def __init__(self):
        pass

    @abc.abstractmethod
    def score(self, predictions):
        """
        Takes a two column matrix of potential and actual values and returns a number score representing the accuracy
        of the predictions.
        """
        return 0.

    @staticmethod
    def sort():
        """
        If a larger score indicates a better fit, then this should return 1.
        If a smaller score indicates a better fit, then this should return -1.
        """
        return -1
