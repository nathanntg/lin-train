class Scorer:
    def __init__(self):
        pass

    def score(self, predictions):
        return 0.

    @staticmethod
    def sort():
        """
        If score get larger as fit improves, then this should return 1.
        If score gets smaller as fit improves, then this should return -1.
        """
        return -1
