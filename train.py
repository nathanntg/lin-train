import numpy as np


class Train:
    def __init__(self, x, y, solver, scorer, number_of_folds=5):
        self.x = x
        self.y = y
        self.solver = solver
        self.scorer = scorer
        self.number_of_folds = number_of_folds
        self.folds = []

    def _get_fold(self, fold):
        # get number rows
        l_indices = np.shape(self.x)[0]

        # get indices
        row_indices_for_validation = self.folds[fold]
        row_indices_for_training = [s for s in xrange(0, l_indices) if s not in row_indices_for_validation]

        return row_indices_for_training, row_indices_for_validation

    def _train(self, col_indices_for_inputs, row_indices):
        # get x and y
        x = self.x[np.ix_(row_indices, col_indices_for_inputs)]
        y = self.y[row_indices]

        # run linear regression
        fit = self.solver.calculate_parameters(x, y)

        return fit

    def _validate(self, col_indices_for_inputs, row_indices, fit):
        # get x and y
        x = self.x[np.ix_(row_indices, col_indices_for_inputs)]
        y = self.y[row_indices]

        # generate predictions
        predicted_y = self.solver.apply_parameters(x, fit)

        # create tuples
        validation = np.concatenate((y, predicted_y), 1)

        return self.scorer.score(validation)

    def _score(self, col_indices_for_inputs):
        score = 0.

        # for each fold in k-fold-cross-validation
        for fold in xrange(self.number_of_folds):
            # get indices
            (row_indices_for_training, row_indices_for_validation) = self._get_fold(fold)

            # train and get fit
            fit = self._train(col_indices_for_inputs, row_indices_for_training)

            # validation score
            score += self._validate(col_indices_for_inputs, row_indices_for_validation, fit)

        # average MSE
        return score / float(self.number_of_folds)