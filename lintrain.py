import random
import numpy as np
import utilities
from scorers.meansquare import MeanSquare


# Feature selection + linear regression training and validation toolkit
class Trainer:
    folds = []
    column_indices = None
    fit = None
    score = None
    debug = 0

    def __init__(self, x, y, scorer=MeanSquare, number_of_folds=5):
        self.x = x
        self.y = utilities.to_column_matrix(y)
        self.scorer = scorer()
        self.number_of_folds = number_of_folds

    def _set_folds(self):
        # get number rows
        l_indices = np.shape(self.x)[0]
        indices = range(l_indices)

        # always shuffle the same way for consistent k-fold
        random.seed(37444887)
        random.shuffle(indices)

        per_fold = l_indices / self.number_of_folds
        if l_indices % self.number_of_folds:
            per_fold += 1
        self.folds = [indices[i:i + per_fold] for i in xrange(0, l_indices, per_fold)]

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
        fit = np.linalg.lstsq(x, y.T)[0]

        return fit

    def _validate(self, col_indices_for_inputs, row_indices, fit):
        # get x and y
        x = self.x[np.ix_(row_indices, col_indices_for_inputs)]
        y = self.y[row_indices]

        # generate predictions
        predicted_y = np.dot(x, fit.T)

        # create tuples
        validation = zip(y, predicted_y)

        return common.score_predictions(validation)

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

    def _do_forward_selection(self, col_indices_for_inputs, highest_score):
        # column indices
        l_indices = np.shape(self.x)[1]

        # no more columns to add
        if len(col_indices_for_inputs) == l_indices:
            return None, highest_score

        # new columns
        new_col_indices = None

        for potential_index in xrange(l_indices):
            # already in linear regression
            if potential_index in col_indices_for_inputs:
                continue

            # add index to copy of list
            potential_col_indices = col_indices_for_inputs + [potential_index]

            # score it
            score = self._score(potential_col_indices)

            if highest_score is None or score > highest_score:
                highest_score = score
                new_col_indices = potential_col_indices

        # add column
        if new_col_indices and self.debug >= 2:
            print "Added column", new_col_indices[-1], "(score:", highest_score, ")"

        return new_col_indices, highest_score

    def _do_backward_selection(self, col_indices_for_inputs, highest_score):
        # new columns
        old_index = None
        old_col_indices = None

        for potential_index in col_indices_for_inputs:
            # remove index from copy of list
            potential_col_indices = [x for x in col_indices_for_inputs if x != potential_index]

            # score it
            score = self._score(potential_col_indices)

            if highest_score is None or score > highest_score:
                highest_score = score
                old_index = potential_index
                old_col_indices = potential_col_indices

        # add column
        if old_col_indices and self.debug >= 2:
            print "Added column", old_index, "(score:", highest_score, ")"

        return old_col_indices, highest_score

    def _forward_selection(self):
        score = 0.
        col_indices = []

        # start training
        while True:
            # add
            new_col_indices, score = self._do_forward_selection(col_indices, score)
            if new_col_indices is None:
                break

            col_indices = new_col_indices

        # set score
        self.score = score

        # set fit and column indices
        self.column_indices = col_indices
        self.fit = self._train(col_indices, range(np.shape(self.x)[0]))

        return col_indices

    def _bidirectional_selection(self):
        score = 0.
        col_indices = []

        # start training
        while True:
            # add
            new_col_indices, score = self._do_forward_selection(col_indices, score)
            if new_col_indices is not None:
                col_indices = new_col_indices
                continue

            # try removing
            new_col_indices, score = self._do_backward_selection(col_indices, score)
            if new_col_indices is not None:
                col_indices = new_col_indices
                continue

            break

        # set score
        self.score = score

        # set fit and column indices
        self.column_indices = col_indices
        self.fit = self._train(col_indices, range(np.shape(self.x)[0]))

        return col_indices

    def run(self):
        # prepare cross folds
        self._set_folds()

        # get best columns
        columns = self._forward_selection()

        if self.debug >= 1:
            print "Final score:", self.score
        if self.debug >= 2:
            print "Columns: ", columns
            print "Fit: ", self.fit

    def select_columns_from_matrix(self, p_x):
        return p_x[:, self.column_indices]

    def select_columns_from_vector(self, a_x):
        return a_x[self.column_indices]

    def apply_to_matrix(self, p_x):
        return np.dot(p_x[:, self.column_indices], self.fit.T)

    def apply_to_vector(self, a_x):
        return np.dot(a_x[self.column_indices], self.fit)
