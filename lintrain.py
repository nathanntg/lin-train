import random
import numpy as np
import utilities
from scorers.meansquare import MeanSquare
from train import Train


# Feature selection + linear regression training and validation toolkit
class Trainer(Train):
    column_indices = None
    fit = None
    score = None
    debug = 0

    def __init__(self, x, y, scorer=MeanSquare, number_of_folds=5):
        Train.__init__(self)
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

    def _do_forward_selection(self, col_indices_for_inputs, best_score):
        # column indices
        l_indices = np.shape(self.x)[1]

        # no more columns to add
        if len(col_indices_for_inputs) == l_indices:
            return None, best_score

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

            if best_score is None or ((score - best_score) * self.scorer.sort()) > 0:
                best_score = score
                new_col_indices = potential_col_indices

        # add column
        if new_col_indices and self.debug >= 2:
            print "Added column", new_col_indices[-1], "(score:", best_score, ")"

        return new_col_indices, best_score

    def _do_backward_selection(self, col_indices_for_inputs, best_score):
        # new columns
        old_index = None
        old_col_indices = None

        for potential_index in col_indices_for_inputs:
            # remove index from copy of list
            potential_col_indices = [x for x in col_indices_for_inputs if x != potential_index]

            # score it
            score = self._score(potential_col_indices)

            if best_score is None or ((score - best_score) * self.scorer.sort()) > 0:
                best_score = score
                old_index = potential_index
                old_col_indices = potential_col_indices

        # add column
        if old_col_indices and self.debug >= 2:
            print "Removed column", old_index, "(score:", best_score, ")"

        return old_col_indices, best_score

    def _forward_selection(self):
        score = None
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
        # allow starting with initial column selection, must calculate score first
        if self.column_indices:
            col_indices = self.column_indices
            score = self._score(col_indices)
        else:
            col_indices = []
            score = None

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
