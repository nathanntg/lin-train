import abc
import random
import numpy as np
import utilities
from scorers.meansquare import MeanSquare
from train import Train


class BaseTrainer(Train):
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

    @abc.abstractmethod
    def _do_forward_selection(self, col_indices_for_inputs, best_score):
        """
        Look at all potential indices (features) and determine which one to add that will most improve the score.
        Should return tuple (new_column_indices, best_score) if a column is added.
        Should return tuple (None, best_score) if no column added.
        """
        return

    @abc.abstractmethod
    def _do_backward_selection(self, col_indices_for_input, best_score):
        """
        Look through currently used indices (features) and determine which one to remove to most improve the score.
        Should return tuple (new_column_indices, best_score) if a column is removed
        Should return tuple (None, best_score) if no column removed.
        """
        return

    def _is_better_score(self, best_score, score):
        return best_score is None or ((score - best_score) * self.scorer.sort()) > 0

    def _configure_initial_conditions(self, initial_column_indices=None, initial_score=None, backward=False):
        self.score = None

        # allow starting with initial column selection, must calculate score first
        if initial_column_indices:
            self.column_indices = initial_column_indices
            if not initial_score:
                self.score = self._score(initial_column_indices)
            else:
                self.score = initial_score
        else:
            if backward:
                self.column_indices = range(np.shape(self.x)[1])
                self.score = self._score(self.column_indices)
            else:
                self.column_indices = []

    def _run_feature_selection(self, forward=True, backward=False):
        # get initial values
        col_indices = self.column_indices
        score = self.score

        # start training
        while True:
            # if running forward selection or bidirectional selection...
            if forward:
                # try adding a column
                new_col_indices, score = self._do_forward_selection(col_indices, score)
                if new_col_indices is not None:
                    # debugging
                    if self.debug >= 2:
                        print "Added column", new_col_indices[-1], "(score:", score, ")"

                    col_indices = new_col_indices
                    continue

            # if running backwards selection or bidirectional selection...
            if backward:
                # try removing a column
                new_col_indices, score = self._do_backward_selection(col_indices, score)
                if new_col_indices is not None:
                    # debugging
                    if self.debug >= 2:
                        print "Removed column", (set(col_indices)-set(new_col_indices)).pop(), "(score:", score, ")"

                    col_indices = new_col_indices
                    continue

            break

        # set score
        self.score = score

        # set fit and column indices
        self.column_indices = col_indices
        self.fit = self._train(col_indices, range(np.shape(self.x)[0]))

        # print debugging information
        if self.debug >= 1:
            print "Final score:", self.score
        if self.debug >= 2:
            print "Columns: ", self.column_indices
            print "Fit: ", self.fit

    def run_forward_selection(self, initial_columns=None, initial_score=None):
        # prepare cross folds
        self._set_folds()

        # configure
        self._configure_initial_conditions(initial_columns, initial_score)

        # get best columns
        self._run_feature_selection()

    def run_bidirectional_selection(self, initial_columns=None, initial_score=None):
        # prepare cross folds
        self._set_folds()

        # configure
        self._configure_initial_conditions(initial_columns, initial_score)

        # get best columns
        self._run_feature_selection(backward=True)

    def run_backward_selection(self, initial_columns=None, initial_score=None):
        # prepare cross folds
        self._set_folds()

        # configure
        self._configure_initial_conditions(initial_columns, initial_score, backward=True)

        # get best columns
        self._run_feature_selection(backward=True, forward=False)

    def select_columns_from_matrix(self, p_x):
        return p_x[:, self.column_indices]

    def select_columns_from_vector(self, a_x):
        return a_x[self.column_indices]

    def apply_to_matrix(self, p_x):
        return np.dot(p_x[:, self.column_indices], self.fit.T)

    def apply_to_vector(self, a_x):
        return np.dot(a_x[self.column_indices], self.fit)


# Feature selection + linear regression training and validation toolkit
class Trainer(BaseTrainer):
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

            if self._is_better_score(best_score, score):
                best_score = score
                new_col_indices = potential_col_indices

        return new_col_indices, best_score

    def _do_backward_selection(self, col_indices_for_inputs, best_score):
        # new columns
        old_col_indices = None

        for potential_index in col_indices_for_inputs:
            # remove index from copy of list
            potential_col_indices = [x for x in col_indices_for_inputs if x != potential_index]

            # score it
            score = self._score(potential_col_indices)

            if self._is_better_score(best_score, score):
                best_score = score
                old_col_indices = potential_col_indices

        return old_col_indices, best_score
