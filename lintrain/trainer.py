import numpy as np
from basetrainer import BaseTrainer


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
