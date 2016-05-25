import numpy as np
import multiprocessing
from train import Train


class Task(object):
    def __init__(self, column_indices):
        self.column_indices = column_indices


class Result(object):
    def __init__(self, column_indices, score):
        self.column_indices = column_indices
        self.score = score


class Worker(multiprocessing.Process, Train):
    def __init__(self, task_queue, result_queue, x, y, folds, solver, scorer):
        multiprocessing.Process.__init__(self)
        Train.__init__(self, x=x, y=y, solver=solver, scorer=scorer, number_of_folds=len(folds))

        # queues for inter-process communication
        self.task_queue = task_queue
        self.result_queue = result_queue

        # store folds
        self.folds = folds

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

    def _run_task(self, task):
        # score potential columns
        score = self._score( task.column_indices )

        # make result
        return Result(task.column_indices, score)

    def run(self):
        while True:
            next_task = self.task_queue.get()

            # no next task, break
            if next_task is None:
                # remove empty task from task queue
                self.task_queue.task_done()

                break

            # run task
            self.result_queue.put(self._run_task(next_task))

            # remove from task queue, and add result
            self.task_queue.task_done()
