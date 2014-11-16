import multiprocessing
import numpy as np
from .. import lintrain
import worker


# Feature selection + linear regression training and validation toolkit
class Trainer(lintrain.BaseTrainer):
    task_queue = None
    result_queue = None
    number_of_processes = None

    def _start_workers(self):
        # establish communication queue
        self.task_queue = multiprocessing.JoinableQueue()
        self.result_queue = multiprocessing.Queue()

        # number of processes
        if self.number_of_processes is None:
            self.number_of_processes = multiprocessing.cpu_count()

        # make and start workers
        workers = [worker.Worker(self.task_queue, self.result_queue, self.x, self.y, self.folds, self.solver,
                                 self.scorer) for i in xrange(self.number_of_processes)]
        for w in workers:
            w.start()

    def _end_workers(self):
        # add a poison pill for each workers
        for i in xrange(self.number_of_processes):
            self.task_queue.put(None)

        # finish sending poison pill
        self.task_queue.join()

        # close queues
        self.task_queue.close()
        self.result_queue.close()

        # clean up
        self.task_queue = None
        self.result_queue = None

    def _run_feature_selection(self, forward=True, backward=False):
        # spin up workers
        self._start_workers()

        # actually run the feature selection
        lintrain.BaseTrainer._run_feature_selection(self, forward, backward)

        # end workers
        self._end_workers()

    def _do_forward_selection(self, col_indices_for_inputs, best_score):
        # column indices
        l_indices = np.shape(self.x)[1]

        # no more columns to add
        if len(col_indices_for_inputs) == l_indices:
            return None, best_score

        # count number of pending tasks
        tasks = 0

        # distribute all potential feature sets to processes
        for potential_index in xrange(l_indices):
            # already in linear regression
            if potential_index in col_indices_for_inputs:
                continue

            # add index to copy of list
            potential_col_indices = col_indices_for_inputs + [potential_index]

            # add potential task
            self.task_queue.put(worker.Task(potential_col_indices))

            # increment number of pending tasks
            tasks += 1

        # let processes score each feature set
        #self.task_queue.join()

        # new columns
        new_col_indices = None

        # collect responses
        while True:
            # get next result
            result = self.result_queue.get()

            # process result
            if self._is_better_score(best_score, result.score):
                best_score = result.score
                new_col_indices = result.column_indices

            # decrement number of pending tasks
            tasks -= 1

            # no more tasks
            if tasks == 0:
                break

        return new_col_indices, best_score

    def _do_backward_selection(self, col_indices_for_inputs, best_score):
        # no more columns to remove
        if len(col_indices_for_inputs) == 1:
            return None, best_score

        # count number of pending tasks
        tasks = 0

        # distribute all potential feature sets to processes
        for potential_index in col_indices_for_inputs:
            # remove index from a copy of the list
            potential_col_indices = [x for x in col_indices_for_inputs if x != potential_index]

            # add potential task
            self.task_queue.put(worker.Task(potential_col_indices))

            # increment number of pending tasks
            tasks += 1

        # let processes score each feature set
        #self.task_queue.join()

        # new columns
        new_col_indices = None

        # collect responses
        while True:
            # get next result
            result = self.result_queue.get()

            # process result
            if self._is_better_score(best_score, result.score):
                best_score = result.score
                new_col_indices = result.column_indices

            # decrement number of pending tasks
            tasks -= 1

            # no more tasks
            if tasks == 0:
                break

        return new_col_indices, best_score
