import unittest
import numpy as np
from lintrain import Trainer
from lintrain import ParallelTrainer


class LinTrainTest(unittest.TestCase):
    """
    Must be run from outside of the LinTrain folder, otherwise parallel library fails due to relative paths.
    """
    def test_trainer(self):
        # generate random data
        num_entries = 50
        num_features = 10

        # seed random number generator for consistency
        np.random.seed(1238927)
        x = np.random.rand(num_entries, num_features)
        y = (30 * x[:, 0]) - (10 * x[:, 2]) + np.random.rand(1, num_entries)

        # run forward
        t = Trainer(x, y)

        # run forward selection
        t.run_forward_selection([1], None)

        # initial column should still be in there when training
        self.assertIn(1, t.column_indices)

        # should have at least one more column than initial data
        self.assertGreater(len(t.column_indices), 1)

        # run forward selection
        t.run_forward_selection()
        self.assertIn(1, t.column_indices)

        # run backward selection
        t.run_backward_selection(range(0, num_features), None)

        # column 0 should still be in there with 99.9% likelihood
        self.assertIn(0, t.column_indices)

        # should have at least one more column than initial data
        self.assertLess(len(t.column_indices), num_features)

    def test_parallel_trainer(self):
        # generate random data
        num_entries = 50
        num_features = 10

        # seed random number generator for consistency
        np.random.seed(1238927)
        x = np.random.rand(num_entries, num_features)
        y = (30 * x[:, 0]) - (10 * x[:, 2]) + np.random.rand(1, num_entries)

        # run forward
        t = ParallelTrainer(x, y)

        # run forward selection
        t.run_forward_selection([1], None)

        # initial column should still be in there when training
        self.assertIn(1, t.column_indices)

        # should have at least one more column than initial data
        self.assertGreater(len(t.column_indices), 1)

        # run forward selection
        t.run_forward_selection()
        self.assertIn(1, t.column_indices)

        # run backward selection
        t.run_backward_selection(range(0, num_features), None)

        # column 0 should still be in there with 99.9% likelihood
        self.assertIn(0, t.column_indices)

        # should have at least one more column than initial data
        self.assertLess(len(t.column_indices), num_features)


if __name__ == '__main__':
    unittest.main()
