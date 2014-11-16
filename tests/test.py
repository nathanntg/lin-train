import unittest
import numpy as np
import lintrain
import parallel.trainer as pl


class LinTrainTest(unittest.TestCase):
    """
    Must be run from outside of the LinTrain folder, otherwise parallel library fails due to relative paths.
    """
    def test_trainer(self):
        # generate random data
        num_entries = 50
        num_features = 10
        x = np.random.rand(num_entries, num_features)
        y = (30 * x[:, 0]) - (10 * x[:, 2]) + np.random.rand(1, num_entries)

        # run forward
        t = lintrain.Trainer(x, y)

        # run forward selection
        t.run_forward_selection([1], None)

        # initial column should still be in there when training
        self.assertIn(1, t.column_indices)

        # should have at least one more column than initial data
        self.assertGreater(len(t.column_indices), 1)

        # run forward selection
        t.run_backward_selection([0, 1, 2], None)

        # column 0 should still be in there with 99.9% likelihood
        self.assertIn(0, t.column_indices)

        # should have at least one more column than initial data
        self.assertLess(len(t.column_indices), 3)

    def test_parallel_trainer(self):
        # generate random data
        num_entries = 50
        num_features = 10
        x = np.random.rand(num_entries, num_features)
        y = (30 * x[:, 0]) - (10 * x[:, 2]) + np.random.rand(1, num_entries)

        # run forward
        t = pl.Trainer(x, y)

        # run forward selection
        t.run_forward_selection([1], None)

        # initial column should still be in there when training
        self.assertIn(1, t.column_indices)

        # should have at least one more column than initial data
        self.assertGreater(len(t.column_indices), 1)

        # run forward selection
        t.run_backward_selection([0, 1, 2], None)

        # column 0 should still be in there with 99.9% likelihood
        self.assertIn(0, t.column_indices)

        # should have at least one more column than initial data
        self.assertLess(len(t.column_indices), 3)


if __name__ == '__main__':
    unittest.main()
