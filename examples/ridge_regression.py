import numpy as np
from lintrain import Trainer
from lintrain.solvers import RidgeRegression

# use parallel trainer instead
# from lintrain import ParallelTrainer as Trainer

if "__main__" == __name__:
    # generate random data
    num_entries = 50
    num_features = 10
    x = np.random.rand(num_entries, num_features)
    y = (30 * x[:, 0]) - (10 * x[:, 2]) + np.random.rand(1, num_entries)

    # create trainer
    t = Trainer(x, y, solver=RidgeRegression)
    t.debug = 2

    # run
    #t.run_forward_selection()
    #t.run_backward_selection()
    t.run_bidirectional_selection([1, 3])

    # print output
    print "COLUMN COEFFICIENTS"
    print t.fit
    print ""

    print "COLUMNS USED"
    print t.column_indices
    print ""

    print "INPUT"
    print x[0, :]
    print "ACTUAL"
    print y[0, 0]
    print "PREDICTED"
    print t.apply_to_vector(x[0, :])
