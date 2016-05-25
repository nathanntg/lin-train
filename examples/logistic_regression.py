import numpy as np
from lintrain import Trainer
from lintrain.solvers import LogisticRegression

# must be placed outside the package and reference updated to use the parallel trainer
# due to limitation related to relative references
# from parallel.trainer import Trainer

if "__main__" == __name__:
    # generate random data
    num_entries = 500
    num_features = 10
    x = np.random.rand(num_entries, num_features)
    y = (5 * x[:, 0]) + (2 * x[:, 2]) + np.random.rand(1, num_entries)
    y = 1 * (y > 3.5)

    # create trainer
    t = Trainer(x, y, solver=LogisticRegression)
    t.debug = 2

    # run
    t.run_forward_selection()
    #t.run_backward_selection()
    #t.run_bidirectional_selection([1, 3])

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
