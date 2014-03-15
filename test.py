import numpy as np
# from lintrain import Trainer
from parallel.trainer import Trainer

# generate random data
num_entries = 50
num_features = 10
x = np.random.rand(num_entries, num_features)
y = (30 * x[:, 0]) - (10 * x[:, 2]) + np.random.rand(1, num_entries)

t = Trainer(x, y)
t.debug = 2
# t.run_forward_selection()
# t.run_backward_selection()
t.run_bidirectional_selection([1, 3])

print "COLUMN COEFFICIENTS"
print t.fit

print "COLUMNS USED"
print t.column_indices
