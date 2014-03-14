import numpy as np
from lintrain import Trainer

# generate random data
x = np.random.rand(50, 4)
y = (30 * x[:, 0]) - (10 * x[:, 2]) + np.random.rand(1, 50)

t = Trainer(x, y)
t.debug = 2
t.run()

print "COLUMN COEFFICIENTS"
print t.fit

print "COLUMNS USED"
print t.column_indices
