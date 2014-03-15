__author__ = 'L. Nathan Perkins'
"""
A multiprocessor version of the trainer that distributes potential feature sets to processes. Each process then
performs the linear regression across the k-folds.

Note that this can be very memory intensive as each process must have a copy of the data.
"""
