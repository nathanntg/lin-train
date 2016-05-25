import numpy as np


def to_column_matrix(arr_or_mat):
    if len(arr_or_mat.shape) == 1:
        arr_or_mat.shape = [len(arr_or_mat), 1]
    elif np.shape(arr_or_mat)[0] == 1:
        return arr_or_mat.T
    return arr_or_mat


def add_int_as_categories(A, int_val, val_max, val_min=0, step=1):
    # reshape int_val to vector column
    col_int_val = to_column_matrix(int_val)

    # add new zero columns
    new_cols = np.zeros((np.shape(A)[0], (1 + val_max - val_min) / step), dtype=A.dtype)

    # values
    for i in xrange(val_min, val_max + 1, step):
        A[:, (col_int_val == i)] = 1

    return np.concatenate((A, new_cols))


def int_as_categories(int_val, val_max, val_min=0, step=1):
    zeroes = [0] * ((1 + val_max - val_min) / step)
    zeroes[(int_val - val_min) / step] = 1
    return zeroes
