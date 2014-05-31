lin-train
=========

Linear Regression Feature Selection and Trainer

This is based on a linear regression trainer and feature selection class initially developed to help
analyze and make predictions for the MIT Big Data Challenge. The actual linear regression is run by
numpy, but the training class provides a simple way to do feature selection over a large feature space.
The trainer does k-fold cross validation to find features that improve validation scores. When complete,
the class has the linear coefficients as well as a score.

Dependencies: Python 2.7, numpy

Usage:

    from lintrain import Trainer

    t = Trainer(x, y) # x is a matrix, y is a vector or single column matrix
    t.debug = 2 # print detailed debugging information regarding the feature selection
    t.run_forward_selection() # run forward feature selection

    print t.column_indices # indices of columns in "x" that were selected
    print t.fit # linear coefficients
    print t.score # score identified by the trainer

    novel_y_prime = t.apply_to_vector(novel_x) # applies column selection and plugs values into linear equation

Methods
-------

The following attributes are available for instances of the Trainer class.

* `run_forward_selection(initial_columns=None, initial_score)` Will begin considering each
  feature (column in X) sequentially, and continue adding columns to the linear Regression 
  as long as the k-fold cross-validation score improves. You can optionally specify 
  initial columns, which will be used as a starting place (initial columns will remain
  in the regression, since forward selection only adds columns).

* `run_backward_selection(initial_columns=None, initial_score)` Will begin considering 
  each feature (column in X) sequentially, removing columns from the linear Regression 
  as long as the k-fold cross-validation score improves. You can optionally specify an 
  initial set, which will be used as a starting place (the final result will be a subset
  of the initial columns). By default, the backward selection uses all columns as the 
  initial set.

* `run_bidirectional_selection(initial_columns=None, initial_score)` Will consider adding
  OR removing features (columns in X), and will continue to do so as long as the addition
  or removal improves the k-fold cross-validation score. You can optionally specify 
  initial columns, which will be used as a starting place. By default, the the initial 
  column set is empty, so it will always begin by adding columns.

* `select_columns_from_matrix(p_x)` Takes a matrix where rows represent entries and
  columns represent features, and returns a matrix with just the features (columns)
  that were selected by the feature selection algorithm. Must be called after running
  the trainer class. The order of returned columns matches the order of the features
  in column_indices.

* `select_columns_from_vector(a_x)` Takes a vector where entries represent features
  related to a single entry, and returnsa vector with just the features (columns) that
  were selected by the feature selection algorithm. Must be called after running the
  trainer class. The order of returned columns matches the order of the features in
  column_indices.

* `apply_to_matrix(p_x)` Applies the feature selection process to novel values and
  returning a vector with the  value predicted. This first selects the features from
  the matrix of novel entries. Using the dot product, it multiplies the coefficients
  and sums the response to return the predicted values.

* `apply_to_vector(a_x)` Applies the feature selection process to novel entry and
  returning a single value (the prediction). This first selects the features from
  the vector of features. Using the dot product, it multiplies the coefficients
  and sums the response to return the predicted value.


Attributes
----------

The following attributes are available for instances of the Trainer class.

* `debug` Allows printing of information about the training process. Can be 0 (no debugging), 1 (minimal debugging) or
   2 (detailed debugging). Minimal debugging prints final scores and such data, while detailed debugging prints
   individual feature (column) additions and removals.

* `number_of_folds` The number of folds to use in the k-fold cross-validation. Defaults to
   5.

* `scorer` An instance of a class that inherits from scorers.scorer.Scorer, which will use a set of predicted values
  (y_prime) and actual values (y) to calculate a score representing how closely the predictions match the actual values.
   By default, the system uses the provided MeanSquare class that calculates this as the Mean Square Error (MSE).

* `score_` After running, this contains the score for the final fit. This is calculated based on the scorer provided. 

* `column_indices` After running, this contains the column indices for the matrix X representing the features that were
  selected to best minimize the predictions without over-fitting to the data provided. (Depending on how the class is
  called, features are either iteratively added or removed as long as the k-fold cross-validation score continues
  to improve).

* `fit` The linear coefficients that correspond with each column in column_indices.

Multiprocess
------------

To decrease the time for training, a Trainer class that runs across multiple Python
processes is available. It is a drop-in replacement for the Trainer function in the 
example above, and will default to using one process per CPU.

To use, replace the import line above with the following:

    from parallel.trainer import Trainer

To change the number of processes used, an added attribute is available:

* `number_of_processes` An integer specifying the number of processes to use (defaults to
  the number of CPUs available.

Utilities
---------

lin-train includes a few utility functions to help generate features, specifically turning
discrete integer features into columns of binary features representing different potential
discrete values. This is useful for turning day of the week or time of day values into
a set of binary values (e.g., "is Monday", "is Tuesday", etc).

* `utilities.add_int_as_categories(A, int_val, val_max, val_min=0, step=1)` Appends new columns
  onto the matrix A based on the integers contained in the vector or column matrix int_val (note
  that int_val must contain the same number of entries as the number of rows in A, as each integer
  value corresponds with one entry in the matrix). Integers in int_val must be discrete and between
  val_min and val_max (inclusive). Binary columns (either 0 or 1) are appended to A representing
  each discrete value based on the step size.

* `utilities.int_as_categories(int_val, val_max, val_min=0, step=1)` Creates a list of binary values
  representing the discrete value int_val as binary features. The integer in int_val must be discrete
  and between val_min and val_max (inclusive). The binary columns (either 0 or 1) returned represent
  each discrete value based on the step size.

