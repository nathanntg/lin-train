lin-train
=========

Linear Regression Feature Selection and Trainer

This is based on a linear regression trainer and feature selection class initially developed to help
analyze and make predictions for the MIT Big Data Challenge. The actual linear regression is run by
numpy, but the training class provides a simple way to do feature selection over a large feature space.
The trainer does k-fold cross validation to find features that improve validation scores. When complete,
the class has the linear coefficients as well as a score.

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
