lin-train
=========

Linear Regression Feature Selection and Trainer

This is based on a linear regression trainer and feature selection class initially developed to help
analyze and make predictions for the MIT Big Data Challenge. The actual linear regression is run by
numpy, but the training class provides a simple way to do feature selection over a large feature space.
The trainer does k-fold cross validation to find features that improve validation scores. When complete,
the class has the linear coefficients as well as a score.

Usage:

    import lintrain

    t = new Trainer(x,y) # x is a matrix, y is a vector or single column matrix
    t.debug = 2 # print detailed debugging information regarding the feature selection
    t.run()

    print t.column_indices # indices of columns in "x" that were selected
    print t.fit # linear coefficients

    novel_y_prime = t.apply_to_vector(novel_x) # applies column selection and plugs values into linear equation
