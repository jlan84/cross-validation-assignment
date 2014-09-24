
## Cross Validation

For this Exercise you will be comparing Ridge Regression and LASSO regression to
Ordinary Least Squares.  You will also get experience with techniques of cross
validation.  We will be using [scikit-learn](http://scikit-
learn.org/stable/supervised_learning.html#supervised-learning) to fit our
models.


```python
%pylab inline

from sklearn.cross_validation import KFold
from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet
from sklearn.linear_model import LassoCV, RidgeCV
from sklearn.cross_validation import train_test_split
import numpy as np
import pylab as pl

from sklearn.datasets import load_boston

boston = load_boston()
X = np.array([np.concatenate((v,[1])) for v in boston.data])
Y = boston.target
```

    Populating the interactive namespace from numpy and matplotlib



```python
print Y[:10]
```

    [ 24.   21.6  34.7  33.4  36.2  28.7  22.9  27.1  16.5  18.9]



```python
print X[:2]
```

    [[  6.32000000e-03   1.80000000e+01   2.31000000e+00   0.00000000e+00
        5.38000000e-01   6.57500000e+00   6.52000000e+01   4.09000000e+00
        1.00000000e+00   2.96000000e+02   1.53000000e+01   3.96900000e+02
        4.98000000e+00   1.00000000e+00]
     [  2.73100000e-02   0.00000000e+00   7.07000000e+00   0.00000000e+00
        4.69000000e-01   6.42100000e+00   7.89000000e+01   4.96710000e+00
        2.00000000e+00   2.42000000e+02   1.78000000e+01   3.96900000e+02
        9.14000000e+00   1.00000000e+00]]


### Dataset

We will be using a [dataset](http://archive.ics.uci.edu/ml/datasets/Housing)
from the UCI machine learning Repository for this Exercise.  Feel free to play
around with any of the others that are [suited](http://archive.ics.uci.edu/ml/da
tasets.html?format=&task=reg&att=&area=&numAtt=&numIns=&type=&sort=nameUp&view=t
able) for regression as well.  This dataset is actually containe in scikit-
learn's built in datasets.


### Exercise:
 
1. Create a new linear regression model and fit it using the dataset.
2. Compute the RMSE on the training data.
3. Examine the coefficients return from your model.  Maybe make a plot of these.
4. Split your data into a training and test set (hold-out set) and compute the fit on only the training data. Test the RMSE of your results on the test data.
5. Experiment around with the ratio of these (i.e. 70%/30% train/test, 80%/20% train/test, etc.)


## K-fold Cross-validation

In **k-fold cross-validation**, the training set is split into *k* smaller sets.
Then, for each of the k "folds":

1. trained model on *k-1* of the folds as training data
2. validate this model the remaining fold, using an appropriate metric

The performance measure reported by k-fold CV is then the average of the *k*
computed values. This approach can be computationally expensive, but does not
waste too much data, which is an advantage over having a fixed test subset.



### Exercise:

1. Repeat the above but this time use K-fold cross validation.
2. Compare the RMSE for your hold-out set and K-fold cross validation.
3. Plot the learning curve for a standard ordinary least squares regression (You might want to use: [cross_val_score](http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.cross_val_score.html) and [ShuffleSplit](http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.ShuffleSplit.html)).
5. Use K-Fold cross validation to evaluate your gradient descent model (for yesterday) and compare to the performance of scikit learn
6. Plot a learning curve and test vs train error curve.
 
### Extra Credit: Stepwise Regression

While stepwise regression has its many [critics](http://andrewgelman.com/2014/06/02/hate-stepwise-regression/), it is a useful exercise to introduce the concept of feature selection in the context of linear regression. This extra credit exercise has two components of different difficulties. First, using the `scikit-learn` reverse feature elimation (a greedy feature elimination algorithm) to implement something similar to sequential backward selection. The second, more difficult part is an implementation of sequential forward selection.

1. Generate a series of of `n=5000` samples, `n=100` features, with a `random_seed=0` using the `make_friedman1` dataset like so:

```python
from sklearn.datasets import make_friedman1
X, y = make_friedman1(n_samples=5000, n_features=100, random_state=0)
```

2. Now, create a `LinearRegression()` object and pass it into the [RFE](http://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.RFE.html) selection algorithm.
3. Using a `for` loop, generate a series of models that take the top `n` features and calculate the `R^2` score using the `.score()` method.
4. Plot the `R^2` as a function of the number of included features. What does this plot tell you about the number of useful features in your model?
5. Extra extra credit. Instead of using RFE to do backward selection, create your own `LinearRegression` class that implements sequential forward selection, which involves starting with no variables in the model, testing the addition of each variable using a chosen model comparison criterion, adding the variable (if any) that improves the model the most, and repeating this process until none improves the model.
