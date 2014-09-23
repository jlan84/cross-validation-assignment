
## Regression Regularization

For this exercise you will be comparing Ridge Regression and LASSO regression to
Ordinary Least Squares.  You will also get experience with techniques of cross
validation.  We will be using [scikit-learn](http://scikit-
learn.org/stable/supervised_learning.html#supervised-learning) to fit our
models, do not worry about the details of how the library works though.  We will
get into the details of this next week.  Look to the [lecture](lecture.ipynb]
notebook for an example of what functions to use with what parameters.


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
from the UCI machine learning Repository for this exercise.  Feel free to play
around with any of the others that are [suited](http://archive.ics.uci.edu/ml/da
tasets.html?format=&task=reg&att=&area=&numAtt=&numIns=&type=&sort=nameUp&view=t
able) for regression as well.  This dataset is actually containe in scikit-
learn's built in datasets.


```python
# EXERCISE: Create a new linear regression model and fit it using the dataset

# Create linear regression object
linear = LinearRegression()

# TODO: Fit the data
```


```python
# EXERCISE: Compute the RMSE on the training data
```


```python
# EXERCISE: Examine the coefficients return from your model.  Maybe make a plot of these.
```


```python
# EXERCISE: Split your data into a training and test set (hold-out set).

# Play around with the ratio of these (i.e. 70%/30% train/test, 80%/20% train/test, etc.)
# and compute the fit on only the training data. Test the RMSE of your results on the test data.
```

## K-fold Cross-validation

In **k-fold cross-validation**, the training set is split into *k* smaller sets.
Then, for each of the k "folds":

1. trained model on *k-1* of the folds as training data
2. validate this model the remaining fold, using an appropriate metric

The performance measure reported by k-fold CV is then the average of the *k*
computed values. This approach can be computationally expensive, but does not
waste too much data, which is an advantage over having a fixed test subset.


```python
# EXERCISE: Repeat the above but this time use K-fold cross validation.
```


```python
# EXERCISE: Compare the RMSE for your hold-out set and K-fold cross validation
```


```python
# EXERCISE: Plot the learning curve for a standard ordinary least squares regression

# You might want to use: http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.cross_val_score.html
# and: http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.ShuffleSplit.html
```


```python
# EXERCISE: Complete Part 4 and the extra credit of the gradient descent sprint if you haven't already
```


```python
# EXERCISE: Use K-Fold cross validation to evaluate your gradient descent model and compare to the performance of scikit learn
```


```python
# EXERCISE: Plot a learning curve and test vs train error curve.
```

## Part 2: Regularization


```python
# EXERCISE: Now that you have experimented with linear regression we will begin exploring Ridge Regression

# Fit the same dataset but with a Ridge Regression with an alpha == 0.5 to start
```

Notice the linear regression is not defined for scenarios where the number of
features/parameters exceeds the number of observations. It performs poorly as
long as the number of sample is not several times the number of features.

One approach for dealing with overfitting is to **regularize** the regession
model.

The **ridge estimator** is a simple, computationally efficient regularization
for linear regression.

$$\hat{\beta}^{ridge} = \text{argmin}_{\beta}\left\{\sum_{i=1}^N (y_i - \beta_0
- \sum_{j=1}^k x_{ij} \beta_j)^2 + \alpha \sum_{j=1}^k \beta_j^2 \right\}$$

Typically, we are not interested in shrinking the mean, and coefficients are
standardized to have zero mean and unit L2 norm. Hence,

$$\hat{\beta}^{ridge} = \text{argmin}_{\beta} \sum_{i=1}^N (y_i - \sum_{j=1}^k
x_{ij} \beta_j)^2$$

$$\text{subject to } \sum_{j=1}^k \beta_j^2 < \alpha$$

Note that this is *equivalent* to a Bayesian model $y \sim N(X\beta, I)$ with a
Gaussian prior on the $\beta_j$:

$$\beta_j \sim \text{N}(0, \alpha)$$

The estimator for the ridge regression model is:

$$\hat{\beta}^{ridge} = (X'X + \alpha I)^{-1}X'y$$


```python
# EXERCISE: Make a plot of the training error and the testing error as a function of the alpha paramter
```

The regularization of the ridge is a *shrinkage*: the coefficients learned are
shrunk towards zero.

The amount of regularization is set via the `alpha` parameter of the ridge,
which is tunable. The `RidgeCV` method in `scikits-learn` automatically tunes
this parameter via cross-validation.


```python
# EXERCISE: Plot the parameters (coefficients) of the Ridge regression (y-axis) versus the value of the alpha parameter.  
# There will be as many lines as there are parameters.

from sklearn import preprocessing

k = X.shape[1]
alphas = np.linspace(0, 4)
params = np.zeros((len(alphas), k))
for i,a in enumerate(alphas):
    X_data = preprocessing.scale(X)
    y = Y
    fit = Ridge(alpha=a, normalize=True).fit(X_data, y)
    params[i] = fit.coef_

figure(figsize=(14,6))
for param in params.T:
    plt.plot(alphas, param)
```


```python
# EXERCISE: Plot the learning curve of the Ridge regression with different alpha parameters
```


```python
# EXERCISE: Plot the learning curves of the Ridge Regression and Ordinary Least Squares Regression.  Compare these two.
```

**The Lasso estimator** is useful to impose sparsity on the coefficients. In
other words, it is to be prefered if we believe that many of the features are
not relevant.

$$\hat{\beta}^{lasso} = \text{argmin}_{\beta}\left\{\frac{1}{2}\sum_{i=1}^N (y_i
- \beta_0 - \sum_{j=1}^k x_{ij} \beta_j)^2 + \lambda \sum_{j=1}^k |\beta_j|
\right\}$$

or, similarly:

$$\hat{\beta}^{lasso} = \text{argmin}_{\beta} \frac{1}{2}\sum_{i=1}^N (y_i -
\sum_{j=1}^k x_{ij} \beta_j)^2$$
$$\text{subject to } \sum_{j=1}^k |\beta_j| < \lambda$$

Note that this is *equivalent* to a Bayesian model $y \sim N(X\beta, I)$ with a
**Laplace** prior on the $\beta_j$:

$$\beta_j \sim \text{Laplace}(\lambda) =
\frac{\lambda}{2}\exp(-\lambda|\beta_j|)$$

Note how the Lasso imposes sparseness on the parameter coefficients:

### Repeat the above steps with LASSO Regression


```python
# Make a plot of the training error and the testing error as a function of the alpha paramter
```


```python
# EXERCISE: Plot the parameters (coefficients) of the LASSO regression (y-axis) versus the value of the alpha parameter.  
# There will be as many lines as there are parameters.
```


```python
# EXERCISE: Plot the learning curve of the LASSO regression with different alpha parameters

k = X.shape[1]
alphas = np.linspace(0.1, 3)
params = np.zeros((len(alphas), k))
for i,a in enumerate(alphas):
    X_data = preprocessing.scale(X)
    y = Y
    fit = linear_model.Lasso(alpha=a, normalize=True).fit(X_data, y)
    params[i] = fit.coef_

figure(figsize=(14,6))
for param in params.T:
    plt.plot(alphas, param)
```


```python
# EXERCISE: Plot the learning curves of the Ridge Regression, LASSO Regression, and 
# Ordinary Least Squares Regression.  Compare these all.
```
