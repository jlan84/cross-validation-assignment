
## Regression Regularization

For this Exercise you will be comparing Ridge Regression and LASSO regression to
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
from the UCI machine learning Repository for this Exercise.  Feel free to play
around with any of the others that are [suited](http://archive.ics.uci.edu/ml/da
tasets.html?format=&task=reg&att=&area=&numAtt=&numIns=&type=&sort=nameUp&view=t
able) for regression as well.  This dataset is actually containe in scikit-
learn's built in datasets.


### Exercise 
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



### Exercise
1. Repeat the above but this time use K-fold cross validation.
2. Compare the RMSE for your hold-out set and K-fold cross validation.
3. Plot the learning curve for a standard ordinary least squares regression (You might want to use: [cross_val_score](http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.cross_val_score.html) and [ShuffleSplit](http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.ShuffleSplit.html)).
4. Complete Part 4 and the extra credit of the gradient descent sprint if you haven't already.
5. Use K-Fold cross validation to evaluate your gradient descent model and compare to the performance of scikit learn
6. Plot a learning curve and test vs train error curve.


## Part 2: Regularization



### Exercise: 
1. Now that you have experimented with linear regression we will begin exploring Ridge Regression.
2. Fit the same dataset but with a Ridge Regression with an `alpha = 0.5` to start


Notice the linear regression is not defined for scenarios where the number of
features/parameters exceeds the number of observations. It performs poorly as
long as the number of sample is not several times the number of features.

One approach for dealing with overfitting is to **regularize** the regession
model.

The **ridge estimator** is a simple, computationally efficient regularization
for linear regression.

<img src="http://latex.codecogs.com/gif.latex?$$\hat{\beta}^{ridge}&space;=&space;\text{argmin}_{\beta}\left\{\sum_{i=1}^N&space;(y_i&space;-&space;\beta_0&space;-&space;\sum_{j=1}^k&space;x_{ij}&space;\beta_j)^2&space;&plus;&space;\alpha&space;\sum_{j=1}^k&space;\beta_j^2&space;\right\}$$" title="$$\hat{\beta}^{ridge} = \text{argmin}_{\beta}\left\{\sum_{i=1}^N (y_i - \beta_0 - \sum_{j=1}^k x_{ij} \beta_j)^2 + \alpha \sum_{j=1}^k \beta_j^2 \right\}$$" />

Typically, we are not interested in shrinking the mean, and coefficients are
standardized to have zero mean and unit L2 norm. Hence,

<img src="http://latex.codecogs.com/gif.latex?$$\hat{\beta}^{ridge}&space;=&space;\text{argmin}_{\beta}&space;\sum_{i=1}^N&space;(y_i&space;-&space;\sum_{j=1}^k&space;x_{ij}&space;\beta_j)^2$$" title="$$\hat{\beta}^{ridge} = \text{argmin}_{\beta} \sum_{i=1}^N (y_i - \sum_{j=1}^k x_{ij} \beta_j)^2$$" />

<img src="http://latex.codecogs.com/gif.latex?$$\text{subject&space;to&space;}&space;\sum_{j=1}^k&space;\beta_j^2&space;<&space;\alpha$$" title="$$\text{subject to } \sum_{j=1}^k \beta_j^2 < \alpha$$" />

Note that this is *equivalent* to a Bayesian model <img src="http://latex.codecogs.com/gif.latex?$y&space;\sim&space;N(X\beta,&space;I)$" title="$y \sim N(X\beta, I)$" /> with a
Gaussian prior on the <img src="http://latex.codecogs.com/gif.latex?$\beta_j$" title="$\beta_j$" />:

<img src="http://latex.codecogs.com/gif.latex?$$\beta_j&space;\sim&space;\text{N}(0,&space;\alpha)$$" title="$$\beta_j \sim \text{N}(0, \alpha)$$" />

The estimator for the ridge regression model is:

<img src="http://latex.codecogs.com/gif.latex?$$\hat{\beta}^{ridge}&space;=&space;(X'X&space;&plus;&space;\alpha&space;I)^{-1}X'y$$" title="$$\hat{\beta}^{ridge} = (X'X + \alpha I)^{-1}X'y$$" />



### Exercise:
 
 1. Make a plot of the training error and the testing error as a function of the alpha parameter.

#### Shrinkage

The regularization of the ridge is a *shrinkage*: the coefficients learned are shrunk towards zero.

The amount of regularization is set via the `alpha` parameter of the ridge,
which is tunable. The `RidgeCV` method in `scikits-learn` automatically tunes
this parameter via cross-validation.



### Exercise
1. Plot the parameters (coefficients) of the Ridge regression (y-axis) versus the value of the alpha parameter. (There will be as many lines as there are parameters)

```python
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


### Exercise: 
1. Plot the learning curve of the Ridge regression with different alpha parameters
2. Plot the learning curves of the Ridge Regression and Ordinary Least Squares Regression.  Compare these two.




## Lasso

**The Lasso estimator** is useful to impose sparsity on the coefficients. In
other words, it is to be prefered if we believe that many of the features are
not relevant.

<img src="http://latex.codecogs.com/gif.latex?$$\hat{\beta}^{lasso}&space;=&space;\text{argmin}_{\beta}\left\{\frac{1}{2}\sum_{i=1}^N&space;(y_i&space;-&space;\beta_0&space;-&space;\sum_{j=1}^k&space;x_{ij}&space;\beta_j)^2&space;&plus;&space;\lambda&space;\sum_{j=1}^k&space;|\beta_j|&space;\right\}$$" title="$$\hat{\beta}^{lasso} = \text{argmin}_{\beta}\left\{\frac{1}{2}\sum_{i=1}^N (y_i - \beta_0 - \sum_{j=1}^k x_{ij} \beta_j)^2 + \lambda \sum_{j=1}^k |\beta_j| \right\}$$" />
or, similarly:

<img src="http://latex.codecogs.com/gif.latex?$$\hat{\beta}^{lasso}&space;=&space;\text{argmin}_{\beta}&space;\frac{1}{2}\sum_{i=1}^N&space;(y_i&space;-&space;\sum_{j=1}^k&space;x_{ij}&space;\beta_j)^2$$&space;$$\text{subject&space;to&space;}&space;\sum_{j=1}^k&space;|\beta_j|&space;<&space;\lambda$$" title="$$\hat{\beta}^{lasso} = \text{argmin}_{\beta} \frac{1}{2}\sum_{i=1}^N (y_i - \sum_{j=1}^k x_{ij} \beta_j)^2$$ $$\text{subject to } \sum_{j=1}^k |\beta_j| < \lambda$$" />

Note that this is *equivalent* to a Bayesian model <img src="http://latex.codecogs.com/gif.latex?$y&space;\sim&space;N(X\beta,&space;I)$" title="$y \sim N(X\beta, I)$" /> with a
**Laplace** prior on the <img src="http://latex.codecogs.com/gif.latex?$\beta_j$" title="$\beta_j$" />:

<img src="http://latex.codecogs.com/gif.latex?$$\beta_j&space;\sim&space;\text{Laplace}(\lambda)&space;=&space;\frac{\lambda}{2}\exp(-\lambda|\beta_j|)$$" title="$$\beta_j \sim \text{Laplace}(\lambda) = \frac{\lambda}{2}\exp(-\lambda|\beta_j|)$$" />

Note how the Lasso imposes sparseness on the parameter coefficients:

### Exercise
 1. Make a plot of the training error and the testing error as a function of the alpha parameter.
 2. Plot the parameters (coefficients) of the LASSO regression (y-axis) versus the value of the alpha parameter. 
 3. Plot the learning curves of the Lasso Regression and Ordinary Least Squares Regression.  Compare these two.


```python
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



### Exercise: 
1. Plot the learning curves of the Ridge Regression, LASSO Regression, and Ordinary Least Squares Regression.  Compare all three. 

