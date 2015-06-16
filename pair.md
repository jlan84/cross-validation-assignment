## Regularization

### Now that you have experimented with linear regression we will begin exploring Ridge Regression.


**Notice the linear regression is not defined for scenarios where the number of
features/parameters exceeds the number of observations. It performs poorly as
long as the number of samples is not several times the number of features.**

One approach for dealing with overfitting is to **regularize** the regession
model.

The **ridge estimator** is a simple and computationally efficient regularization
for linear regression.

<img src="http://latex.codecogs.com/gif.latex?%5Clarge%20%5Chat%7B%5Cbeta%7D%5E%7Bridge%7D%20%3D%20%5Ctext%7Bargmin%7D_%7B%5Cbeta%7D%5Cleft%5C%7B%5Csum_%7Bi%3D1%7D%5EN%20%5Cleft%28y_i%20-%20%5Cbeta_0%20-%20%5Csum_%7Bj%3D1%7D%5Ek%20x_%7Bij%7D%20%5Cbeta_j%20%5Cright%29%5E2%20&plus;%20%5Calpha%20%5Csum_%7Bj%3D1%7D%5Ek%20%5Cbeta_j%5E2%20%5Cright%5C%7D" />

Typically, we are not interested in shrinking the mean. Furthermore, coefficients are
standardized to have zero mean and unit L2 norm. Hence,

<img src="http://latex.codecogs.com/gif.latex?%5Clarge%20%5Chat%7B%5Cbeta%7D%5E%7Bridge%7D%20%3D%20%5Ctext%7Bargmin%7D_%7B%5Cbeta%7D%20%5Csum_%7Bi%3D1%7D%5EN%20%5Cleft%20%28y_i%20-%20%5Csum_%7Bj%3D1%7D%5Ek%20x_%7Bij%7D%20%5Cbeta_j%20%5Cright%20%29%5E2" /> <img src="http://latex.codecogs.com/gif.latex?%5Clarge%20%5Ctext%7B%20subject%20to%20%7D%20%5Csum_%7Bj%3D1%7D%5Ek%20%5Cbeta_j%5E2%20%3C%20%5Calpha" />

Note that this is *equivalent* to a Bayesian model <img src="http://latex.codecogs.com/gif.latex?%5Clarge%20y%20%5Csim%20N%28X%5Cbeta%2C%20I%29" /> with a
Gaussian prior on the <img src="http://latex.codecogs.com/gif.latex?%5Clarge%20%5Cbeta_j" />: 

<img src="http://latex.codecogs.com/gif.latex?%5Clarge%20%5Cbeta_j%20%5Csim%20%5Ctext%7BN%7D%280%2C%20%5Calpha%29" />

The estimator for the ridge regression model is:

<img src="http://latex.codecogs.com/gif.latex?%5Clarge%20%5Chat%7B%5Cbeta%7D%5E%7Bridge%7D%20%3D%20%28X%27X%20&plus;%20%5Calpha%20I%29%5E%7B-1%7DX%27y" />

where `X'` denotes transpose of `X`.


#### Shrinkage

The regularization of the ridge is a *shrinkage*: the coefficients learned are shrunk towards zero.

The amount of regularization is set via the `alpha` parameter of the ridge,
which is tunable. The `RidgeCV` method in `scikits-learn` automatically tunes
this parameter via cross-validation.



### Exercise:

1. Fit the same Boston housing dataset but with a Ridge Regression with an `alpha = 0.5` to start
2. Plot the parameters (coefficients) of the Ridge regression (y-axis) versus the value of the alpha parameter. (There will be as many lines as there are predictors)

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


3. Plot the test error and training error curves for Ridge regression with different alpha parameters.
   Which model would you select based on your test and training curves?



## Lasso

**The Lasso estimator** is useful for imposing sparsity on the coefficients. In
other words, it is generally preferred if we believe many of the features are
not relevant.

<img src="http://latex.codecogs.com/gif.latex?%5Clarge%20%5Chat%7B%5Cbeta%7D%5E%7Blasso%7D%20%3D%20%5Ctext%7Bargmin%7D_%7B%5Cbeta%7D%5Cleft%5C%7B%5Cfrac%7B1%7D%7B2%7D%5Csum_%7Bi%3D1%7D%5EN%20%28y_i%20-%20%5Cbeta_0%20-%20%5Csum_%7Bj%3D1%7D%5Ek%20x_%7Bij%7D%20%5Cbeta_j%29%5E2%20&plus;%20%5Clambda%20%5Csum_%7Bj%3D1%7D%5Ek%20%7C%5Cbeta_j%7C%20%5Cright%5C%7D" />

or, similarly:

<img src="http://latex.codecogs.com/gif.latex?%5Clarge%20%5Chat%7B%5Cbeta%7D%5E%7Blasso%7D%20%3D%20%5Ctext%7Bargmin%7D_%7B%5Cbeta%7D%20%5Cfrac%7B1%7D%7B2%7D%5Csum_%7Bi%3D1%7D%5EN%20%28y_i%20-%20%5Csum_%7Bj%3D1%7D%5Ek%20x_%7Bij%7D%20%5Cbeta_j%29%5E2%5Ctext%7B%20subject%20to%20%7D%20%5Csum_%7Bj%3D1%7D%5Ek%20%7C%5Cbeta_j%7C%20%3C%20%5Clambda%24%24" />

Note that this is *equivalent* to a Bayesian model <img src="http://latex.codecogs.com/gif.latex?%5Clarge%20y%20%5Csim%20N%28X%5Cbeta%2C%20I%29" /> with a
**Laplace** prior on the <img src="http://latex.codecogs.com/gif.latex?%5Clarge%20%5Cbeta_j" />: 

<img src="http://latex.codecogs.com/gif.latex?%5Clarge%20%5Cbeta_j%20%5Csim%20%5Ctext%7BLaplace%7D%28%5Clambda%29%20%3D%20%5Cfrac%7B%5Clambda%7D%7B2%7D%5Cexp%28-%5Clambda%7C%5Cbeta_j%7C%29" />

Note how the Lasso imposes sparseness on the parameter coefficients:

### Exercise:

 1. Plot the parameters (coefficients) of the LASSO regression (y-axis) versus the value of the alpha parameter.
 2. Make a plot of the training error and the testing error as a function of the alpha parameter.
 3. Select a model based on the test and training error curves.


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
1.  Finally, compare three models:  your chosen Ridge model, your chosen Lasso model, and your chosen Ordinary Least Squares model.
