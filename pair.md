## Regularization

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



### Exercise:

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

### Exercise:

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
