# Regression in Practice

1. [Training and Testing](#training-and-testing)
1. [Regression with sklearn](#regression-with-sklearn)
1. [Fitting a polynomial](#fitting-a-polynomial)
1. [Bias and Variance](#bias-and-variance)
1. [Overfitting and Underfitting](#overfitting-and-underfitting)


## Training and Testing

You should always, before you begin, split your dataset into a train dataset
and a test dataset. You will use the train dataset to build your model and the
test dataset to measure your success.

You should generally keep 10-25% of the data for the test set and use the rest
for training.

You should always randomly split your data. Data often is sorted in some way (
by date or even by the value you are trying to predict). *Never* just split your
data into the first 90% and the remaining 10%. Lucky for us, there is a nice
method implemented in scipy that splits the dataset randomly for us called
[test_train_split](http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.train_test_split.html).

## Regression with sklearn

There are several good modules with implementations of regression. We've
used
[statsmodels](http://statsmodels.sourceforge.net/devel/generated/statsmodels.regression.linear_model.OLS.html).
Today we will be using [sklearn](http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html).
[numpy](docs.­scipy.­org/­doc/­numpy/­reference/­generated/­numpy.­polyfit.­html) and
[scipy](http://docs.scipy.org/doc/scipy-0.13.0/reference/generated/scipy.stats.linregress.html)
also have implmentations.

Resources:
* [sklearn documentation](http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html)
* [sklearn example](http://scikit-learn.org/0.11/auto_examples/linear_model/plot_ols.html)

For all `sklearn` modules, the `fit` method is used to train and the `score`
method is used to test. You can also use the `predict` method to see the
predicted y values.

#### Example

This is the general workflow for working with sklearn. Any algorithm we use from sklearn will have the same form.

```python
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.cross_validation import train_test_split

# Load data from csv file
df = pd.read_csv('data/housing_prices.csv')
X = df[['square_feet', 'num_rooms']]
y = df['price']

# Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15)

# Run Linear Regression
regr = LinearRegression()
regr.fit(X_train, y_train)
print "Intercept:", regr.intercept_
print "Coefficients:", regr.coef_
print "R^2 error:", regr.score(X_test, y_test)
predicted_y = regr.predict(X_test)
```

 `LinearRegression` is a class and you have to create an instance of it. If there are any parameters to the model, you should set them when you instantiate the object. For example, with `LinearRegression`, you can choose whether to normalize you data:

 ```python
 regr = LinearRegression(normalize=True)    # the default is False
 ```

You should call the `fit` method once. Here you give it the training data and it will train your model. Once you have that, you can get the coefficients for the equation (`intercept_` and `coef_`) and also get the score for your test set (`score` method). You can also get the predicted values for any new data you would like to give to the model (`predict` method).


## Fitting a polynomial

Oftentimes you'll notice that you're data isn't linear and that really you should be fitting a higher degree polynomial to the line. This is called *underfitting*, which we'll get to later.

![quadratic](images/quadratic.png)

So how do we do this? We can use the same algorithm, we just need to modify our features. Let's look at the one feature world for a minute. We have data that looks something like this:

|     x |     y |
| ----- | ----- |
|     3 |     8 |
|     4 |    17 |
|     7 |    40 |
|     9 |    78 |
|    11 |   109 |

 For linear regression, we are trying to find the `b` and `c` that minimize the error in the following equation:

    bx + c = y

To do a *quadratic* regression instead of a *linear* regression, we instead want to find the optimal `a`, `b` and `c` in this equation:

    ax^2 + bx + c = y

We can just add a new feature to our feature matrix by computing `x^2`:

|     x |   x^2 |    y |
| ----- | ----- | ----- |
|     3 |     9 |     8 |
|     4 |    16 |    17 |
|     7 |    49 |    40 |
|     9 |    81 |    78 |
|    11 |   121 |   109 |

Now you can do linear regression with these features. If there's a linear relationship between `x^2` and `y` that means there's a quadratic relationship between `x` and `y`.

If you have more than one feature, you need to do all the combinations. If you start with two features, `x` and `z`, to do the order 2 polynomial, you will need to add these features: `x^2`, `z^2`, `xz`.

In `sklearn`, you should use `PolynomialFeatures` for generating these additional features ([documentation](http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.PolynomialFeatures.html#sklearn.preprocessing.PolynomialFeatures)). Here's how you would modify the above example to include polynomial features up to degree 3:

```python
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.cross_validation import train_test_split

# Load data is identical as above
df = pd.read_csv('data/housing_prices.csv')
X = df[['square_feet', 'num_rooms']]
y = df['price']

# Add the polynomial features
poly = PolynomialFeatures(3)
X_new = poly.fit_transform(X)

# Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X_new, y, test_size=0.15)

# Run Linear Regression (the same as above)
regr = LinearRegression()
regr.fit(X_train, y_train)
print "Intercept:", regr.intercept_
print "Coefficients:", regr.coef_
print "R^2 error:", regr.score(X_test, y_test)
predicted_y = regr.predict(X_test)
```

Here's sklearn's [example](http://scikit-learn.org/stable/auto_examples/linear_model/plot_polynomial_interpolation.html). They use a `pipeline` to make the code a little simpler (with the same functionality).

In `numpy`, you can use the `polyfit` function ([documentation](http://docs.scipy.org/doc/numpy/reference/generated/numpy.polyfit.html)).

The discussion below will give insight into how to determine which degree of polynomial to pick.


## Bias and Variance

We first want to discuss two key terms: *bias* and *variance*.

#### Error due to bias
Imagine that you construct your model with the same process several hundred times. A *biased* model would center around the incorrect solution. How you collect data can lead to bias (for example, you only get user data from San Francisco and try to use your model for your whole userbase).

#### Error due to variance
Again, imagine that you construct your model several hundred times. A model with high *variance* would have wildly different results each time. The main contributor to high variance is insufficient data or that what you're trying to predict isn't actually correlated to your features.

To see the bais variance tradeoff visually:

![bias variance](images/bias_variance.png)

Note that both high bias or high variance are bad. Note that high variance is worse than it sounds since you will only be constructing the model once, so with high variance there's a low probability that your model will be near the optimal one.


## Overfitting and underfitting

Let's get back to fitting the polynomial. Take a look at the following example with three potential curves fit to the data.

![underfitting overfitting](images/underfitting_overfitting.png)

In the first graph, we've fit a line to the data. It clearly doesn't fully represent the data. This is called *underfitting*. This represents high *bias* since the error will be consistently high. It also represents low *variance* since with new data, we would still construct approximately the same model.

In the third graph, we've fit a polynomial of high degree to the data. Notice how it accurately gets every point but we'd say this does not accurately represent the data. This is called *overfitting*. This represents high *variance*. If we got new data, we would construct a wildly different model. This also represents low *bias* since the error is very low.

The one in the middle is the optimal choice.

You can see this graphically:

![bias variance](images/bias_variance_graph.png)

Model complexity in our case is the degree of the polynomial.

Another way of viewing this is by comparing the error on the training set with the error on the test set. When you fit a model, it minimizes the error on the training set. An overfit model can reduce this to 0. However, what we really care about it how well it does on a new test set, since we want our model to perform well on unseen data. This paradigm is represented with the following graph.

![overfitting underfitting](images/graph_overfitting_underfitting.png)

You can see on the left side of the graph that the data is *underfit*: both the train error and the test error are high. On the right hand side of the graph, the data is *overfit*: the train error is low but the test error is high. The best model is where the train error is lowest, around a degree 5 polynomial.
