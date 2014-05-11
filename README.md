Main teacher: Jonathan and Oren

## Introduction

![lasso](https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcTnUsvQ-lHQeY-oAftGIeWpJ2E02HReXf5BVrVf5hgtXGUkalLOUQ)

* Regression is a best-fit line.
* Least-squares regression minimizes squared error

### Goals
* Cross Validation
* Overfitting
* LASSO (L1) Regularization
* Ridge (L2) Regularization

### Some approaches to Regression

#### Bayesian

* [Sample code](http://scrogster.wordpress.com/2011/04/05/pymc-for-bayesian-models/)
* Making inferences
* Non-normal responses

#### Frequentist

* How we compute it (with [drawings](http://thomaslevine.com/!/statistics-with-doodles-sudoroom))
* [Sample code](http://scikit-learn.org/stable/modules/linear_model.html)
* Making inferences
* Generalized linear models, link functions

### Fit, validity, &c.

* Regression draws lines, even if that seems weird.
  * Residuals ~ Independent variable (if there's only one)
  * Residuals ~ Fitted Values
* Independence, collinearity, &c.

### Other things of note

* Hierarchical models
* Representing categorical variables
* Detecting outliers

## Exercise

Fill in the necessary cells in the [regression.ipynb](regression.ipynb) notebook.

### Univariate Regression
1. Using pandas, plot a scatterplot matrix to explore the different columns (features) of the dataset.
2. Pick 2 features: one the regressor/predictor vs. the other as the regressand/response
3. Manually fit a line trying to minimize the residuals by experimenting with the slope and intercept of the line.
4. Once you have a line that looks like it is a good fit (and only then), start computing the RMSE (or r-squared) for your fit.  Use this statistic (RMSE) to fine turn the parameters of your line.
5. (Extra) Plot the RMSE as a function of slope/intercept combinations.  You probably want to use a heatmap for this in a similar fashion to what you did for the Kolmorgorov-Smirnov test.
6. Compare your manually fit line to the result of a [scipy.stats](http://docs.scipy.org/doc/scipy/reference/stats.html) or [statsmodels](http://statsmodels.sourceforge.net/) regression.   Look at RMSE, r-squared, or any other parameter. 

### Multivariate Regression

Now that you are comfortable/familiar with regression with one predictor, we will begin to play with many predictors!

1. Using the paris wifi data (or anything, really), fit one regression using either scipy.stats or statsmodels.
2. Inspect the output it gives you.
3. Evalute the fit of the line.
4. Interpret the coefficients, and comment as to whether this was an appropriate model.

## Multicolinearity and heteroscedasticity

* What is the significance of the RANK of a matrix, its DETERMINANT and LINEAR REGRESSION.
* Try to come up with intuitive model of the tradeoff between the number of regressors and a minimum value for adjusted R-squared. (hint: A matrix of regressor combinations may explaint the tradeoff)
* What is the simplest example of data in need of heteroscedastic treatment that you can come up with?
* Why should you manage the existence of any collinearity or multicollinearity first before looking for the presence of heteroscedasticity?
* Is there multicolinearity in the Cars data set? How many regressors do you suspect? 
* Can you find individual groups that display heteroscedastic behaviour? 
* Did the correction for multicolinearity effect the heteroscedasticity of the regression? 

## Extra Credit
Try a different sort of regression analysis. Here are some possibilities.


* fit one regression in a Bayesian manner and another in a frequentist manner.
   * Maybe make a graph too and post it on your blog.
* Use the regression line to make predictions.
* Use PyMC to build a hierarchical model.
* Build a generalized linear model that uses a link function other than the
    identity function. (Logistic regression is a decent choice.)
* Run a regression in R (with `lm`, `glm`, `MASS::rlm`, `lme4::lmer`, &c.).
* Create more features, either by transforming the dataset or joining
    with another one. Then figure out how to use these best in a regression.

## Data
Datasets are explained more [here](https://github.com/tlevine/comparing-wifi-usage).

## Resources

Ordinary least-squares regression:
[Khan Academy](https://www.khanacademy.org/math/probability/regression)

Generalized linear models and hierarchical models:
[Data Analysis Using Regression and Multilevel/Hierarchical Models](http://www.stat.columbia.edu/~gelman/arm/)

Here are some guides to regression modeling in Python in R.
They're good references if you already know something about regression,
but they'll probably make no sense if you don't.

* [statsmodels](http://statsmodels.sourceforge.net/) (Python)
* [lm](http://www.statmethods.net/stats/regression.html) and
    [glm](http://www.statmethods.net/advstats/glm.html) in R
