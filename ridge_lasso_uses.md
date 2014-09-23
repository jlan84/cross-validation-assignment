## L1 regularization [Lasso regression](http://statweb.stanford.edu/~tibs/lasso/simple.html)
Math description: Regularization adds a random variable (for L1, a Laplacian) to the hat matrix so that it can be inverted.

  1. Turns most regressors to zeros
  2. Uses a Laplacian prior

### When to use: 

  1. Large sparse data sets, many regressors will become zero. 
  2. When you have many regressors but are unsure of the important ones.

<u>Pros</u>:

  1. Good for recovering sparse datasets
  2. Reduce overfitting

<u>Cons</u>:

  1. More difficult to interpret
  2. Loss of predictive power
  3. Large estimation error for non-sparse data.

## L2 regularization (Ridge regression):
1. Ridge regression suppresses the influence of the leading regressors lightly and the lagging regressors  heavily. 
2. Uses a Gaussian prior

### When to use: 
  1. When you have many regressors but are unsure of the important ones
  2. Non-sparse data. 

<u>Pros</u>:
  1. Good for recovering non-sparse signals. 
  2. Reduce overfitting.
  3. Less variance than the OLS estimator [reference](http://tamino.wordpress.com/2011/02/12/ridge-regression/)

<u>Cons</u>:

  1. The new estimates of the regressors are lower than the OLS estimates [reference](http://tamino.wordpress.com/2011/02/12/ridge-regression/)
  2. Loss of predictive power


## <u>More references</u>:

* [Ridge regression](http://tamino.wordpress.com/2011/02/12/ridge-regression/)
* [Lasso regression](http://statweb.stanford.edu/~tibs/lasso/simple.html)
* [Difference between L1 and L2](http://www.quora.com/Machine-Learning/What-is-the-difference-between-L1-and-L2-regularization), Aleks Jakulins answer. 
* [Matrix for of regression models](http://global.oup.com/booksites/content/0199268010/samplesec3)
* [The statistics bible](http://statweb.stanford.edu/~tibs/ElemStatLearn/printings/ESLII_print10.pdf), chapter 3
* [stats.stackexchange: Ridge vs. LASSO](http://stats.stackexchange.com/questions/866/when-should-i-use-lasso-vs-ridge)
* [MetaOptimize: L1 vs. L2](http://metaoptimize.com/qa/questions/5205/when-to-use-l1-regularization-and-when-l2)

