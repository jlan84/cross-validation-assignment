from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
import numpy as np
from sklearn.datasets import load_boston
from sklearn.metrics import mean_squared_error

boston = load_boston()
X = boston.data # housing features
y = boston.target # housing prices

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25)

def rmse(true, predicted):
    accum = 0
    for i in range(true.shape[0]):
        accum += (true[i]-predicted[i])**2
    return (accum/true.shape[0])**0.5

# a = np.array([1,2,3])
# b = np.array([4,1,3])
# print(rmse(a, b))
# print(mean_squared_error(a,b)**0.5)

