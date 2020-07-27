from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
import numpy as np
from sklearn.datasets import load_boston
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score

boston = load_boston()
X = boston.data # housing features
y = boston.target # housing prices

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25)

def my_rmse(true, predicted):
    accum = 0
    for i in range(true.shape[0]):
        accum += (true[i]-predicted[i])**2
    return (accum/true.shape[0])**0.5

# a = np.array([1,2,3])
# b = np.array([4,1,3])
# print(rmse(a, b))
# print(mean_squared_error(a,b)**0.5)

# Fit your model using the training set
reg = KNeighborsRegressor(n_neighbors=19)
reg.fit(X_train, y_train)

# Call predict to get the predicted values for training and test set
train_predicted = reg.predict(X_train)
test_predicted = reg.predict(X_test)

# Calculate RMSE for training and test set
print( 'RMSE for training set ', my_rmse(y_train, train_predicted))
print( 'RMSE for test set ', my_rmse(y_test, test_predicted))

# print(np.mean(rmse(X_train, train_predicted)))
# print(np.mean(rmse(X_test, test_predicted)))
# print(train_predicted)
# print(len(X_train[1]))
# print(len(train_predicted))

# print(len(X_test))
# print(len(test_predicted))

# def crossVal(X_train, y_train, k):
#     kf = KFold(n_splits=k)
#     for train

reg = KNeighborsRegressor(n_neighbors=5)

def crossVal_SK(reg, X_train, y_train, k):
    return  np.mean(-1 * cross_val_score(reg, X_train, y_train, cv=k, scoring='neg_root_mean_squared_error'))

print(crossVal_SK(reg, X_train, y_train, 5))
    

def my_cross_val_scores(X_data, y_data, num_folds=3):
    ''' Returns error for k-fold cross validation. '''
    kf = KFold(n_splits=num_folds)
    train_error = np.empty(num_folds)
    test_error = np.empty(num_folds)
    index = 0
    reg = KNeighborsRegressor()
    for train, test in kf.split(X_data):
        reg.fit(X_data[train], y_data[train])
        pred_train = reg.predict(X_data[train])
        pred_test = reg.predict(X_data[test])
        train_error[index] = my_rmse(pred_train, y_data[train])
        test_error[index] = my_rmse(pred_test, y_data[test])
        index += 1
    return np.mean(test_error), np.mean(train_error)

print(my_cross_val_scores(X_train, y_train, 5))



def crossVal_SK_2(reg, X_train, y_train, k):
    scores = []
    this_score = cross_val_score(reg, X_train, y_train, cv=k, scoring='neg_root_mean_squared_error')
    scores.append(np.mean(this_score))
    return -1 * scores[0]

print(crossVal_SK_2(reg, X_train,y_train,5))