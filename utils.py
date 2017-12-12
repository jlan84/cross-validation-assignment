from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler


class XyStandardizer(BaseEstimator, TransformerMixin):
    
    def __init__(self):
        self.X_scaler = StandardScaler()
        self.y_scaler = StandardScaler()
        
    def fit(self, X, y, *args, **kwargs):
        self.X_scaler.fit(X)
        self.y_scaler.fit(y.reshape(-1, 1))
        return self
    
    def transform(self, X, y, *args, **kwargs):
        return (self.X_scaler.transform(X),
                self.y_scaler.transform(y.reshape(-1, 1)).flatten())
