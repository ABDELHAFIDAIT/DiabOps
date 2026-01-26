import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler


NON_ZERO_COLUMNS = [
    'Glucose', 
    'BloodPressure', 
    'SkinThickness', 
    'Insulin', 
    'BMI'
]


class ZeroToNan(BaseEstimator, TransformerMixin) :
    def __init__(self, columns=None):
        self.columns = columns
    
    def fit(self, X, y=None) :
        return self
    
    def transform(self, X) :
        X_copy = X.copy
        
        if isinstance(X_copy, pd.DataFrame) :
            for col in self.columns :
                if col in X_copy.columns :
                    X_copy[col] = X_copy[col].replace(0, np.nan)
            
        return X_copy
    


def create_pipeline() :
    pipeline = Pipeline([
        ('zero_to_nan', ZeroToNan(columns=NON_ZERO_COLUMNS)),
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    return pipeline


if __name__ == "__main__" :
    print("Pipeline définie avec succès !")
    print("Columns :", NON_ZERO_COLUMNS)