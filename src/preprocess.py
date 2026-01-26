import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler


PHYSIOLOGICAL_COLS = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']

class ZeroToNaN(BaseEstimator, TransformerMixin):
    def __init__(self, columns=None):
        self.columns = columns

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_copy = X.copy()
        
        if isinstance(X_copy, pd.DataFrame):
            for col in self.columns:
                if col in X_copy.columns:
                    X_copy[col] = pd.to_numeric(X_copy[col], errors='coerce')
                    X_copy[col] = X_copy[col].mask(X_copy[col] == 0, np.nan)
        
        return X_copy


def create_pipeline():
    
    return Pipeline(steps=[
        ('zero_to_nan', ZeroToNaN(columns=PHYSIOLOGICAL_COLS)),
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])