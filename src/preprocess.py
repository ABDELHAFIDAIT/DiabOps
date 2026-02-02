import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.impute import KNNImputer


def preprocess(source: str) -> pd.DataFrame :
    data = pd.read_csv(source)
    
    if "Unnamed: 0" in data.columns:
        data = data.drop(columns=["Unnamed: 0"])
    
    cols_with_zeros = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']

    data[cols_with_zeros] = data[cols_with_zeros].replace(0, np.nan)
    
    imputer = KNNImputer(n_neighbors=5)
    data[cols_with_zeros] = imputer.fit_transform(data[cols_with_zeros])
    
    return data



def scaling(data: pd.DataFrame) :
    scaler = StandardScaler()
    
    numeric_cols = data.select_dtypes(include=[np.number]).columns
    
    data[numeric_cols] = scaler.fit_transform(data[numeric_cols])
    
    return data, scaler