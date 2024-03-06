from typing import List
import sys
import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.base import BaseEstimator, TransformerMixin


class ColumnScaler(BaseEstimator, TransformerMixin):
    """
    Scale specified columns in a DataFrame using StandardScaler.
    """

    def __init__(self, columns_to_scale):
        # Ensure columns_to_scale is always a list
        if isinstance(columns_to_scale, str):
            columns_to_scale = [columns_to_scale]
        self.columns_to_scale = columns_to_scale
        self.scaler = StandardScaler()

    def fit(self, X: pd.DataFrame, y: pd.Series = None):
        self.scaler.fit(X[self.columns_to_scale])
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        X[self.columns_to_scale] = self.scaler.transform(X[self.columns_to_scale])
        return X



class OutlierHandler(BaseEstimator, TransformerMixin):
    """
    Change the outlier values: 
        - to upper-bound, if the value is higher than upper-bound, or
        - to lower-bound, if the value is lower than lower-bound respectively.
    """

    def __init__(self, variable:str):

        if not isinstance(variable, str):
            raise ValueError("variable name should be a string")

        self.variable = variable

    def fit(self, X: pd.DataFrame, y: pd.Series = None):
        # we need the fit statement to accomodate the sklearn pipeline
        X = X.copy()
        q1 = X.describe()[self.variable].loc['25%']
        q3 = X.describe()[self.variable].loc['75%']
        iqr = q3 - q1
        self.lower_bound = q1 - (1.5 * iqr)
        self.upper_bound = q3 + (1.5 * iqr)
        
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        
        for i in X.index:
            if X.loc[i, self.variable] > self.upper_bound:
                X.loc[i, self.variable]= self.upper_bound
            if X.loc[i, self.variable] < self.lower_bound:
                X.loc[i, self.variable]= self.lower_bound

        return X
