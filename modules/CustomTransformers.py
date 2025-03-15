import pandas as pd
import numpy as np
# We are using this transformer in the app.py file


from sklearn.base import BaseEstimator, TransformerMixin
class CategoricalBinaryTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, name_of_features=[], feature_value=[]):  
        self.name_of_features = name_of_features
        self.feature_value = feature_value

    def fit(self, X, y=None):
        return self  

    def transform(self, X):
        Xcopy = X.copy()  # Ensure X is copied correctly
        
        if not self.name_of_features or not self.feature_value:
            raise ValueError("No features and values provided for transformation")  

        if len(self.name_of_features) != len(self.feature_value):
            raise ValueError("Mismatch: 'name_of_features' and 'feature_value' must have the same length")

        for i, valueFeat in zip(self.name_of_features, self.feature_value):
            if i not in Xcopy.columns:
                raise ValueError(f"Feature '{i}' not found in DataFrame columns")
            Xcopy[i] = (Xcopy[i] == valueFeat).astype(int)

        return Xcopy


# Customer transformer to add the bmi_smoker feature

class CombinedAttributesAdder(BaseEstimator, TransformerMixin):
    def __init__(self): # no *args or **kargs
        pass
    
    def fit(self, X, y=None):
        return self  # nothing else to do
    
    def transform(self, X):
        Xcopy = X.copy()
        
        # Check if 'smoker' column exists
        if 'smoker' not in Xcopy.columns:
            raise ValueError("The input DataFrame must contain a 'smoker' column")
        
        # Ensure 'smoker' column is converted to float
        try:
            Xcopy["smoker"] = Xcopy["smoker"].astype(float)
        except ValueError:
            raise ValueError("The 'smoker' column must be convertible to float")
        
        # Print unique values for debugging
        print(Xcopy["smoker"].unique())
        
        # Check if 'bmi' column exists
        if 'bmi' not in Xcopy.columns:
            raise ValueError("The input DataFrame must contain a 'bmi' column")
        
        # Add the 'bmi_smoker' feature
        Xcopy["bmi_smoker"] = Xcopy["bmi"] * Xcopy["smoker"]
        
        return Xcopy
    

# ND transform array

class NDArrayToDataFrameTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, column_names):
        self.column_names = column_names

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if not isinstance(X, np.ndarray):
            raise ValueError("Input must be a NumPy ndarray")
        return pd.DataFrame(X, columns=self.column_names)
