from flask import Flask, request, render_template
import numpy as np
import pandas as pd
import joblib
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.base import BaseEstimator, TransformerMixin

app = Flask(__name__)

# Load the pre-trained model
model = joblib.load('forest_reg_model.pkl')

class CategoricalBinaryTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, name_of_features, feature_value):
        self.name_of_features = name_of_features
        self.feature_value = feature_value
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        Xcopy = X.copy()
        for feature, value in zip(self.name_of_features, self.feature_value):
            Xcopy[feature] = (Xcopy[feature] == value).astype(float)
        return Xcopy

class CombinedAttributesAdder(BaseEstimator, TransformerMixin):
    def __init__(self): # no *args or **kargs
        pass
    
    def fit(self, X, y=None):
        return self  # nothing else to do
    
    def transform(self, X):
        Xcopy = X.copy()
        if 'smoker' not in Xcopy.columns:
            raise ValueError("The input DataFrame must contain a 'smoker' column")
        try:
            Xcopy["smoker"] = Xcopy["smoker"].astype(float)
        except ValueError:
            raise ValueError("The 'smoker' column must be convertible to float")
        if 'bmi' not in Xcopy.columns:
            raise ValueError("The input DataFrame must contain a 'bmi' column")
        Xcopy["bmi_smoker"] = Xcopy["bmi"] * Xcopy["smoker"]
        return Xcopy

num_pipeline = Pipeline([
    ('BinaryTransformer_addr', CategoricalBinaryTransformer(name_of_features=["sex", "smoker"], feature_value=["female", "yes"])),
    ('attr_adder', CombinedAttributesAdder()),
    ('std_scaler', StandardScaler())
])

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        age = float(request.form['age'])
        sex = request.form['sex']
        bmi = float(request.form['bmi'])
        children = int(request.form['children'])
        smoker = request.form['smoker']
        region = request.form['region']
        
        input_data = pd.DataFrame([[age, sex, bmi, children, smoker, region]], 
                                  columns=['age', 'sex', 'bmi', 'children', 'smoker', 'region'])
        
        prepared_data = num_pipeline.fit_transform(input_data)
        model = joblib.load('forest_reg_model.pkl')
        prediction = model.predict(prepared_data)
        
        return render_template('index.html', prediction=prediction[0])
    
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)