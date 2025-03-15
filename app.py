from flask import Flask, request, render_template
import numpy as np
import pandas as pd
import joblib
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.base import BaseEstimator, TransformerMixin
from modules.CustomTransformers import CategoricalBinaryTransformer,CombinedAttributesAdder,NDArrayToDataFrameTransformer
app = Flask(__name__)

# Load the pre-trained model
model = joblib.load('forest_reg_model.pkl')

# Loading the pre-trained pipeline
transFull_pipeline = joblib.load('full_pipeline_medicalInsTransformData.pkl')

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
        
        prepared_data = transFull_pipeline.transform(input_data)
        prediction = model.predict(prepared_data)
        
        return render_template('index.html', prediction=prediction[0])
    
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)