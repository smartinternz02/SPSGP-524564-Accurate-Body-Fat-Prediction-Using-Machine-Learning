# -*- coding: utf-8 -*-
"""
Created on Mon Jul  3 21:26:06 2023

@author: HOME
"""

import numpy as np
from flask import Flask, request, jsonify,render_template

#create flask app
app=Flask(__name__)

#load joblib model
import joblib
model=joblib.load(open('random_forest_model.pkl','rb'))

@app.route('/')
def Home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    float_features=[float(x) for x in request.form.values()]
    features=[np.array(float_features)]
    prediction=model.predict(features)
    
    return render_template('index.html',prediction_text="Body Fat of Person is {}".format(prediction))

if __name__=="__main__":
    app.run(debug=True)