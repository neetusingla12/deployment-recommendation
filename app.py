# -*- coding: utf-8 -*-
"""
Created on Sat Oct 24 11:36:26 2020

@author: Neetu
"""

import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__)
model = pickle.load(open('model1.pkl', 'rb'))

def home():
    return render_template('index.html',prediction_text="News recommended are:", output)



#@app.route('/predict',methods=['POST'])
#def predict():
#    '''
#    For rendering results on HTML GUI
#    '''
#   # int_features = [int(x) for x in request.form.values()]
#    #final_features = [np.array(int_features)]
#    #prediction = model.predict(final_features)
#
#    #output = round(prediction[0], 2)
#    output = model
#
#    return render_template('index.html', prediction_text="News recommended are:", output)



if __name__ == "__main__":
    app.run(debug=True)
