from flask import Flask,request
from flask_cors import  CORS, cross_origin
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow import keras
import os
# from tensorflow import keras
# from keras.models import load_model


app = Flask(__name__)
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'


# Load Models
# heart_model =  tf.keras.models.load_model('heartd_prediction_model.h5')
heart_model = keras.models.load_model("heartd_prediction_model.h5")
# tf.lite.TFLiteConverter.from_keras_model('heartd_prediction_model.h5')
# load_model("heartd_prediction_model.h5")
# heart_model= load_model(heart_model)

# diab_model =  tf.keras.models.load_model('diabetes_prediction_model.h5')
diab_model = keras.models.load_model("diabetes_prediction_model.h5")
# tf.lite.TFLiteConverter.from_keras_model('diabetes_prediction_model.h5')
# load_model("diabetes_prediction_model.h5")




# Scaler
scaler = MinMaxScaler()


@app.route('/')
@cross_origin()
def register():
    return "Hello world"

@app.route('/Diabetes', methods=['GET'])
def predDiabetes():
    if request.method == 'GET':
        age = int(request.args.get('age'))
        alopecia = int(request.args.get('alopecia'))
        polyuria = int(request.args.get('polyuria'))
        polydipsia = int(request.args.get('polydipsia'))
        gender = int(request.args.get('sex'))
        itching = int(request.args.get('itching'))
        delayed_healing = int(request.args.get('delayed'))
        irritability = int(request.args.get('irritab'))
        dfd = pd.DataFrame(np.array([[age, alopecia, polyuria, polydipsia, gender, itching, delayed_healing, irritability]]), 
        columns=['Age','Alopecia', 'Polyuria','Polydibsia','Gender', 'Itching', 'delayed healing','Irritability'])
        dfd[['Age']] = scaler.fit_transform(dfd[['Age']])
        predd = diab_model.predict(dfd)
    return predd


@app.route('/Heart', methods=['GET'])
def predHeart():
    if request.method == 'GET':
        age = request.args.get('age')
        thalach = int(request.args.get('thalach'))
        oldpeak = float(request.args.get('oldpeak'))
        ca = int(request.args.get('ca'))
        chol = int(request.args.get('chol'))
        trestpbs = int(request.args.get('trestbps'))
        cp = int(request.args.get('cp'))
        gender = request.args.get('sex')
        dfh = pd.DataFrame(np.array([[age, thalach, oldpeak, ca, chol, trestpbs, cp, gender]]), 
        columns=['age', 'thalach', 'oldpeak','ca','chol', 'trestbps','cp','sex'])
        dfh = scaler.fit_transform(dfh)
        predh = heart_model.predict(dfh)
    return predh


if __name__ == '__main__':
    app.run(debug=True)