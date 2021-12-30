# -*- coding: utf-8 -*-
"""
Created on Thu Dec 30 00:49:22 2021

@author: jerf
"""
# Import all project dependencies
import os
import requests
import numpy as np
import tensorflow as tf

from flask import Flask, request, jsonify
from imageio import imread

print(tf.__version__)

# Load the pretrained model
with open('fashion_model_flask.json', 'r') as f:
    model_json = f.read()

model = tf.keras.models.model_from_json(model_json)

# Creating the Flask API
# Crear una aplicacion de Flask
app = Flask(__name__)

# Definir la funcion de clasificacion de imagenes
@app.route('/api/v1/<string:img_name>', methods = ['POST'])
def classify_image(img_name):
    # definir la ruta de la carpeta de imagenes
    upload_dir = 'D://jose/cursos/cursos udemy/deep learning projects/tensorflow/Flask-API/uploads/'
    # cargar una de las imagenes de la carpeta
    image = imread(upload_dir + img_name)
    
    # definir la lista de posibles clases de la imagen
    classes = ['Camiseta', 'Pantalon', 'Sudadera', 'Vestido', 'Abrigo', 'Sandalia', 'Jersey', 'Zapatilla', 'Bolsa', 'Botas']
    
    # Hacer la prediccion utilizando el modelo pre entrenado
    prediction = model.predict([image.reshape(1, 28 * 28)])
    print(prediction)
    # Devolver la prediccion al usuario
    return jsonify({
        'object_identified': classes[np.argmax(prediction[0])]
        })

# Iniciar la aplicacion Flask
app.run(port = 8000, debug = False)