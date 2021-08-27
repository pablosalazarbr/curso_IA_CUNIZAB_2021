# -*- coding: utf-8 -*-
"""

Cursos Libres CUNIZAB -USAC-
Curso de Introduccion a la IA con Python y Tensorflow
Autor: Juan Pablo Salazar Barrios
Fecha de creacion: 19/08/2021

"""

#Importando librerias a utilizar
import sys
import os
import tensorflow as tf
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
#from tensorflow.python.keras.optimizer_v2 import Adam
from tensorflow.python.keras.optimizer_v2.adam import Adam
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dropout, Flatten, Dense, Activation
from tensorflow.python.keras.layers import Convolution2D, MaxPooling2D
from tensorflow.python.keras import backend as k

#Limpiamos la sesion almacenada en el backend de keras
k.clear_session()

#Definimos compatibilidad con la version de keras
tf.compat.v1.disable_eager_execution()
tf.compat.v1.experimental.output_all_intermediates(True)

#Asignamos variables para el almacenamiento del archivo de entrenamiento
data_entrenamiento = './data/entrenamiento'
data_validacion = './data/validacion'

#Configuramos los parametros de nuestra CNN
epocas = 20 #Configuramos la cantidad de epocas a recorrer
altura, longitud = 100, 100 #Definimos la redimension de las imagenes a procesar
#Configuracion general de la cnn
batch_size = 32
pasos = 1000
pasos_validacion = 200
filtrosConv1 = 32
filtrosConv2 = 64
tamano_filtro1 = (3,3)
tamano_filtro2 = (2,2)
tamano_pool = (2,2)
clases = 3
lr = 0.0005

#Programa de entrenamiento fase 1 (preprocesamiento de imagenes)
entrenamiento_datagen = ImageDataGenerator(
    rescale = 1/255,
    shear_range = 0.3,
    zoom_range = 0.3,
    horizontal_flip = True
    )

validacion_datagen = ImageDataGenerator(
    rescale = 1/255
    )

imagen_entrenamiento = entrenamiento_datagen.flow_from_directory(
    data_entrenamiento,
    target_size = (altura, longitud),
    batch_size = batch_size,
    class_mode = 'categorical'
    )

imagen_validacion = validacion_datagen.flow_from_directory(
    data_validacion,
    target_size = (altura, longitud),
    batch_size = batch_size,
    class_mode = 'categorical'
    )

#Generamos nuestra Red Neuronal Convolucional
cnn = Sequential() #Definimos el modelo a utilizar

cnn.add(Convolution2D(filtrosConv1, tamano_filtro1, padding = 'same', input_shape = (altura, longitud, 3), activation = 'relu'))
cnn.add(MaxPooling2D(pool_size = tamano_pool))
cnn.add(Convolution2D(filtrosConv2, tamano_filtro2, padding = 'same', activation = 'relu'))
cnn.add(MaxPooling2D(pool_size = tamano_pool))

cnn.add(Flatten())
cnn.add(Dense(256, activation = 'relu'))
cnn.add(Dropout(0.5))
cnn.add(Dense(clases, activation = 'softmax'))

cnn.compile(loss = 'categorical_crossentropy', optimizer = Adam(learning_rate = lr), metrics = ['accuracy'])

cnn.fit(imagen_entrenamiento, steps_per_epoch = pasos, epochs = epocas, validation_data = imagen_validacion, validation_steps = pasos_validacion)

dir = './modelo/'

if not os.path.exists(dir):
    os.mkdir(dir)
    cnn.save('./modelo/modelo.h5')
    cnn.save_weights('./modelo/pesos.h5')
