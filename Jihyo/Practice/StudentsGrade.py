# 1. Import Libraries

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress all logs except errors

import tensorflow as tf
import numpy as np
import matplotlib as plt
import pandas as pd


# 2. Process Data

data = pd.read_csv('c:/Users/baekj/Documents/GitHub/TensorFlow/Jihyo/DataSets/StudentsPerformance.csv')
data = data.dropna()

xData = []
yData = []

for i, rows in data.iterrows():
    if rows['test preparation course'] == "none":
        yData.append(0)
    else:
        yData.append(1)

for i, rows in data.iterrows():
    xData.append([ rows["math score"], rows["reading score"], rows["writing score"] ])


# 3. Build a Model

model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(64, activation = 'relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(32, activation = 'relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])


# 4. Train the model

model.fit(xData, yData, epochs = 10)


# 5. Predict

prediction = model.predict([ [100, 100, 100], [0, 0, 0] ])
print(prediction)