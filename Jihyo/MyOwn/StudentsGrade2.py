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

yData = (data['writing score'].values) / 100

for i, rows in data.iterrows():
    xData.append([ rows["math score"], rows["reading score"] ])


# 3. Build a Model // 여기서부터는 GPT가 추천해 주는 걸 내가 조금 수정했어(layer, nodes(neurons) and activation function)

model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(64, activation='relu', input_shape=(2,)),  # First hidden layer
    tf.keras.layers.Dropout(0.2),  # Dropout to prevent overfitting
    tf.keras.layers.Dense(32, activation='relu'),  # Second hidden layer
    tf.keras.layers.Dropout(0.2),  # Dropout to prevent overfitting
    tf.keras.layers.Dense(1, activation = 'sigmoid')  # Output layer (no activation for regression) // sigmoid는 넣지 말라고 했는데 prediction이 이상하길래 그냥 넣었어
])

model.compile(optimizer='adam', loss='mse', metrics=['mae'])  # Use MSE for regression // MSE가 뭔지 모르겠어.


# 4. Train the Model

model.fit(np.array(xData), np.array(yData), epochs=10, batch_size=32)

# 5. Predict

predictions = model.predict([[100, 100], [0, 0]])
print(predictions * 100)