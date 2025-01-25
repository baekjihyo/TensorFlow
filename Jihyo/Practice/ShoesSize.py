import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress all logs except errors

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

height = np.array( [200, 200, 200,  
                    190, 190, 190,
                    180, 180, 180,
                    170, 170, 170,
                    160, 160, 160, 
                    150, 150, 150] )
shoes = np.array( [260, 255, 257,
                   250, 230, 260,
                   245, 250, 240,
                   240, 260, 255,
                   235, 240, 210,
                   230, 210, 207] )

# Weights(random number)
w1 = tf.Variable(1.0)
w2 = tf.Variable(100.0)

def GradientDescent():
    prediction = height * w1 + w2
    return tf.square(shoes - prediction)


# learning rate optimizer
opt = tf.keras.optimizers.Adam(learning_rate = 0.1)

# Gradient Descent method + checking if it really works
w1Values, w2Values = [], []

for i in range(1000):
    opt.minimize(GradientDescent, var_list = [w1, w2])
    if i % 200 == 0:
        print(f"w1 = {w1.numpy()}, w2 = {w2.numpy()}")

plt.plot(height, shoes, 'o')
plt.plot(height, (height * w1 + w2))
plt.show()