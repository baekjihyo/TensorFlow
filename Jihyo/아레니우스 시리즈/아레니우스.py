path = r"C:\Users\baekj\Documents\Github\Tensorflow\DataSets\아레니우스.csv"

# 1. 불러오기
import pandas as pd
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# 2. 데이터 처리
data = pd.read_csv(path).dropna()
data["T (K)"] = data["온도 (℃)"] + 273.15
data["ln_sigma"] = np.log(data["이온 전도도 (S/cm)"])

# Convert to numpy arrays
xData = 1 / data["T (K)"].to_numpy().reshape(-1, 1)
yData = data["ln_sigma"].to_numpy().reshape(-1, 1)

# Scale the data
from sklearn.preprocessing import StandardScaler

x_scaler = StandardScaler()
y_scaler = StandardScaler()

x_scaled = x_scaler.fit_transform(xData)
y_scaled = y_scaler.fit_transform(yData)

# 3. 모델 만들기
model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(1)
])

optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)

model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])

# 4. 모델 학습시키기
model.fit(x_scaled, y_scaled, epochs=500)

# 5. 모델 테스트하기
y_pred_scaled = model.predict(x_scaled)
y_pred = y_scaler.inverse_transform(y_pred_scaled)

# 데이터 처리하기
T_K = data["T (K)"].to_numpy().reshape(-1, 1)
sort_idx = np.argsort(xData.flatten())

T_sorted = T_K[sort_idx]
yData_sorted = yData[sort_idx]
y_pred_sorted = y_pred[sort_idx]

x_line = np.linspace(x_scaled.min(), x_scaled.max(), 200).reshape(-1, 1)
y_line_scaled = model.predict(x_line)
y_line = y_scaler.inverse_transform(y_line_scaled)
x_line_orig = x_scaler.inverse_transform(x_line)

T_line = 1 / x_line_orig

# 6. 그래프로 나타내기
plt.figure(figsize=(8, 6))
plt.plot(T_sorted, yData_sorted, 'bo')
plt.plot(T_line, y_line, color='red', linestyle='-', linewidth=2)
plt.xlabel('T (K)')
plt.ylabel('ln(σ)')
plt.legend()
plt.grid(True)
plt.title('Arrhenius Prediction with Deep Learning')
plt.show()