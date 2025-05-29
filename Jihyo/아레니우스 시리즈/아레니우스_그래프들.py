import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# 데이터 불러오기
path = r"C:\Users\baekj\Documents\Github\Tensorflow\DataSets\아레니우스.csv"
data = pd.read_csv(path).dropna()

# 절대 온도 (K), ln(σ) 계산
data["T (K)"] = data["온도 (℃)"] + 273.15
data["ln_sigma"] = np.log(data["이온 전도도 (S/cm)"])

# x = 1/T, y = ln(σ)
x = 1 / data["T (K)"].to_numpy().reshape(-1, 1)
y = data["ln_sigma"].to_numpy().reshape(-1, 1)

# 선형 회귀 (Arrhenius 선형 모델 피팅)
reg = LinearRegression()
reg.fit(x, y)

# 계수 계산
slope = reg.coef_[0][0]          # -Ea / k
intercept = reg.intercept_[0]    # ln(σ₀)
Ea = -slope * 8.617e-5           # eV

print(f"ln(σ) = {intercept:.4f} - ({-slope:.4f}) * (1/T)")
print(f"→ Ea ≈ {Ea:.4f} eV")

# 회귀 예측 값
x_sorted = np.sort(1 / data["T (K)"])
ln_sigma_pred = reg.predict(x_sorted.reshape(-1, 1))

# 시각화 (x축: T(K))
plt.figure(figsize=(8, 6))
plt.plot(data["T (K)"], data["ln_sigma"], 'bo', label='Actual ln(σ)', markersize=5)
plt.plot(1 / x_sorted, ln_sigma_pred, 'r-', linewidth=2, label='Arrhenius Fit')
plt.xlabel('T (K)')
plt.ylabel('ln(σ)')
plt.grid(True)
plt.legend()
plt.title('Arrhenius Fit (Linear Regression)')
plt.show()
