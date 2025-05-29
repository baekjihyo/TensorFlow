import numpy as np
import matplotlib.pyplot as plt

# 아레니우스 방정식의 상수 설정
A = 1e13               # 전지수 인자 (s⁻¹)
Ea = 75e3              # 활성화 에너지 (J/mol)
R = 8.314              # 기체 상수 (J/mol·K)

highEa = 75e3 + 20e3           # 활성화 에너지 (J/mol)
lowEa = 75e3 - 20e3            # 활성화 에너지 (J/mol)

# 온도 범위 설정 (Kelvin)
T = np.linspace(100, 2000, 100)

# 1. 지수 형태: k = A * exp(-Ea / RT)
k = A * np.exp(-Ea / (R * T))
highk = A * np.exp(-highEa / (R * T))
lowk = A * np.exp(-lowEa / (R * T))

# 2. 로그 형태: ln(k) vs 1/T
lnk = np.log(k)
highlnk = np.log(highk)
lowlnk = np.log(lowk)
invT = 1 / T           # todo

# 지수 
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.plot(T, k, color='violet', linewidth=2)
plt.plot(T, highk, color='red', linewidth=2)
plt.plot(T, lowk, color='blue', linewidth=2)
plt.title('Arrhenius Plot (Exponential Form)')
plt.xlabel('Temperature (K)')
plt.ylabel('Rate Constant k (1/s)')
plt.grid(True)

# 로그 
plt.subplot(1, 2, 2)
plt.plot(T, lnk, color='violet', linewidth=2)
plt.plot(T, highlnk, color='red', linewidth=2)
plt.plot(T, lowlnk, color='blue', linewidth=2)
plt.title('Arrhenius Plot (Log Form)')
plt.xlabel('1 / Temperature (1/K)')
plt.ylabel('ln(k)')
plt.grid(True)

plt.tight_layout()
plt.show()
