from scipy.stats import linregress
import matplotlib.pyplot as plt
import numpy as np
# 参数设定
r_M = 0.04  # 雄性海七鳃鳗的繁殖率
d_M = 0.02  # 雄性海七鳃鳗的自然死亡率
b_M = 0.002 # 雄性海七鳃鳗的捕食效率

r_F = 0.03  # 雌性海七鳃鳗的繁殖率
d_F = 0.02  # 雌性海七鳃鳗的自然死亡率
b_F = 0.001 # 雌性海七鳃鳗的捕食效率

r_P = 0.1   # 猎物的自然增长率
a = 0.001    # 猎物被捕食的概率

# 初始条件
M0 = 78   # 雄性海七鳃鳗的初始种群数量
F0 = 22   # 雌性海七鳃鳗的初始种群数量
P0 = 300   # 猎物的初始种群数量

# 时间设定
dt = 0.05  # 时间步长
T = 100    # 总模拟时间
N = int(T / dt) # 总步数

# 初始化数组
M = np.zeros(N)
F = np.zeros(N)
P = np.zeros(N)
R = np.zeros(N)

M[0] = M0
F[0] = F0
P[0] = P0
R[0] =M0/F0;
# 模拟过程
for t in range(N - 1):
    M[t + 1] = M[t] + (r_M * M[t] - d_M * M[t] + b_M * a * P[t] * M[t]) * dt
    F[t + 1] = F[t] + (r_F * F[t] - d_F * F[t] + b_F * a * P[t] * F[t]) * dt
    P[t + 1] = P[t] + (r_P * P[t] - a * P[t] * (M[t] + F[t])) * dt
    R[t+1]=(M[t+1]/F[t+1])
# Assuming P and R are your arrays of prey population and sex ratios, respectively
slope, intercept, r_value, p_value, std_err = linregress(P, R)

# Generate points for your regression line
P_lin = np.linspace(min(P), max(P), 100)
R_lin = slope * P_lin + intercept

# Visualization
plt.figure(figsize=(10, 6))
plt.scatter(P, R, color='blue', label='Original Data')
plt.plot(P_lin, R_lin, 'r', label=f'Regression Line: y = {slope:.5f}x + {intercept:.5f}')
plt.title('Linear Regression of Sex Ratio on Prey Population')
plt.xlabel('Prey Population')
plt.ylabel('Sex Ratio (M/F)')
plt.legend()
plt.show()
