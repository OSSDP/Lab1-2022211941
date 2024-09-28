import numpy as np
import matplotlib.pyplot as plt

# 参数设定
r_M = 0.04  # 雄性海七鳃鳗的繁殖率
d_M = 0.02  # 雄性海七鳃鳗的自然死亡率
b_M = 0.002 # 雄性海七鳃鳗的捕食效率

r_F = 0.03  # 雌性海七鳃鳗的繁殖率
d_F = 0.02  # 雌性海七鳃鳗的自然死亡率
b_F = 0.001 # 雌性海七鳃鳗的捕食效率

r_P = 0.1   # 猎物的自然增长率
a = 0.001    # 猎物被捕食的概率

d_C = 0.03   # 寄生虫的自然死亡率
i_M = 0.005  # 寄生虫增加的七鳃鳗雄性死亡率
i_F = 0.005  # 寄生虫增加的七鳃鳗雌性死亡率
i_P = 0.002  # 寄生虫增加的猎物死亡率
C0 = 50     # 寄生虫的初始种群数量
r_C_base = 0.02  # 调整后的寄生虫自然增长率

# 初始条件
M0 = 56   # 雄性海七鳃鳗的初始种群数量
F0 = 44   # 雌性海七鳃鳗的初始种群数量
P0 = 300  # 猎物的初始种群数量

# 时间设定
dt = 0.05  # 时间步长
T = 100    # 总模拟时间
N = int(T / dt) # 总步数

# 初始化数组
M = np.zeros(N)
F = np.zeros(N)
P = np.zeros(N)
C = np.zeros(N)

M[0] = M0
F[0] = F0
P[0] = P0
C[0] = C0

# 模拟过程
for t in range(N - 1):
    M[t + 1] = M[t] + (r_M * M[t] - d_M * M[t] - i_M * C[t]/C[0] * M[t]+ b_M * a * P[t] * M[t]) * dt
    F[t + 1] = F[t] + (r_F * F[t] - d_F * F[t] - i_F * C[t]/C[0] * F[t] + b_F * a * P[t] * F[t]) * dt
    P[t + 1] = P[t] + (r_P * P[t] - i_P * P[t]*C[t]/C[0]-a*P[t]*(M[t]+F[t])) * dt

    r_C = r_C_base + 0.015 * M[t]/(F[t]+M[t])
    C[t + 1] = C[t] + (r_C * C[t] - d_C * C[t]) * dt

# 绘图
plt.figure(figsize=(18, 6))

plt.subplot(1, 3, 1)
plt.plot(np.linspace(0, T, N), M, label='Male Sea Lampreys', color='blue')
plt.plot(np.linspace(0, T, N), F, label='Female Sea Lampreys', color='red')
plt.plot(np.linspace(0, T, N), P, label='Prey', color='green')
plt.xlabel('Time')
plt.ylabel('Population')
plt.title('Population Dynamics with  Parasite Growth Rate')
plt.legend()

plt.subplot(1, 3, 2)
plt.plot(np.linspace(0, T, N), M / (F+M), label='Sex Ratio', color='green')
plt.xlabel('Time')
plt.ylabel('Sex Ratio')
plt.title('Sex Ratio Dynamics')
plt.legend()

plt.subplot(1, 3, 3)
plt.plot(np.linspace(0, T, N), C, label='Parasites', color='purple')
plt.xlabel('Time')
plt.ylabel('Parasite Population')
plt.title('Parasite Dynamics')
plt.legend()

plt.tight_layout()
plt.show()
