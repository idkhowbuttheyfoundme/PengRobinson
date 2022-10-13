from main import PengRobinson
import numpy as np
import matplotlib.pyplot as plt

Tc = 513.92
Pc = 61.48 / 10  # bar to MPa
W = 0.649
M = 46.069

fluid = PengRobinson(Tc, Pc, W)

# density
T = 300
P = np.arange(1.0, 50, 0.5)
density_300 = fluid.density_calc(P, T)

T = 520
P_1 = np.arange(1.0, 50, 0.5)
density_520 = fluid.density_calc(P_1, T)

data = np.array([[784.60, 10],
                 [785.50, 20],
                 [788.30, 50],
                 [790.40, 75],
                 [792.51, 100],
                 [796.60, 150],
                 [800.40, 200],
                 [804.1, 250],
                 [807.6, 300],
                 [814.4, 400],
                 [820.7, 500]])
data_1 = np.array([[397.2, 75],
                   [465.6, 100],
                   [515.9, 150],
                   [544.5, 200],
                   [565.2, 250],
                   [581.6, 300],
                   [607.3, 400],
                   [627.4, 500]])

plt.style.use('seaborn-whitegrid')
fig, ax = plt.subplots()
data[:, 0] = data[:, 0] / M
data[:, 1] = data[:, 1] / 10
ax.plot(data[:, 1], data[:, 0], 'o', color='g', label='golubev-1980')
data_1[:, 0] = data_1[:, 0] / M
data_1[:, 1] = data_1[:, 1] / 10
ax.plot(data_1[:, 1], data_1[:, 0], '*', color='b', label='golubev-1980')
ax.plot(P, density_300, label='T = 300K', color='g')
ax.plot(P_1, density_520, label='T = 520K', color='b')
ax.set(xlabel='p, MPa', ylabel='ρ, mol\l')
ax.legend()
plt.legend(frameon=True)

# pressure

n = 100

# 1st part
T_0_1 = np.linspace(318, 350, num=n)
P_0_1 = np.linspace(0.020, 0.095, num=n)
root1 = fluid.pressure_calc(P_0_1, T_0_1)

# 2nd part
T_0_2 = np.linspace(350, 425, num=n)
P_0_2 = np.linspace(0.095, 1.1, num=n)
root2 = fluid.pressure_calc(P_0_2, T_0_2)

# 3rd part
T_0_3 = np.linspace(425, 475, num=n)
P_0_3 = np.linspace(1.1, 3.1, num=n)
root3 = fluid.pressure_calc(P_0_3, T_0_3)

# 4th part
T_0_4 = np.linspace(475, 508, num=100)
P_0_4 = np.linspace(3.1, 5.55, num=100)
root4 = fluid.pressure_calc(P_0_4, T_0_4)

# 5th part
T_0_5 = np.linspace(508, 516, num=100)
P_0_5 = np.linspace(5.55, 6.38, num=100)
root5 = fluid.pressure_calc(P_0_5, T_0_5)

T_sat = np.concatenate((T_0_1, T_0_2, T_0_3, T_0_4, T_0_5))
p_sat = np.concatenate((root1, root2, root3, root4, root5))

data_mousa = np.array([
    # T,k p,Kpa
    [318.7, 20],
    [322.8, 25],
    [326.5, 30],
    [329.2, 35],
    [331.6, 40],
    [334.3, 45],
    [336.3, 50],
    [338.1, 55],
    [340, 60],
    [341.8, 65],
    [343.3, 70],
    [344.4, 75],
    [346.1, 80],
    [347.6, 85],
    [348.7, 90],
    [350, 95],
    [351, 100],
    [417.4, 911],
    [442.3, 1621],
    [448.9, 1741],
    [456.4, 2160],
    [463.7, 2465],
    [469.3, 2714],
    [475.4, 3038],
    [481.3, 3380],
    [486.7, 3740],
    [491.8, 4067],
    [496.5, 4400],
    [497, 4420],
    [501.7, 4732],
    [505.5, 5006],
    [510.5, 5373],
    [516.2, 6325],
])
data_mousa[:, 1] /= 1000
data_gomez = np.array([
    # T,K p, KPa
    [351.47, 101.33],
    [358.986, 135.52],
    [362.756, 155.82],
    [366.631, 179.32],
    [379.1, 275.24],
    [398.47, 501.14],
    [402.89, 570.6],
    [408.53, 666.92],
    [418.68, 873.15],
    [428.88, 1129.3],
    [448.38, 1773.6],
    [468.8, 2715.2],
    [477.94, 3243.3],
    [487.81, 3894.9],
    [498.32, 4697],
    [503.21, 5114.5],
    [508.01, 5552.4],
    [508.67, 5615.5],
    [511.08, 5846.9],
    [513.92, 6148.4],
    [516.26, 6379.4],
])
data_gomez[:, 1] /= 1000
plt.style.use('seaborn-whitegrid')
fig, ax = plt.subplots()
ax.plot(data_mousa[:, 0], data_mousa[:, 1], '*', label='mousa-1987')
ax.plot(data_gomez[:, 0], data_gomez[:, 1], 'o', label='gomez-nieto-1976')
ax.plot(T_sat, p_sat, color='k')
ax.set(xlabel='T, K', ylabel='p, MPa')
ax.legend()
plt.legend(frameon=True)
plt.show()
