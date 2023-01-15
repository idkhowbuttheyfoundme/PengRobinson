from main import PengRobinson
import numpy as np
import matplotlib.pyplot as plt

Tc = 706.9
Pc = 58.5 / 10  # bar to MPa
W = 0.35
M = 78.13

fluid = PengRobinson(Tc, Pc, W)

# density at p = 0.1Mpa
P = np.array([0.1])
density = []
T = list(np.arange(293, 364))
for t in T:
    density.append(max(list(fluid.density_calc(P, t))))
print(density)
data_agieienko = np.array(
    # T,k ρ, kg/m^3
    [[308.15, 1085.22],
     [313.15, 1080.2],
     [318.15, 1075.18],
     [323.15, 1070.16],
     [328.15, 1065.14],
     [333.15, 1060.12],
     [338.15, 1055.1],
     [343.15, 1050.07],
     [348.15, 1045.04],
     [353.15, 1040],
     [358.15, 1034.95],
     [363.15, 1029.9]])
data_yue = np.array([[293.15, 1.1004],
                     [298.15, 1.0956],
                     [303.15, 1.0905],
                     [308.15, 1.0854],
                     [313.15, 1.0804],
                     [318.15, 1.0759]])
data_agieienko[:, 1] = data_agieienko[:, 1] / M
data_yue[:, 1] = data_yue[:, 1] * 1000 / M
plt.style.use('seaborn')
fig, ax = plt.subplots()
ax.plot(data_agieienko[:, 0], data_agieienko[:, 1], '*', label='agienko-2020')
ax.plot(data_yue[:, 0], data_yue[:, 1], 'o', label='yue-2018')
ax.plot(T, density, color='k', label='comp')

ax.set(xlabel='T, K', ylabel='ρ, mol\l')
ax.legend()
plt.legend(frameon=True)
# pressure
n = 100
# 1st part
T_0_1 = np.linspace(464, 447, num=n)
P_0_1 = np.linspace(0.099, 0.064, num=n)
root1 = fluid.pressure_calc_roots(P_0_1, T_0_1)

# 2nd part
T_0_2 = np.linspace(447, 406, num=n)
P_0_2 = np.linspace(0.098, 0.018, num=n)
root2 = fluid.pressure_calc_roots(P_0_2, T_0_2)

# 3rd part
T_0_3 = np.linspace(406, 356, num=n)
P_0_3 = np.linspace(0.018, 0.002, num=n)
root3 = fluid.pressure_calc_roots(P_0_3, T_0_3)

# 4th part
T_0_4 = np.linspace(356, 330, num=n)
P_0_4 = np.linspace(0.002, 0.00067, num=n)
root4 = fluid.pressure_calc_roots(P_0_4, T_0_4)

# 5th part
T_0_5 = np.linspace(330, 300, num=n)
P_0_5 = np.linspace(0.00067, 0.00008, num=n)
root5 = fluid.pressure_calc_roots(P_0_5, T_0_5)

T_sat = np.concatenate((T_0_1, T_0_2, T_0_3, T_0_4, T_0_5))
p_sat = np.concatenate((root1, root2, root3, root4, root5))

data_tochigi = np.array([
    # T,K   p,kPa
    [462.52, 97.09],
    [459.81, 90.5],
    [456.92, 83.91],
    [453.87, 77.32],
    [450.6, 70.72],
    [447.03, 64.13],
    [443.15, 57.52],
    [439.41, 51.59],
    [434.67, 45.11],
    [429.55, 38.42],
    [423.39, 31.78],
    [416.11, 25.15],
    [415.54, 24.8],
    [406.26, 18.16],
    [393.51, 11.5],
    [386.51, 8.81],
    [377.58, 6.17],
    [365.19, 3.65],
    [363.15, 3.28],
    [356.23, 2.44]])
data_tochigi[:, 1] = data_tochigi[:, 1] / 1000
data_nishimura = np.array(
    # T,C p,mmHg
    [[27.1, 0.67],
     [31.8, 0.97],
     [38.8, 1.5],
     [46.9, 2.78],
     [47.7, 2.85],
     [50, 3.21],
     [54, 4.14],
     [55.6, 4.39],
     [56.9, 4.94],
     [58, 5.35],
     [61.1, 6.12],
     [70.9, 10.19],
     [77, 13.74],
     [86.1, 21.32],
     [93.9, 30.08],
     [101.9, 42.19],
     [107.7, 53.41],
     [113.4, 66.71],
     [117.5, 77.89],
     [121.8, 91.16],
     [125, 102.7],
     [130, 122.4],
     [134.8, 146.3],
     [140, 174.2],
     [144.8, 204.7],
     [150.2, 239.5],
     ])
data_nishimura[:, 0] = data_nishimura[:, 0] + 273.15
data_nishimura[:, 1] = data_nishimura[:, 1] * 0.000133322
plt.style.use('seaborn')
fig, ax = plt.subplots()
ax.plot(data_tochigi[:, 0], data_tochigi[:, 1], 'o', label='tochigi-1999')
ax.plot(data_nishimura[:, 0], data_nishimura[:, 1], '*', label='nishimura-1972')
ax.plot(T_sat, p_sat, color='k', label='comp')
ax.set(xlabel='T, K', ylabel='p, MPa')
ax.legend()
plt.legend(frameon=True)
plt.show()
