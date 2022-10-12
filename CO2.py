from main import PengRobinson
import numpy as np
import matplotlib.pyplot as plt

Tc = 304.12
Pc = 73.74 / 10  # bar to MPa
W = 0.225
M = 44.01

fluid = PengRobinson(Tc, Pc, W)

# density_300 at T = 300

T = 300
P = np.arange(0.5, 30.5, 0.5)
density_300 = fluid.density_calc(P, T)

# density_300 at T = 360

T = 360
P_1 = np.arange(0.5, 30.1, 0.5)
density_360 = fluid.density_calc(P_1, T)

# plotting density
data = np.array([[9.1205, 0.50403],
                 [9.1214, 0.50408],
                 [28.7774, 1.50497],
                 [39.959, 2.0242],
                 [64.3553, 3.03772],
                 [92.5681, 4.01902],
                 [131.549, 5.07342],
                 [178.921, 5.9519],
                 [753.954, 8.02334],
                 [802.815, 10.0657],
                 [833.144, 12.0505],
                 [891.667, 18.0433],
                 [930.344, 24.0587],
                 [959.843, 30.0501]])
data_1 = np.array([
    [177.676, 8.9378],
    [237.408, 10.9306],
    [273.475, 11.993],
    [302.223, 12.787],
    [375.618, 14.7057],
    [479.295, 17.5408],
    [556.172, 20.1844],
    [556.263, 20.188],
    [647.965, 24.764],
    [686.03, 27.4211],
    [717.383, 30.0864]])

plt.style.use('seaborn-whitegrid')
fig, ax = plt.subplots()
data[:, 0] = data[:, 0] / M
data_1[:, 0] = data_1[:, 0] / M
ax.plot(data[:, 1], data[:, 0], '*', color='b', label='klimeck-2001')
ax.plot(P, density_300, label='T = 300K', color='b')

ax.plot(data_1[:, 1], data_1[:, 0], 'o', color='g', label='klimeck-2001')
ax.plot(P_1, density_360, label='T = 360K', color='g')

ax.set(xlabel='p, MPa', ylabel='ρ, mol\l')
ax.legend()
plt.legend(frameon=True)


# computing pressure
n = 100

# 1st part
T_0_1 = np.linspace(194, 280, num=n)
P_0_1 = np.linspace(0.097, 4.1, num=n)
root1 = fluid.pressure_calc(P_0_1, T_0_1)

# 2nd part
T_0_2 = np.linspace(280, 300, num=n)
P_0_2 = np.linspace(4.1, 6.7, num=n)
root2 = fluid.pressure_calc(P_0_2, T_0_2)

# 3rd part
T_0_3 = np.linspace(300, 320, num=n)
P_0_3 = np.linspace(6.7, 10.0, num=n)
root3 = fluid.pressure_calc(P_0_3, T_0_3)

T_sat = np.concatenate((T_0_1, T_0_2, T_0_3))
p_sat = np.concatenate((root1, root2, root3))

# plotting pressure
data_fernandez = np.array([
    # T,K p,MPa
    [194.225, 0.0977],
    [194.565, 0.1004],
    [196.116, 0.1201],
    [198.153, 0.1341],
    [199.809, 0.1527],
    [200.993, 0.1677],
    [201.014, 0.1679],
    [202.512, 0.1883],
    [203.953, 0.21],
    [205.639, 0.2381],
    [207.031, 0.264],
    [208.6, 0.2959],
    [210.036, 0.3281],
    [211.568, 0.3659],
    [211.569, 0.3659],
    [214.062, 0.4358],
    [215.47, 0.4803],
    [216.167, 0.5037],
    [216.268, 0.5072],
    [216.472, 0.5143],
    [216.56, 0.5174],
    [216.725, 0.5211],
    [216.831, 0.5234],
    [216.924, 0.5255],
    [217.041, 0.5282],
    [217.118, 0.53],
    [217.96, 0.5499],
    [218.168, 0.5545],
    [219.129, 0.5778],
    [220.173, 0.6036],
    [221.13, 0.6284],
    [222.07, 0.6533],
    [223.034, 0.6794],
    [227.812, 0.8213],
    [228.344, 0.8385],
    [232.653, 0.9865],
    [232.719, 0.9889],
    [238.05, 1.1982],
    [238.255, 1.2069],
    [243.04, 1.423],
    [243.082, 1.4248],
    [243.13, 1.4273],
])

data_sengers = np.array([
    # T, C p,bar
    [-5.9843, 29.646],
    [-3.9706, 31.3227],
    [-2.9805, 32.1759],
    [-1.9898, 33.0499],
    [0, 34.8501],
    [2.0126, 36.7426],
    [3.9963, 38.6816],
    [6.0197, 40.7359],
    [8.0024, 42.8287],
    [9.99, 45.0071],
    [11.9963, 47.288],
    [13.9891, 49.641],
    [16.0001, 52.0999],
    [16.5273, 52.7607],
    [17.9804, 54.6209],
    [20.0089, 57.2958],
    [21.9907, 60.0083],
    [23.9847, 62.8474],
    [24.9935, 64.3231],
    [25.9917, 65.812],
    [26.997, 67.3433],
    [28.0088, 68.9209],
    [28.4692, 69.6459],
    [28.999, 70.4939],
    [29.3968, 71.1354],
    [29.7991, 71.7925],
    [29.993, 72.1092],
    [30.1942, 72.4425],
    [30.3941, 72.7746],
    [30.4935, 72.9414],
    [30.5936, 73.1096],
    [30.695, 73.2782],
    [30.7913, 73.4395],
    [30.8915, 73.6094],
    [30.8946, 73.6133],
    [30.94, 73.6895],
    [30.9984, 73.7892],
    [31.0924, 73.9477],
    [31.1971, 74.1249],
    [31.3007, 74.3014],
    [31.406, 74.4776],
    [31.491, 74.6264],
    [31.5983, 74.809],
    [31.7785, 75.1163],
    [31.9904, 75.4788],
    [32.1894, 75.8184],
    [32.5925, 76.5086],
    [32.9896, 77.1889],
    [33.4904, 78.0464],
    [33.9885, 78.9024],
    [34.6893, 80.1072],
    [34.9875, 80.6186],
    [35.9883, 82.3451],
    [36.9935, 84.0809],
    [38.9889, 85.8049],
    [39.9902, 89.2749],
    [41.9894, 92.7498],
    [43.9898, 96.2331],
    [45.9894, 99.7202],
])
data_sengers[:, 0] += 273.15
data_sengers[:, 1] /= 10

data_michels = np.array([
    # p, atm, T,K
    [5.2339, 217.113],
    [5.3578, 217.654],
    [5.4955, 218.252],
    [5.643, 218.87],
    [6.0825, 220.655],
    [7.9409, 227.282],
    [10.197, 233.911],
    [13.392, 241.648],
    [15.95, 246.905],
    [18.742, 251.974],
    [19.895, 253.907],
    [24.424, 260.79],
    [29.726, 267.746],
    [32.889, 271.473],
    [33.694, 272.379],
    [34.4045, 273.165],
    [35.345, 274.186],
    [36.268, 275.171],
    [37.222, 276.165],
])
data_michels[:, 0] /= 9.869

plt.style.use('seaborn-whitegrid')
fig, ax = plt.subplots()
ax.plot(data_fernandez[:, 0], data_fernandez[:, 1], '*', label='fernandez-1984')
ax.plot(data_sengers[:, 0], data_sengers[:, 1], 'o', label='sengers-1972')
ax.plot(data_michels[:, 1], data_michels[:, 0], 'x', label='michels-1950')
ax.plot(T_sat, p_sat, color='k', label='comp')
ax.set(xlabel='T, K', ylabel='p, MPa')
ax.legend()
plt.legend(frameon=True)
plt.show()
