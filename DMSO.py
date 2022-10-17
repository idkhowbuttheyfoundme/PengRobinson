from main import PengRobinson
import numpy as np
import matplotlib.pyplot as plt

Tc = 706.9
Pc = 58.5 / 10  # bar to MPa
W = 0.45
M = 78.13

fluid = PengRobinson(Tc, Pc, W)

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

T_sat = np.concatenate((T_0_1, T_0_2, T_0_3))
p_sat = np.concatenate((root1, root2, root3))

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

plt.style.use('seaborn')
fig, ax = plt.subplots()
ax.plot(data_tochigi[:, 0], data_tochigi[:, 1], 'o', label='tochigi 1999')
ax.plot(T_sat, p_sat)
ax.legend()
plt.legend(frameon=True)
plt.show()
