import numpy as np
from cubiceq import cubiceq

from scipy.optimize import fsolve

'''
Peng Robinson EOS.
Example of creating an object:
fluid = PengRobinson(Tc, Pc, W)
'''


class PengRobinson:
    R = 8.3145 / 1000

    def __init__(self, Tc, Pc, W):
        self.Tc = Tc
        self.Pc = Pc
        self.W = W
        self.ac = 0.45724 * (self.R * self.Tc) ** 2 / self.Pc
        self.bc = 0.07780 * (self.R * self.Tc) / self.Pc

    def alpha_calc(self, T):
        # calculating alpha
        return (1 + (0.37464 + 1.54226 * self.W - 0.26992 * self.W ** 2) * (1 - np.sqrt(T / self.Tc))) ** 2

    def a_calc(self, alpha):
        # calculating a(T)
        return self.ac * alpha

    def A_calc(self, a, p, T):
        # calculating A
        return a * p / np.power(self.R * T, 2)

    def B_calc(self, p, T):
        # calculating B
        return self.bc * p / (self.R * T)

    def density_calc(self, p, T):
        """
        Function to calculate density at the given temperature
        :param p: numpy array/constant - values of pressure
        :param T: constant - value of temperature
        :return: numpy array - values of density
        Example: density_T = fluid.density_calc(P, T)
        """
        # calculating density_300
        alpha = self.alpha_calc(T)
        a = self.a_calc(alpha)
        A, B = self.A_calc(a, p, T), self.B_calc(p, T)
        Z = cubiceq(- (1 - B), A - 3 * np.power(B, 2) - 2 * B, - (A * B - np.power(B, 2) - np.power(B, 3)))
        Z = Z[~np.isnan(Z)]
        return p / (self.R * T * Z)

    def pressure_calc(self, p, T):
        """
        Function to calculate saturated steam pressure
        :param p: numpy array - values of pressure (initial guess)
        :param T: numpy array - values of T
        :return: numpy array - values of saturated steam pressure
        Example: root1 = fluid.pressure_calc(P, T)
        """

        def f(p_opt, T_opt):
            alpha = self.alpha_calc(T_opt)
            a = self.a_calc(alpha)
            A, B = self.A_calc(a, p_opt, T_opt), self.B_calc(p_opt, T_opt)
            Z = cubiceq(- (1 - B), A - 3 * B ** 2 - 2 * B, - (A * B - B ** 2 - B ** 3))
            Zv = np.nanmax(Z, axis=1)
            Zl = np.nanmin(Z, axis=1)
            return \
                Zv - Zl - np.log(Zv - B) + np.log(Zl - B) \
                - A / (2 * np.sqrt(2) * B) * (
                        np.log((Zv + 2.414 * B) / (Zv - 0.414 * B)) - np.log((Zl + 2.414 * B) / (Zl - 0.414 * B)))

        return fsolve(f, p, T)
