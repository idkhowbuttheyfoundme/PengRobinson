import numpy as np


def cubic_f(x, b, c, d):
    return np.power(x, 3) + b * np.power(x, 2) + c * x + d


'''
Newton's method
x0 - initial guess
'''


def newton(x0, b, c, d):
    x = np.copy(x0)
    nan = np.isnan(x0)
    f = cubic_f(x, b, c, d)
    zeros = np.zeros(f.shape, dtype=np.float_)
    if np.any(~nan):
        solution = (~nan & ~(np.isclose(f, zeros)))
        while np.any(solution):
            x0[solution] = x[solution]
            f = cubic_f(x[solution], b[solution], c[solution], d[solution])
            df = 3 * np.power(x[solution], 2) + 2 * b[solution] * x[solution] + c[solution]
            x[solution] = x0[solution] - f / df
            solution = (solution & (np.abs(x - x0) >= 1e-10))

    return x


'''
  bisection method
  x1 - lower boundary
  x2 - upper boundary
'''


def bisect(x1, x2, b, c, d):

    xc = np.full(x1.size, np.NaN, dtype=np.float_)
    f1 = cubic_f(x1, b, c, d)
    f2 = cubic_f(x2, b, c, d)

    # non-defined roots
    non_def = (np.sign(f1) * np.sign(f2) > 0)
    xc[non_def] = np.NaN

    # roots equal to the left side
    left_side = (f1 == 0)
    if np.any(left_side):
        xc[left_side] = x1[left_side]

    # roots equal to the right side
    right_side = (f2 == 0)
    if np.any(right_side):
        xc[right_side] = x2[right_side]

    # all other possible roots
    solution = ((~non_def & ~left_side & ~right_side) & (np.abs(x2 - x1) > 0.01))

    while np.any(solution):
        xc[solution] = x1[solution] + (x2[solution] - x1[solution]) / 2
        fc = cubic_f(xc, b, c, d)

        left = ((np.sign(f1) * np.sign(fc) < 0) & solution)
        right = ((np.sign(f2) * np.sign(fc) < 0) & solution)

        f2[left], x2[left] = fc[left], xc[left]
        f1[right], x1[right] = fc[right], xc[right]

        solution = ((~non_def & ~left_side & ~right_side) & (np.abs(x2 - x1) > 0.01) & ~(fc == 0))
    return xc


'''
  main function for solving
  if D > 0 - three real roots are possible but they must be >= 0 for Peng-Robinson EOS
  if D = 0 or D < 0 - 1 real root is possible
  '''


def add_root(x_low, x_high, b, c, d):
    xnew = newton(bisect(x_low, x_high, b, c, d), b, c, d)
    return xnew


def cubiceq(b, c, d):
    D = np.power(b, 2) - 3 * c
    x = np.full((b.size, 3), np.NaN, dtype=np.float_)

    # positive D
    positive_D = (D > 0)
    if np.any(positive_D):
        xe1 = (-b[positive_D] - np.sqrt(D[positive_D])) / 3
        xe2 = (-b[positive_D] + np.sqrt(D[positive_D])) / 3

        zero = np.zeros(xe1.shape, dtype=np.float_)
        three = np.full(xe1.shape, 3, dtype=np.float_)

        x[positive_D, 0] = add_root(zero, xe1, b[positive_D], c[positive_D], d[positive_D])
        x[positive_D, 1] = add_root(xe1, xe2, b[positive_D], c[positive_D], d[positive_D])
        x[positive_D, 2] = add_root(xe2, three, b[positive_D], c[positive_D], d[positive_D])

    # null D
    null_D = (D == 0)
    if np.any(null_D):
        xe = -b[null_D] / 3
        zero = np.zeros(xe.shape, dtype=np.float_)
        three = np.full(xe.shape, 3, dtype=np.float_)

        x[null_D, 0] = add_root(zero, xe, b[null_D], c[null_D], d[null_D])
        x[null_D, 1] = add_root(xe, three, b[null_D], c[null_D], d[null_D])

    # other cases
    other = ~positive_D & ~null_D

    if np.any(other):
        one = np.full(D[other].shape, 1, dtype=np.float_)
        x[other, 0] = newton(one, b[other], c[other], d[other])

    # ignore all negative solutions

    x[x < 0] = np.NaN
    return x
