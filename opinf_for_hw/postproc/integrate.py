import numpy as np
from ..utils import *
# from scipy.optimize import broyden1


def forwardEuler(f, u0, t):
    u = np.zeros((np.size(u0), len(t)))
    u[:, 0] = u0
    for i in range(len(t) - 1):
        dt = t[i + 1] - t[i]
        u[:, i + 1] = u[:, i] + dt * f(u[:, i], t[i])
        if np.any(np.isnan(u[:, i + 1])):
            print("NaN encountered at time " + str(t[i + 1]))
            break
    return u


def rk2inp(f, u0, t, inp):
    u = np.zeros((np.size(u0), len(t)))
    u[:, 0] = u0
    for i in range(len(t) - 1):
        dt = t[i + 1] - t[i]
        temp = f(u[:, i], t[i], inp[i])
        u[:, i + 1] = u[:, i] + dt * 0.5 * (
            temp + f(u[:, i] + dt * temp, t[i + 1], inp[i + 1])
        )
        if np.any(np.isnan(u[:, i + 1])):
            print("NaN encountered at time " + str(t[i + 1]))
            break
    return u


def rk2(f, u0, t):
    is_nan = False

    u = np.zeros((np.size(u0), len(t)))
    u[:, 0] = u0
    for i in range(len(t) - 1):
        dt = t[i + 1] - t[i]
        temp = f(u[:, i], t[i])
        u[:, i + 1] = u[:, i] + dt * 0.5 * (temp + f(u[:, i] + dt * temp, t[i + 1]))
        if np.any(np.isnan(u[:, i + 1])):
            print("NaN encountered at time " + str(t[i + 1]))

            is_nan = True

            break
    return u, is_nan


def rk4(f, u0, t):
    u = np.zeros((np.size(u0), len(t)))
    u[:, 0] = u0
    for i in range(len(t) - 1):
        dt = t[i + 1] - t[i]
        temp = f(u[:, i], t[i])
        k1 = f(u[:, i], t[i])
        k2 = f(u[:, i] + dt * k1 / 2, t[i] + dt / 2)
        k3 = f(u[:, i] + dt * k2 / 2, t[i] + dt / 2)
        k4 = f(u[:, i] + dt * k3, t[i] + dt)
        u[:, i + 1] = u[:, i] + dt / 6 * (k1 + 2 * k2 + 2 * k2 + k4)

        if np.any(np.isnan(u[:, i + 1])):
            print("NaN encountered at time " + str(t[i + 1]))
            break
    return u


# def backwardEuler(f, u0, t):
#     u = np.zeros((len(t), np.size(u0)))
#     u[0, :] = u0
#     for i in range(len(t) - 1):
#         dt = t[i + 1] - t[i]
#         g = lambda x: (x - u[i, :] - dt * f(x, t[i + 1]))
#         temp = u[i, :] + dt * f(u[i, :], t)
#         u[i + 1, :] = broyden1(g, temp)
#     return u


# def crankNicolson(f, u0, t):
#     u = np.zeros((np.size(u0), len(t)))
#     u[:, 0] = u0
#     for i in range(len(t) - 1):
#         dt = t[i + 1] - t[i]
#         g = lambda x: (x - u[:, i] - dt * 0.5 * (f(x, t[i + 1]) + f(u[:, i], t[i])))
#         u[:, i + 1] = broyden1(g, u[:, i])
#     return u


def semiImp_AFC(A, F, C, u0, t):
    n = np.size(u0)
    u = np.zeros((n, len(t)))
    u[:, 0] = u0
    for i in range(len(t) - 1):
        dt = t[i + 1] - t[i]
        u[:, i + 1] = np.linalg.solve(
            np.eye(n) - dt * A, u[:, i] + dt * C + dt * np.dot(F, get_x_sq(u[:, i]))
        )
    return u


def backwardEuler_AFC(A, F, C, u0, t):
    n = np.size(u0)

    H = expand_Hc(F)
    H = H.reshape((n, n, n))

    dgdy = lambda y: np.eye(n) - dt * A - 2 * dt * np.dot(H, y)

    u = np.zeros((n, len(t)))
    u[:, 0] = u0
    for i in range(len(t) - 1):
        dt = t[i + 1] - t[i]
        g = lambda y: y - u[:, i] - dt * (np.dot(A, y) + np.dot(F, get_x_sq(y)) + C)

    return u
