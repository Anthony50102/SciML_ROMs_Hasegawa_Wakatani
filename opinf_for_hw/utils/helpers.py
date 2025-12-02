import numpy as np


def get_r(proj_err, cummulative_error):
    if cummulative_error[0] < proj_err:
        r = 1
    else:
        arr_temp = cummulative_error / proj_err - 1

        r = arr_temp[arr_temp >= 0].argmin() + 1

    return r


def get_square_number(r1, r2):
    r_min = min(r1, r2)
    r_max = max(r1, r2)

    sq_no = 0

    for i in range(r_min):
        sq_no += r_max - i

    return sq_no


# gets non-redundant quadratic terms of X
def get_x_sq(X):
    if len(np.shape(X)) == 1:  # if X is a vector
        r = np.size(X)
        prods = []
        for i in range(r):
            temp = X[i] * X[i:]
            prods.append(temp)
        X2 = np.concatenate(tuple(prods))

    elif len(np.shape(X)) == 2:  # if X is a matrix
        K, r = np.shape(X)

        prods = []
        for i in range(r):
            temp = np.transpose(np.broadcast_to(X[:, i], (r - i, K))) * X[:, i:]
            prods.append(temp)
        X2 = np.concatenate(tuple(prods), axis=1)

    else:
        print("invalid input size for helpers.get_x_sq")
    return X2


def get_x_times_y(X, Y):
    if len(np.shape(X)) == 1 and len(np.shape(Y)) == 1:  # if X and Y are vectors
        r1 = np.size(X)
        r2 = np.size(Y)

        prods = []

        if r1 <= r2:
            for i in range(r1):
                temp = X[i] * Y[i:]
                prods.append(temp)

        else:
            for i in range(r2):
                temp = Y[i] * X[i:]
                prods.append(temp)

        XY = np.concatenate(tuple(prods))

    elif len(np.shape(X)) == 2 and len(np.shape(Y)) == 2:  # if X is a matrix
        K1, r1 = np.shape(X)
        K2, r2 = np.shape(Y)

        prods = []

        if r1 <= r2:
            for i in range(r1):
                temp = np.transpose(np.broadcast_to(X[:, i], (r2 - i, K2))) * Y[:, i:]
                prods.append(temp)
        else:
            for i in range(r2):
                temp = np.transpose(np.broadcast_to(Y[:, i], (r1 - i, K1))) * X[:, i:]
                prods.append(temp)

        XY = np.concatenate(tuple(prods), axis=1)

    else:
        print("invalid input size for helpers.get_x_sq")
    return XY


def get_x_sq_dd_2(X, r1, r2):
    assert X.shape[0] == r1 + r2

    X1 = X[:r1]
    X2 = X[r1 : r1 + r2]

    X1_sq = get_x_sq(X1)
    X12 = get_x_times_y(X1, X2)
    X2_sq = get_x_sq(X2)

    X_dd = np.concatenate((X1_sq, X12, X2_sq), axis=0)

    return X_dd


# returns vector of 2-norms of each column of matrix
def col2norm(X):
    return np.sqrt(np.sum(X**2, axis=0))


# return time derivative approximation
def ddt(X, dt, order=1, axis=1):
    if len(np.shape(X)) == 1:  # if vector pretend it's a matrix
        X = np.array([X])

    if axis == 0:  # if successive times are in different rows
        X = np.transpose(X)

    # first order differencing
    if order == 1:
        dXdt = (X[:, 1:] - X[:, :-1]) / dt

    # second order forward differencing
    elif order == "2f":
        dXdt = (-0.5 * X[:, 2:] + 2 * X[:, 1:-1] - 1.5 * X[:, :-2]) / dt

    # second order central differencing
    elif order == "2c":
        dXdt = (0.5 * X[:, 2:] - 0.5 * X[:, :-2]) / dt

    # second order backward differencing
    elif order == "2b":
        dXdt = (1.5 * X[:, 2:] - 2 * X[:, 1:-1] + 0.5 * X[:, :-2]) / dt

    # third order forward differencing
    elif order == "3f":
        dXdt = (
            1.0 / 3 * X[:, 3:]
            - 1.5 * X[:, 2:-1]
            + 3 * X[:, 1:-2]
            - 11.0 / 6 * X[:, :-3]
        ) / dt

    # third order backward differencing
    elif order == "3b":
        dXdt = (
            11.0 / 6 * X[:, 3:]
            - 3 * X[:, 2:-1]
            + 1.5 * X[:, 1:-2]
            - 1.0 / 3 * X[:, :-3]
        ) / dt

    # fourth order forward differencing
    elif order == "4f":
        dXdt = (
            -0.25 * X[:, 4:]
            + 4.0 / 3 * X[:, 3:-1]
            - 3 * X[:, 2:-2]
            + 4 * X[:, 1:-3]
            - 25.0 / 12 * X[:, :-4]
        ) / dt

    # fourth order central differencing
    elif order == "4c":
        dXdt = (
            -1.0 / 12 * X[:, 4:]
            + 2.0 / 3 * X[:, 3:-1]
            - 2.0 / 3 * X[:, 1:-3]
            + 1.0 / 12 * X[:, :-4]
        ) / dt

    # fourth order backward differencing
    elif order == "4b":
        dXdt = (
            25.0 / 12 * X[:, 4:]
            - 4 * X[:, 3:-1]
            + 3 * X[:, 2:-2]
            - 4.0 / 3 * X[:, 1:-3]
            + 0.25 * X[:, :-4]
        ) / dt

    if axis == 0:  # if row-wise, undo the initial transpose
        dXdt = np.transpose(dXdt)
    return dXdt


def expand_Hc(Hc):
    """Calculate the matricized quadratic operator that operates on the full
    Kronecker product.
    Parameters
    ----------
    Hc : (r,s) ndarray
        The matricized quadratic tensor that operates on the COMPACT Kronecker
        product. Here s = r * (r+1) / 2.
    Returns
    -------
    H : (r,r**2) ndarray
        The matricized quadratic tensor that operates on the full Kronecker
        product. This is a symmetric operator in the sense that each layer of
        H.reshape((r,r,r)) is a symmetric (r,r) matrix.
    """
    r, s = Hc.shape
    # if s != r*(r+1)//2:
    #    raise ValueError(f"invalid shape (r,s) = {(r,s)} with s != r(r+1)/2")

    H = np.zeros((r, r**2))
    fj = 0
    for i in range(r):
        for j in range(i + 1):
            if i == j:  # Place column for unique term.
                H[:, (i * r) + j] = Hc[:, fj]
            else:  # Distribute columns for repeated terms.
                fill = Hc[:, fj] / 2
                H[:, (i * r) + j] = fill
                H[:, (j * r) + i] = fill
            fj += 1

    return H


def compress_H(H):
    """Calculate the matricized quadratic operator that operates on the compact
    Kronecker product.
    Parameters
    ----------
    H : (r,r**2) ndarray
        The matricized quadratic tensor that operates on the Kronecker product.
        This should be a symmetric operator in the sense that each layer of
        H.reshape((r,r,r)) is a symmetric (r,r) matrix, but it is not required.
    Returns
    -------
    Hc : (r,s) ndarray
        The matricized quadratic tensor that operates on the COMPACT Kronecker
        product. Here s = r * (r+1) / 2.
    """
    r = H.shape[0]
    r2 = H.shape[1]
    # if r2 != r**2:
    #   raise ValueError(f"invalid shape (r,a) = {(r,r2)} with a != r**2")
    s = r * (r + 1) // 2
    Hc = np.zeros((r, s))

    fj = 0
    for i in range(r):
        for j in range(i + 1):
            if i == j:  # Place column for unique term.
                Hc[:, fj] = H[:, (i * r) + j]
            else:  # Combine columns for repeated terms.
                fill = H[:, (i * r) + j] + H[:, (j * r) + i]
                Hc[:, fj] = fill
            fj += 1

    return Hc


# dt = 0.1
# t = np.arange(0,5,dt)
# x = 0.5*t**2
# print x
# print ddt(x,dt)

import xarray as xr
def loader(path, ENGINE="h5netcdf"):
    try:
        fh = xr.open_dataset(path, engine=ENGINE)
        return fh
    except Exception as e:
        print(f"\033[91m ERROR: Could not open file {path}: {e} \033[0m")
        print("  Retrying with phony_dims='sort'...")
        try:
            fh = xr.open_dataset(path, engine=ENGINE, phony_dims="sort")
            return fh
        except Exception as e:
            print(f"\033[91m ERROR: Could not open file {path} with phony_dims: {e} \033[0m")

