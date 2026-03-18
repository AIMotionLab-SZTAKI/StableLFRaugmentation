import numpy as np
from jax import numpy as jnp
from jax_sysid.utils import vec_reshape

from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union
Array = Union[np.ndarray, jnp.ndarray]


def create_ndarray_from_list(var_list: List[Array]) -> np.ndarray:
    """"
    Concatenates a list of ndarrays into one single ndarray.

    Parameters
    ----------
    var_list : list of ndarrays
        A list containing ndarrays of size (Ni, n), where Ni can vary, but n must be the same for each entry.

    Returns
    -------
    var_array : ndarray
        The concatenated ndarray.
    """
    var_array = np.array([])
    for i in range(len(var_list)):
        var_i = vec_reshape(var_list[i])
        if i == 0:
            var_array = var_i.copy()
        else:
            var_array = np.vstack((var_array, var_i))
    return var_array


def NRMSE_loss(Yhat: Array,
               Y: Array,
               ) -> float:
    """
    Computes the Normalized Root Mean Squared loss between a measured and simulated output trajectory.

    Parameters
    ----------
    Yhat : ndarray
        Simulated output trajectory.
    Y : ndarray
        Measured output trajectory.

    Returns
    -------
    loss : float
        Normalized Root Mean Squared loss.
    """
    Y = vec_reshape(Y)
    Yhat = vec_reshape(Yhat)
    return float(jnp.mean(jnp.sqrt(jnp.mean((Y - Yhat)**2, axis=0)) / jnp.std(Y, axis=0)))


def BestFitRatio(Yhat: Array,
                 Y: Array,
                 ) -> float:
    """
    Computes the Best Fit Ratio between a measured and simulated output trajectory.

    Parameters
    ----------
    Yhat : ndarray
        Simulated output trajectory.
    Y : ndarray
        Measured output trajectory.

    Returns
    -------
    bfr : float
    Best Fit Ratio.Best Fit Ratio.
    """
    Y = vec_reshape(Y.copy())
    Yhat = vec_reshape(Yhat.copy())
    Ymean = jnp.mean(Y, axis=0)

    t = jnp.sum(jnp.mean((Y - Yhat)**2, axis=1))
    b = jnp.sum(jnp.mean((Y - Ymean)**2, axis=1))
    bfr = float(jnp.maximum(1 - jnp.sqrt(t/b), 0))

    return bfr


def build_N_from_XYZ(
        X: Array,
        Y: Array,
        Z: Array,
) -> Array:
    """Builds N matrix from X, Y, Z tunable matrices for the generalized Cayley transformation."""
    # X: (m,m), Y: (m,m), Z: (n-m,m)
    return X.T @ X + (Y - Y.T) + Z.T @ Z + 1e-6 * jnp.eye(X.shape[0])


def build_N_from_XY(
        X: Array,
        Y: Array,
) -> Array:
    """Builds N matrix from tunable X and Y matrices for the simple Cayley transformation."""
    # X: (n,n), Y: (n,n)
    return X.T @ X + (Y - Y.T) + 1e-6 * jnp.eye(X.shape[0])


def general_cayley(
        X: Array,
        Y: Array,
        Z: Array,
) -> Array:
    """General Cayley transformation for n-by-m matrices."""
    N = build_N_from_XYZ(X, Y, Z)
    m = N.shape[0]
    I = jnp.eye(m)

    # compute M_top = Cayley(N) = N2 @ N1^{-1} and M_bottom = -2 Z @ N1^{-1}
    inv = jnp.linalg.solve(I + N, I)
    M_top = (I - N) @ inv
    M_bot = -2. * Z @ inv
    return jnp.vstack([M_top, M_bot])


def simple_cayley(
        X: Array,
        Y: Array,
) -> Array:
    """Simple Cayley transformation for square matrices."""
    N = build_N_from_XY(X, Y)
    n = N.shape[0]
    I = jnp.eye(n)
    # N1 = I + N
    # N2 = I - N

    # compute M = Cayley(N)
    M = (I - N) @ jnp.linalg.solve(I + N, I)
    return M
