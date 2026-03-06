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
