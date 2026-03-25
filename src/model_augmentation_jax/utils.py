import numpy as np
import jax
from jax import numpy as jnp
from jax_sysid.utils import vec_reshape
from joblib import Parallel, delayed, cpu_count

from typing import Any, List, Tuple, Union
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


def compute_normalization_constants(
        U: Union[Array, list[Array]],
        Y: Union[Array, list[Array]],
        X: Union[Array, list[Array]],
) -> dict[str, Array]:
    """Computes normalization constants for a set of U, Y, and X."""
    if isinstance(U, list):
        U_all = create_ndarray_from_list(U)
        Y_all = create_ndarray_from_list(Y)
        X_all = create_ndarray_from_list(X)
    else:
        U_all = vec_reshape(U.copy())
        Y_all = vec_reshape(Y.copy())
        X_all = vec_reshape(X.copy())

    std_u = np.std(U_all, axis=0)
    mu_u = np.std(U_all, axis=0)
    std_y = np.std(Y_all, axis=0)
    mu_y = np.std(Y_all, axis=0)
    std_x = np.std(X_all, axis=0)
    mu_x = np.mean(X_all, axis=0)

    norm = {"std_u": std_u, "mean_u": mu_u, "std_y": std_y, "mean_y": mu_y, "std_x": std_x, "mean_x": mu_x}
    return norm


def find_best_model(
        models: list[Any],
        Y: Union[Array, list[Array]],
        U: Union[Array, list[Array]],
        X0=None,
        n_jobs=None,
        verbose=True,
        seeds=None,
        use_training_x0=False,
        x0_estim_kwargs=None,
        state_estim_len=None
) -> Tuple[Any, Array]:
    """
    Finds the best model based on simulation RMSE on the provided IO trajectory.

    """
    if not isinstance(models, list):
        raise Exception("\033[1mPlease provide a list of models to compare.\033[0m")

    if len(models) == 1:
        return models[0]

    if isinstance(Y, list):
        N_meas = len(Y)
    else:
        N_meas = 1
        U = [U.copy()]
        Y = [Y.copy()]

    if state_estim_len is None:
        Uhist = [vec_reshape(u) for u in U]
        Yhist = [vec_reshape(y) for y in Y]
    else:
        Uhist = [vec_reshape(u)[:state_estim_len, :] for u in U]
        Yhist = [vec_reshape(y)[:state_estim_len, :] for y in Y]

    def get_X0(k):
        if X0 is not None:
            return X0
        elif use_training_x0:
            return models[k].x0
        else:
            # estimate x0 based on model
            if x0_estim_kwargs is None:
                X0_est = []
                for i in range(N_meas):
                    x0i = models[k].learn_x0(Uhist[i], Yhist[i], verbosity=False)
                    X0_est.append(x0i)
            else:
                X0_est = []
                for i in range(N_meas):
                    x0i = models[k].learn_x0(Uhist[i], Yhist[i], *x0_estim_kwargs, verbosity=False)
                    X0_est.append(x0i)

    def score_model(k):
        if not jax.config.jax_enable_x64:
            # Enable 64-bit computations
            jax.config.update("jax_enable_x64", True)
        X0k = get_X0(k)
        sim_results = models[k].simulate(U, X0k)
        Yhat = sim_results[0]
        rmse = np.mean(np.sqrt(np.mean((create_ndarray_from_list(Y) - create_ndarray_from_list(Yhat)) ** 2, axis=0)))  # RMSE (averaged over all channels)
        return rmse

    if n_jobs is None:
        n_jobs = cpu_count()  # Use all available cores by default

    if verbose:
        print("Evaluating models...\n")

    scores = Parallel(n_jobs=n_jobs)(delayed(score_model)(k) for k in range(len(models)))
    best_id = np.nanargmin(np.array(scores))  # get best score (lowest RMSE)

    if verbose:
        print("Scores:")
        for k in range(len(models)):
            print(f"Model {k}: RMSE = {scores[k]}")
        if seeds is None:
            print(f"Best model: {best_id}, score: {scores[best_id]}")
        else:
            print(f"Best model: {best_id}, score: {scores[best_id]} at seed {seeds[best_id]}")

    return models[best_id], scores[best_id]
