import jax
import jax.numpy as jnp
import numpy as np
from jax_sysid.utils import vec_reshape

from typing import Optional, Tuple, Union
Array = Union[np.ndarray, jnp.ndarray]


class GeneralNonlinearSystem(object):
    """
    Base class of a general nonlinear baseline model in state-space form. The exact dynamical relations should be
    implemented in child.
    """
    def __init__(self,
                 nx: int,
                 ny: int,
                 nu: int,
                 params: Optional[Array] = None,
                 ts: Optional[float] = None,
                 tune_params: bool = False
                 ) -> None:
        """
        Initialized the model.

        Parameters
        ----------
        nx : int
            Model state dimension.
        ny : int
            Output dimension.
        nu : int
            Input dimension.
        params : ndarray, optional
            Initial baseline parameters as a 1-dimensional numpy array. Should be provided if baseline parameters are
            to be co-estimated with augmentation parameters. If None, the p
        ts : float, optional
            Sampling-time.
        tune_params : bool, optional
            Whether to co-estimate the baseline parameters with the learning component or not.
        """
        self.nx = nx
        self.ny = ny
        self.nu = nu
        self.tune_params = tune_params
        if params is not None:
            self.init_params = params.copy()
        else:
            self.tune_params = False
            self.init_params = np.zeros(1)
        self.ts = ts

    def simulate(self,
                U: Union[Array, list[Array]],
                x0: Optional[Union[Array, list[Array]]] = None,
                params: Optional[Array] = None,
                ) -> Tuple[Union[Array, list[Array]], Union[Array, list[Array]]]:
        """
        Simulate the baseline model.

        Parameters
        ----------
        U : ndarray or list of ndarrays
            Input data shaped (N, nu) or a list of input data sequences, each with a shape of (Ni, nu).
        x0 : ndarray or list of ndarrays, optional
            Initial state (ot list of initial states in case of multiple measurement sequences). if None, x0=0 is applied.
        params : ndarray, optional
            Physical parameters of the model. If Nine, the initial parameters are used.

        Returns
        -------
        Y : ndarray or list of ndarrays
            Simulated output trajectory (or trajectories).
        X : ndarray or list of ndarrays
            Simulated state trajectory / trajectories.
        """
        # Prepare input and state data
        if isinstance(U, list):
            N_meas = len(U)
            if x0 is None:
                x0 = [np.zeros(self.nx)] * N_meas
        else:
            N_meas = 1
            if x0 is None:
                x0 = np.zeros(self.nx)

        if params is None:
            params = self.init_params

        @jax.jit
        def model_step(x, u):
            y = jnp.hstack((self.h(x, u, params), x))
            x = self.f(x, u, params).reshape(-1)
            return x, y

        if N_meas == 1:
            _, YX = jax.lax.scan(model_step, x0.reshape(-1), vec_reshape(U))
            Y = YX[:, 0:self.ny]
            X = YX[:, self.ny:]
        else:
            Y = []
            X = []
            for i in range(N_meas):
                _, YX = jax.lax.scan(model_step, x0[i].reshape(-1), vec_reshape(U[i]))
                Y.append(YX[:, 0:self.ny])
                X.append(YX[:, self.ny:])
        return Y, X

    def f(self, x: Array, u: Array, params: Array) -> Array:
        """Discrete-time (or discretized) state-transition function. """
        raise NotImplementedError("State transition function should be implemented in child!")

    def h(self, x: Array, u: Array, params: Array):
        """Output map of the model."""
        raise NotImplementedError("Output map should be implemented in child!")


class LinearTimeInvariantSystem(GeneralNonlinearSystem):
    """
    DT Linear Time Invariant (LTI) baseline model. This class does not enable tuning of the baseline parameters.
    For that application, the LTI system dynamics should be manually implemented into a child of GeneralNonlinearSystem.
    """
    def __init__(self,
                 A: Array,
                 B: Array,
                 C: Array,
                 D: Optional[Array] = None
                 ) -> None:
        """
        Initializes the DT LTI model with system matrices.

        Parameters
        ----------
        A : ndarray
            A matrix shaped (nx, nx).
        B : ndarray
            B matrix shaped (nx, nu),
        C : ndarray
            C matrix shaped (ny, nx).
        D : ndarray, optional
            D matrix shaped (ny, nu). If None, D=0 is applied.
        """
        self.A = A
        self.B = B
        self.C = C
        super().__init__(nx=A.shape[0], nu=B.shape[1], ny=C.shape[0])
        if D is None:
            self.D = np.zeros((self.ny, self.nu))
        else:
            self.D = D

    def f(self,
          x: Array,
          u: Array,
          params: Array,
          ) -> Array:
        x_next = self.A @ vec_reshape(x) + self.B @ vec_reshape(u)
        return x_next.reshape(-1)

    def h(self,
          x: Array,
          u: Array,
          params: Array,
          ) -> Array:
        y = self.C @ vec_reshape(x) + self.D @ vec_reshape(u)
        return y.reshape(-1)


def verify_known_sys(sys):
    if isinstance(sys, LinearTimeInvariantSystem):
        tune_physical_params = False
    elif issubclass(type(sys), GeneralNonlinearSystem):
        tune_physical_params = sys.tune_params
    else:
        raise ValueError("'known_sys' parameter should be either an instance of 'LinearTimeInvariantSystem' or an instance of "
                         "a subclass of 'GeneralNonlinearSystem'!")
    init_phys_params = sys.init_params
    return tune_physical_params, init_phys_params

if __name__ == "__main__":
    A = np.array([[0.7, 0.2], [-0.1, 0.9]])
    B = np.array([[0.1],[0.05]])
    C = np.array([[1.0, 0.0]])

    sys = LinearTimeInvariantSystem(A, B, C)

    N = 200
    U = np.random.randn(N, 1)
    x0 = np.array([0.5, -0.2])
    Y, X = sys.simulate(U, x0)

    from matplotlib import pyplot as plt
    plt.figure()
    plt.plot(Y)
    plt.xlabel("Sim. index")
    plt.ylabel("Output")
    plt.show()
