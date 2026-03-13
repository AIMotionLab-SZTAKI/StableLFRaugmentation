from model_augmentation_jax.optimization_base import AugmentationBase
from model_augmentation_jax.networks import initialize_network, generate_simple_ann
from model_augmentation_jax.utils import simple_cayley, general_cayley
import numpy as np
from jax import numpy as jnp
from jax_sysid.utils import vec_reshape
import jax
import jaxopt
import flax.linen as nn

from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union
Array = Union[np.ndarray, jnp.ndarray]


########################################################################################################################
#                                       STATIC LFR-BASED STRUCTURE
########################################################################################################################
class StaticLFRAugmentation(AugmentationBase):
    def __init__(self,
                 known_sys: Any,
                 hidden_layers: int,
                 nodes_per_layer: int,
                 activation: str,
                 nz: int,
                 nw: int,
                 x0: Optional[Union[Array, list[Array]]] = None,
                 seed: Union[int, list[int]] = 42,
                 std_x: Optional[np.ndarray] = None,
                 std_u: Optional[np.ndarray] = None,
                 std_y: Optional[np.ndarray] = None,
                 mu_x: Optional[np.ndarray] = None,
                 mu_u: Optional[np.ndarray] = None,
                 mu_y: Optional[np.ndarray] = None,
                 Dzw_structure: Optional[str] = None,
                 ) -> None:
        """
        Initialize the static LFR-based model augmentation structure.

        Parameters
        ----------
        known_sys : object
            Baseline (first-principles) model.
        hidden_layers : int
            Number of hidden layers in the ANN.
        nodes_per_layer : int
            Neurons per hidden layer in the ANN.
        activation : str
            Activation function for the ANN.
        x0 : array or list of arrays, optional
            Initial state(s).
        seed : int or list, optional
            Initialization seed(s).
        std_x, std_u, std_y : ndarray, optional
            Standard deviations for normalization.
        mu_x, mu_u, mu_y : ndarray, optional
            Means for normalization.
        Dzw_structure : str, optional
            If "lower", the $D_{zw}$ matrix is implemented as a strictly lower triangular matrix, if "upper", than a
            strictly lower triangular matrix. If None, $D_{zw}\equiv 0$ is applied.
        """
        self.nz = nz
        self.nw = nw
        if Dzw_structure == "upper":
            raise NotImplementedError("Dzw structure can not be an upper triangular matrix at the moment.")
        elif Dzw_structure == "lower":
            self.Dzw_structure = Dzw_structure
        elif Dzw_structure is None:
            self.Dzw_structure = None
        else:
            raise ValueError("Dzw_structure must be either 'upper' or 'lower' or None (for zero matrix).")
        super().__init__(known_sys=known_sys, hidden_layers=hidden_layers, nodes_per_layer=nodes_per_layer,
                         activation=activation, x0=x0, seed=seed, std_y=std_y, std_x=std_x, std_u=std_u, mu_y=mu_y,
                         mu_x=mu_x, mu_u=mu_u)

    def sparsity_analysis(self) -> Tuple[int, int]:
        """
        Provides a sparsity analysis of LFR matrices.

        Returns
        -------
        z_reduction : int
            The number of redundant dimensions in the latent variable z_a.
        w_reduction : int
            The number of redundant dimensions in the latent variable w_a.
        """
        th = self.params
        Cz_a = np.array(th[-2])
        Dzu_a = np.array(th[-1])
        ANN_params = self.get_network_params(th)
        W0 = np.array(ANN_params[0])
        W_last = np.array(ANN_params[-2])
        b_last = np.array(ANN_params[-1])
        Bw_a = np.array(th[-7])
        Dyw_a = np.array(th[-3])

        # zero-out coefficients smaller than self.zero_coeff
        Cz_a[np.abs(Cz_a) <= self.zero_coeff] = 0.
        Dzu_a[np.abs(Dzu_a) <= self.zero_coeff] = 0.
        W0[np.abs(W0) <= self.zero_coeff] = 0.
        W_last[np.abs(W_last) <= self.zero_coeff] = 0.
        b_last[np.abs(b_last) <= self.zero_coeff] = 0.
        Bw_a[np.abs(Bw_a) <= self.zero_coeff] = 0.
        Dyw_a[np.abs(Dyw_a) <= self.zero_coeff] = 0.

        if self.Dzw_structure is not None:  # now only lower-triangular Dzw structure is supported
            Dzw_ab_state = np.array(th[-12])
            Dzw_ab_state[np.abs(Dzw_ab_state) <= self.zero_coeff] = 0.
            Dzw_ab_output = np.array(th[-11])
            Dzw_ab_output[np.abs(Dzw_ab_output) <= self.zero_coeff] = 0.

        z_reduction = 0
        w_reduction = 0
        print("Sparsity analysis results:")

        for i in range(self.nz):
            if np.max(np.abs(Cz_a[i, :])) <= self.zero_coeff and np.max(np.abs(Dzu_a[i, :])) <= self.zero_coeff and np.max(np.abs(W0[:, i])) <= self.zero_coeff:
                if self.Dzw_structure is not None:
                    if np.max(np.abs(Dzw_ab_state[i, :])) <= self.zero_coeff and np.max(np.abs(Dzw_ab_output[i, :])) <= self.zero_coeff:
                        print(f"z_{i+1} can be eliminated")
                        z_reduction += 1
                else:
                    print(f"z_{i + 1} can be eliminated")
                    z_reduction += 1

        for i in range(self.nw):
            if (np.max(np.abs(W_last[i, :])) <= self.zero_coeff and np.abs(b_last[i]) <= self.zero_coeff and
                np.max(np.abs(Bw_a[:, i])) <= self.zero_coeff and np.max(np.abs(Dyw_a[:, i])) <= self.zero_coeff):
                print(f"w_{i + 1} can be eliminated")
                w_reduction += 1

        return z_reduction, w_reduction

    def _initialize_parameters(self, known_sys: Any, hidden_layers: int, nodes_per_layer: int,
                              x0: Optional[Union[Array, list[Array]]], seed: int, activation: str) -> None:
        """Initializes the parameters according to the given static LFR-based augmentation structure."""
        key = jax.random.key(seed)
        key_net, key_params = jax.random.split(key, 2)

        # init. network parameters for static additive case
        network_params = initialize_network(input_features=self.nz, output_features=self.nw, hidden_layers=hidden_layers,
                                            nodes_per_layer=nodes_per_layer, key=key_net, act_fun=activation)

        if self.tune_physical_params:
            # if Dzw_structure is None: theta = params[-13]
            # if Dzw_structure is not None: theta = params[-15]
            network_params.append(known_sys.init_params)

        if self.Dzw_structure == "lower":
            self.physical_param_idx = -15
            keys = jax.random.split(key, 10)
            Dzw_ab_state = jax.random.uniform(key=keys[8], shape=(self.nz, self.nx), minval=-1, maxval=1, dtype=jnp.float32)  # Dzw_ab_state = params[-14]
            network_params.append(Dzw_ab_state)
            Dzw_ab_output = jax.random.uniform(key=keys[9], shape=(self.nz, self.ny), minval=-1, maxval=1, dtype=jnp.float32)  # Dzw_ab_output = params[-13]
            network_params.append(Dzw_ab_output)
        else:
            self.physical_param_idx = -13
            keys = jax.random.split(key, 8)

        A = jax.random.uniform(key=keys[0], shape=(self.nx, self.nx), minval=-1e-3, maxval=1e-3, dtype=jnp.float32)
        network_params.append(A)  # A = params[-12]
        Bu = jax.random.uniform(key=keys[1], shape=(self.nx, self.nu), minval=-1e-3, maxval=1e-3, dtype=jnp.float32)
        network_params.append(Bu)  # Bu = params[-11]
        Bw_b = jnp.eye(self.nx, dtype=jnp.float32)
        network_params.append(Bw_b)  # Bw_b = params[-10]
        Bw_a = jax.random.uniform(key=keys[2], shape=(self.nx, self.nw), minval=-1e-3, maxval=1e-3, dtype=jnp.float32)
        network_params.append(Bw_a)  # Bw_a = params[-9]
        Cy = jax.random.uniform(key=keys[3], shape=(self.ny, self.nx), minval=-1e-3, maxval=1e-3, dtype=jnp.float32)
        network_params.append(Cy)  # Cy = params[-8]
        Dyu = jax.random.uniform(key=keys[4], shape=(self.ny, self.nu), minval=-1e-3, maxval=1e-3, dtype=jnp.float32)
        network_params.append(Dyu)  # Dyu = params[-7]
        Dyw_b = jnp.eye(self.ny, dtype=jnp.float32)
        network_params.append(Dyw_b)  # Dyw_b = params[-6]
        Dyw_a = jax.random.uniform(key=keys[5], shape=(self.ny, self.nw), minval=-1e-3, maxval=1e-3, dtype=jnp.float32)
        network_params.append(Dyw_a)  # Dyw_a = params[-5]
        Cz_b = jnp.vstack((jnp.eye(self.nx, dtype=jnp.float32), jnp.zeros(shape=(self.nu, self.nx), dtype=jnp.float32)))
        network_params.append(Cz_b)  # Cz_b = params[-4]
        Cz_a = jax.random.uniform(key=keys[6], shape=(self.nz, self.nx), minval=-1, maxval=1, dtype=jnp.float32)
        network_params.append(Cz_a)  # Cz_a = params[-3]
        Dzu_b = jnp.vstack((jnp.zeros(shape=(self.nx, self.nu), dtype=jnp.float32), jnp.eye(self.nu, dtype=jnp.float32)))
        network_params.append(Dzu_b)  # Dzu_b = params[-2]
        Dzu_a = jax.random.uniform(key=keys[7], shape=(self.nz, self.nu), minval=-1, maxval=1, dtype=jnp.float32)
        network_params.append(Dzu_a)  # Dzu_a = params[-1]
        self._init(params=network_params, x0=x0)

    def _create_jitted_model_step(self, known_sys: Any, hidden_layers: int, activation: str,
                                  ) -> Callable[[Array, Array, list[Array]], Tuple[Array, Array]]:
        """Creates JIT-compiled model step (combined state transition and output map) for static LFR-based model augmentation."""

        learning_component = generate_simple_ann(hidden_layers, activation)

        @jax.jit
        def model_step(x, u, params):

            # calculate wb zb
            phys_params = self.get_physical_params(params)
            zb = params[-4] @ x + params[-2] @ u  # z_b = Cz_b @ x + Dzu_b @ u
            w_b_state = (known_sys.f(zb[:self.nx] * self.std_x + self.mu_x, zb[self.nx:] * self.std_u + self.mu_u,
                                     phys_params) - self.mu_x) / self.std_x
            w_b_output = (known_sys.h(zb[:self.nx] * self.std_x + self.mu_x, zb[self.nx:] * self.std_u + self.mu_u,
                                      phys_params) - self.mu_y) / self.std_y

            # calculate za and wa
            if self.Dzw_structure is None:
                za = params[-3] @ x + params[-1] @ u  # z_a = Cz_a @ x + Dzu_a @ u
            else:  # now only lower-triangular Dzw structure is supported
                za = params[-3] @ x + params[-1] @ u + params[-14] @ w_b_state + params[
                    -13] @ w_b_output  # z_a = Cz_a @ x + Dzu_a @ u + Dzw_ab @ wb

            # ANN learning component
            w_a = learning_component(za, params)

            # calculate state transition
            x_plus = params[-12] @ x + params[-11] @ u + params[-10] @ w_b_state + params[-9] @ w_a  # x+ = A @ x + Bu @ u + Bw_b @ w_b + Bw_a @ w_a

            # calculate output
            y = params[-8] @ x + params[-7] @ u + params[-6] @ w_b_output + params[-5] @ w_a  # y = Cy @ x + Dyu @ u + Dyw_b @ w_b + Dyw_a @ w_a
            return x_plus, y

        return model_step

    def _add_group_lasso_z(self, tau_z):

        @jax.jit
        def group_lasso_reg(th):
            cost = 0.
            Cz_a = th[-3]
            Dzu_a = th[-1]
            ANN_params = self.get_network_params(th)
            W0 = ANN_params[0]
            if self.Dzw_structure is not None:  # now only lower-triangular Dzw structure is supported
                Dzw_ab_state = th[-14]
                Dzw_ab_output = th[-13]
            for i in range(self.nz):
                cost += tau_z * jnp.sqrt(jnp.sum(Cz_a[i, :]**2) + jnp.sum(Dzu_a[i, :]**2) + jnp.sum(W0[:, i]**2))
                if self.Dzw_structure is not None:
                    cost += tau_z * jnp.sqrt(jnp.sum(Dzw_ab_state[i, :]**2) + jnp.sum(Dzw_ab_output[:, i]**2))
            return cost
        return group_lasso_reg

    def _add_group_lasso_w(self, tau_w):

        @jax.jit
        def group_lasso_reg(th):
            cost = 0.
            ANN_params = self.get_network_params(th)
            W_last = ANN_params[-2]
            b_last = ANN_params[-1]
            Bw_a = th[-9]
            Dyw_a = th[-5]
            for i in range(self.nw):
                cost += tau_w * jnp.sqrt(
                    jnp.sum(W_last[i, :] ** 2) + b_last[i] ** 2 + jnp.sum(Bw_a[:, i] ** 2) + jnp.sum(
                        Dyw_a[:, i] ** 2))
            return cost
        return group_lasso_reg

    def _add_lfr_mx_l1_reg(self, tau: float, reg_coeffs: Optional[Array]) -> Callable[[list[Array]], float]:
        raise NotImplementedError("L1 regularization on LFR matrices is currently only implemented for the fully"
                                  "parametrized case (with the well-posedness conditions).")


########################################################################################################################
#                                  STATIC WELL-POSED LFR-BASED STRUCTURE
########################################################################################################################
class StaticWellPosedLFRAugmentation(AugmentationBase):
    """
    Static LFR-based model augmentation structure with guaranteed well-posedness parametrization.
    """
    def __init__(self,
                 known_sys: Any,
                 hidden_layers: int,
                 nodes_per_layer: int,
                 activation: str,
                 nz: int,
                 nw: int,
                 lipschitz_const: float,
                 x0: Optional[Union[Array, list[Array]]] = None,
                 seed: Union[int, list[int]] = 42,
                 std_x: Optional[np.ndarray] = None,
                 std_u: Optional[np.ndarray] = None,
                 std_y: Optional[np.ndarray] = None,
                 mu_x: Optional[np.ndarray] = None,
                 mu_u: Optional[np.ndarray] = None,
                 mu_y: Optional[np.ndarray] = None,
                 fpi_n_max: int = 100,
                 fpi_tol: float = 1e-3,
                 mask_params: Optional[list[Array]] = None,
                 mask_eps: float = 1e-4,
                 ) -> None:
        """
        Initializes the model structure.

        Parameters
        ----------
        known_sys: Any
            Baseline model (first-principle model) to be augmented.
        hidden_layers : int
            Number of hidden layers in the ANN.
        nodes_per_layer : int
            Neurons per hidden layer in the ANN.
        activation : str
            Activation function for the ANN.
        nz : int
            Dimension of the latent variable z_a.
        nw : int
            Dimension of the latent variable w_a.
        lipschitz_const : float
            Lipschitz constant of the baseline model (with normalization terms).
        x0 : array or list of arrays, optional
            Initial state(s).
        seed : int or list, optional
            Initialization seed(s).
        std_x, std_u, std_y : ndarray, optional
            Standard deviations for normalization.
        mu_x, mu_u, mu_y : ndarray, optional
            Means for normalization.
        fpi_n_max : int, optional
            Maximum iteration for the Fixed-Point Iterations during model evaluation. Defaults to 100.
        fpi_tol : float, optional
            Tolerance for the Fixed-Point Iterations during model evaluation. Defaults to 1e-3.
        mask_params : list of ndarrays, optional
            Tuned parameters from a previous model training. Elements that are approximately zero in mask_params are kept
            as zeros in the new model instance. If None, no masking is applied. Defaults to None.
        mask_eps : float, optional
            Threshold to determine which variables are zeroed out based on the masking parameters. Parameters smaller than
            this in absolute value are viewed as 0. Defaults to 1e-4.
        """
        self.nz = nz
        self.nw = nw
        self.fpi_n_max = fpi_n_max
        self.fpi_tol = fpi_tol
        self.lipschitz_const = lipschitz_const
        self.Dzw_dim1 = nz + known_sys.nx + known_sys.nu
        self.Dzw_dim2 = nw + known_sys.nx + known_sys.ny
        self.n = max(self.Dzw_dim1, self.Dzw_dim2)
        if mask_params is not None:
            self.W_mask = self.create_LFR_matrix_mask(mask_params, mask_eps)
        else:
            self.W_mask = None
        self.model_step_with_iter_count = None
        super().__init__(known_sys=known_sys, hidden_layers=hidden_layers, nodes_per_layer=nodes_per_layer,
                         activation=activation, x0=x0, seed=seed, std_y=std_y, std_x=std_x, std_u=std_u, mu_y=mu_y,
                         mu_x=mu_x, mu_u=mu_u)

    def simulate(self,
                 U: Union[Array, list[Array]],
                 X0: Optional[Union[Array, list[Array]]] = None,
                 ) -> Tuple[Union[Array, list[Array]], Union[Array, list[Array]], Union[Array, list[Array]], Union[Array, list[Array]]]:
        """
        Simulate the response of an augmented model structure on the test data.

        Parameters
        ----------
        U : ndarray or list of ndarrays
            Input signals for the data set. Must be N-by-nu shaped array or a list of Ni-by-nu shaped arrays.
        X0 : ndarray or list of ndarrays, optional
            Initial state vector for each data set. If None, simulations are started from x0=0.

        Returns
        -------
        Y : ndarray or list of ndarrays
            Simulated output trajectory/trajectories. Shape is (N, ny) for a single trajectory or a list of arrays
            with shape (Ni, ny).
        X : ndarray or list of ndarrays
             Simulated state trajectory/trajectories. Shape is (N, nx) for a single trajectory or a list of arrays with
             shape (Ni, nx).
        iter_counter : ndarray or list of ndararys
            Contains how many fixed-point iterations were needed to evaluate the model at each time step.
        fpi_residuals : ndarray or list of ndararys
            Contains the (approximated) residuals of the fixed-point iterations needed to evaluate the model at each time step.
        """
        # Scale input data and init. state
        if isinstance(U, list):
            N_meas = len(U)
            U_scaled = [(ui - self.mu_u) / self.std_u for ui in U]
            if X0 is None:
                x0_scaled = [(np.zeros(self.nx) - self.mu_x) / self.std_x] * N_meas
            else:
                x0_scaled = [(x0i - self.mu_x) / self.std_x for x0i in X0]
        else:
            N_meas = 1
            U_scaled = (U.copy() - self.mu_u) / self.std_u
            if X0 is None:
                x0_scaled = (np.zeros(self.nx) - self.mu_x) / self.std_x
            else:
                x0_scaled = (X0.copy() - self.mu_x) / self.std_x

        def model_step_fixed_params(x, u):
            x_next, y, iter_num, residual = self.model_step_with_iter_count(x, u, self.params)
            y = jnp.hstack((iter_num, residual, y, x))
            x_next = x_next.reshape(-1)
            return x_next, y

        if N_meas == 1:
            _, YX = jax.lax.scan(model_step_fixed_params, x0_scaled.reshape(-1), vec_reshape(U_scaled))
            iter_counter = YX[:, 0]
            fpi_residuals = YX[:, 1]
            Y = YX[:, 2:2+self.ny] * self.std_y + self.mu_y
            X = YX[:, 2+self.ny:] * self.std_x + self.mu_x
        else:
            Y = []
            X = []
            iter_counter = []
            fpi_residuals = []
            for i in range(N_meas):
                _, YX = jax.lax.scan(model_step_fixed_params, x0_scaled[i].reshape(-1), vec_reshape(U_scaled[i]))
                iter_counter.append(YX[:, 0])
                fpi_residuals.append(YX[:, 1])
                Y.append(YX[:, 2:2+self.ny] * self.std_y + self.mu_y)
                X.append(YX[:, 2+self.ny:] * self.std_x + self.mu_x)
        return Y, X, iter_counter, fpi_residuals

    def sparsity_analysis(self) -> Tuple[int, int]:
        """
        Provides a sparsity analysis of LFR matrices.

        Returns
        -------
        z_reduction : int
            The number of redundant dimensions in the latent variable z_a.
        w_reduction : int
            The number of redundant dimensions in the latent variable w_a.
        """
        th = self.params
        ANN_params = self.get_network_params(th)

        Cz_a = np.array(th[-7])
        Dzu_a = np.array(th[-5])
        Bw_a = np.array(th[-13])
        Dyw_a = np.array(th[-9])

        if self.Dzw_dim1 == self.Dzw_dim2:
            D_bar = simple_cayley(th[-4], th[-3])
            Dzw = nn.sigmoid(th[-1]) * D_bar / self.lipschitz_const
        elif self.Dzw_dim1 > self.Dzw_dim2:
            D_bar = general_cayley(th[-4], th[-3], th[-4])
            Dzw = nn.sigmoid(th[-1]) * D_bar / self.lipschitz_const
        else:
            D_bar = general_cayley(th[-4], th[-3], th[-2])
            Dzw = nn.sigmoid(th[-1]) * D_bar.T / self.lipschitz_const
        Dzw_ab_aa = np.array(Dzw[self.nx + self.nu:, :])
        Dzw_ba_aa = np.array(Dzw[:, self.nx + self.ny:])

        W0 = np.array(ANN_params[0])
        W_last = np.array(ANN_params[-2])
        b_last = np.array(ANN_params[-1])

        # zero-out coefficients smaller than self.zero_coeff
        Cz_a[np.abs(Cz_a) <= self.zero_coeff] = 0.
        Dzu_a[np.abs(Dzu_a) <= self.zero_coeff] = 0.
        Bw_a[np.abs(Bw_a) <= self.zero_coeff] = 0.
        Dyw_a[np.abs(Dyw_a) <= self.zero_coeff] = 0.
        Dzw_ab_aa[np.abs(Dzw_ab_aa) <= self.zero_coeff] = 0.
        Dzw_ba_aa[np.abs(Dzw_ba_aa) <= self.zero_coeff] = 0.
        W0[np.abs(W0) <= self.zero_coeff] = 0.
        W_last[np.abs(W_last) <= self.zero_coeff] = 0.
        b_last[np.abs(b_last) <= self.zero_coeff] = 0.

        z_reduction = 0
        w_reduction = 0
        print("Sparsity analysis results:")

        for i in range(self.nz):
            if (np.max(np.abs(Cz_a[i, :])) <= self.zero_coeff and np.max(np.abs(Dzu_a[i, :])) <= self.zero_coeff and
                np.max(np.abs(W0[:, i])) <= self.zero_coeff and np.max(np.abs(Dzw_ab_aa[i, :])) <= self.zero_coeff):
                print(f"z_{i+1} can be eliminated")
                z_reduction += 1

        for i in range(self.nw):
            if (np.max(np.abs(W_last[i, :])) <= self.zero_coeff and np.abs(b_last[i]) <= self.zero_coeff and
                np.max(np.abs(Bw_a[:, i])) <= self.zero_coeff and np.max(np.abs(Dyw_a[:, i])) <= self.zero_coeff and
                np.max(np.abs(Dzw_ba_aa[:, i])) <= self.zero_coeff):
                print(f"w_{i + 1} can be eliminated")
                w_reduction += 1

        return z_reduction, w_reduction

    def _initialize_parameters(self, known_sys: Any, hidden_layers: int, nodes_per_layer: int,
                               x0: Optional[Union[Array, list[Array]]], seed: int, activation: str) -> None:
        """Initializes the parameters of the static well-posed LFR-based augmentation structure."""
        key = jax.random.key(seed)
        key_net, key_params = jax.random.split(key, 2)

        # init. network parameters for static additive case
        network_params = initialize_network(input_features=self.nz, output_features=self.nw, hidden_layers=hidden_layers,
                                            nodes_per_layer=nodes_per_layer, key=key_net, act_fun=activation)

        # add physical parameters to optimization variables (if necessary)
        if self.tune_physical_params:
            network_params.append(known_sys.init_params)  # theta = params[-17]
        self.physical_param_idx = -17

        n_D = max(self.Dzw_dim1, self.Dzw_dim2)
        m_D = min(self.Dzw_dim1, self.Dzw_dim2)

        keys = jax.random.split(key_params, 11)
        # generate matrix structure for initialization
        A = jax.random.uniform(key=keys[0], shape=(self.nx, self.nx), minval=-1e-3, maxval=1e-3, dtype=jnp.float64)
        Bu = jax.random.uniform(key=keys[1], shape=(self.nx, self.nu), minval=-1e-3, maxval=1e-3, dtype=jnp.float64)
        Bw_b = jnp.hstack((jnp.eye(self.nx, dtype=jnp.float64), jnp.zeros((self.nx, self.ny), dtype=jnp.float64)))
        Bw_a = jax.random.uniform(key=keys[2], shape=(self.nx, self.nw), minval=-1e-3, maxval=1e-3, dtype=jnp.float64)
        Cy = jax.random.uniform(key=keys[3], shape=(self.ny, self.nx), minval=-1e-3, maxval=1e-3, dtype=jnp.float64)
        Dyu = jax.random.uniform(key=keys[4], shape=(self.ny, self.nu), minval=-1e-3, maxval=1e-3, dtype=jnp.float64)
        Dyw_b = jnp.hstack((jnp.zeros((self.ny, self.nx), dtype=jnp.float64), jnp.eye(self.ny, dtype=jnp.float64)))
        Dyw_a = jax.random.uniform(key=keys[5], shape=(self.ny, self.nw), minval=-1e-3, maxval=1e-3, dtype=jnp.float64)
        Cz_b = jnp.vstack((jnp.eye(self.nx, dtype=jnp.float64), jnp.zeros(shape=(self.nu, self.nx), dtype=jnp.float64)))
        Cz_a = jax.random.uniform(key=keys[6], shape=(self.nz, self.nx), minval=-1, maxval=1, dtype=jnp.float64)
        Dzu_b = jnp.vstack((jnp.zeros(shape=(self.nx, self.nu), dtype=jnp.float64), jnp.eye(self.nu, dtype=jnp.float64)))
        Dzu_a = jax.random.uniform(key=keys[7], shape=(self.nz, self.nu), minval=-1, maxval=1, dtype=jnp.float64)
        X_D = jax.random.uniform(key=keys[8], shape=(m_D, m_D), minval=-1., maxval=1., dtype=jnp.float64)
        Y_D = jax.random.uniform(key=keys[9], shape=(m_D, m_D), minval=-1., maxval=1., dtype=jnp.float64)
        Z_D = jax.random.uniform(key=keys[10], shape=(n_D - m_D, m_D), minval=-1., maxval=1., dtype=jnp.float64)
        d_D = jnp.array([-5.], dtype=jnp.float64)

        if self.W_mask is not None:
            A *= self.W_mask["A"]
            Bu *= self.W_mask["Bu"]
            Bw_b *= self.W_mask["Bw_b"]
            Bw_a *= self.W_mask["Bw_a"]
            Cy *= self.W_mask["Cy"]
            Dyu *= self.W_mask["Dyu"]
            Dyw_b *= self.W_mask["Dyw_b"]
            Dyw_a *= self.W_mask["Dyw_a"]
            Cz_b *= self.W_mask["Cz_b"]
            Cz_a *= self.W_mask["Cz_a"]
            Dzu_b *= self.W_mask["Dzu_b"]
            Dzu_a *= self.W_mask["Dzu_a"]

        # add interconnection matrix to optimized variables
        network_params.append(A)  # A = params[-16]
        network_params.append(Bu)  # Bu = params[-15]
        network_params.append(Bw_b)  # Bw_b = params[-14]
        network_params.append(Bw_a)  # Bw_a = params[-13]
        network_params.append(Cy)  # Cy = params[-12]
        network_params.append(Dyu)  # Dyu = params[-11]
        network_params.append(Dyw_b)  # Dyw_b = params[-10]
        network_params.append(Dyw_a)  # Dyw_a = params[-9]
        network_params.append(Cz_b)  # Cz_b = params[-8]
        network_params.append(Cz_a)  # Cz_a = params[-7]
        network_params.append(Dzu_b)  # Dzu_b = params[-6]
        network_params.append(Dzu_a)  # Dzu_a = params[-5]
        network_params.append(X_D)  # X_D = params[-4]
        network_params.append(Y_D)  # Y_D = params[-3]
        network_params.append(Z_D)  # Z_D = params[-2]
        network_params.append(d_D)  # d_D = params[-1]
        self._init(params=network_params, x0=x0)

    def _create_jitted_model_step(self, known_sys: Any, hidden_layers: int, activation: str,
                                  ) -> Callable[[Array, Array, list[Array]], Tuple[Array, Array]]:
        """Creates JIT-compiled state transition and output functions according to the given augmentation structure."""

        learning_component = generate_simple_ann(hidden_layers, activation)

        @jax.jit
        def nonlinear_components(z, params):
            zb = z[:self.nx + self.nu]
            za = z[self.nx + self.nu:]
            zb_x = zb[:self.nx]
            zb_u = zb[self.nx:]
            phys_params = self.get_physical_params(params)

            x_plus = (known_sys.f(zb_x * self.std_x + self.mu_x, zb_u * self.std_u + self.mu_u, phys_params) - self.mu_x) / self.std_x
            y = (known_sys.h(zb_x * self.std_x + self.mu_x, zb_u * self.std_u + self.mu_u, phys_params) - self.mu_y) / self.std_y

            w_a = learning_component(za, params)

            return jnp.concatenate((x_plus, y, w_a))

        @jax.jit
        def contractive_map(z, x, u, params):
            # 1-Lipschitz D_bar, then scaling with L
            if self.Dzw_dim1 == self.Dzw_dim2:
                D_bar = simple_cayley(params[-4], params[-3])
                Dzw = nn.sigmoid(params[-1]) * D_bar / self.lipschitz_const
            elif self.Dzw_dim1 > self.Dzw_dim2:
                D_bar = general_cayley(params[-4], params[-3], params[-2])
                Dzw = nn.sigmoid(params[-1]) * D_bar / self.lipschitz_const
            else:
                D_bar = general_cayley(params[-4], params[-3], params[-2])
                Dzw = nn.sigmoid(params[-1]) * D_bar.T / self.lipschitz_const

            if self.W_mask is None:
                zb_feedthrough = params[-8] @ x + params[-6] @ u  # zb(x,u) = Cz_b @ x + Dzu_b @ u
                za_feedthrough = params[-7] @ x + params[-5] @ u  # za(x,u) = Cz_a @ x + Dzu_a @ u
                z_feedthrough = jnp.hstack((zb_feedthrough, za_feedthrough))
                z_next = Dzw @ nonlinear_components(z, params) + z_feedthrough
            else:
                zb_feedthrough = (self.W_mask["Cz_b"] * params[-8]) @ x + (self.W_mask["Dzu_b"] * params[-6]) @ u  # zb(x,u) = Cz_b @ x + Dzu_b @ u
                za_feedthrough = (self.W_mask["Cz_a"] * params[-7]) @ x + (self.W_mask["Dzu_a"] * params[-5]) @ u  # za(x,u) = Cz_a @ x + Dzu_a @ u
                z_feedthrough = jnp.hstack((zb_feedthrough, za_feedthrough))
                z_next = (self.W_mask["Dzw"] * Dzw) @ nonlinear_components(z, params) + z_feedthrough
            return z_next

        fpi = jaxopt.FixedPointIteration(fixed_point_fun=contractive_map, maxiter=self.fpi_n_max, implicit_diff=True,
                                         tol=self.fpi_tol)

        @jax.jit
        def model_step_with_iter_count(x, u, params):
            # f : (nx+nu) --> (nx)
            # h : (nx+nu) --> (ny)

            z0 = jnp.concatenate((x, u, jnp.zeros(self.nz, dtype=jnp.float64)))
            z_star, fpi_state = fpi.run(z0, x, u, params)
            iter_num = fpi_state.iter_num
            residual = fpi_state.error

            w = nonlinear_components(z_star, params)
            wb = w[:self.nx + self.ny]
            wa = w[self.nx + self.ny:]
            if self.W_mask is None:
                x_next = params[-16] @ x + params[-15] @ u + params[-14] @ wb + params[-13] @ wa  # x+ = A @ x + Bu @ u + Bw_b @ wb + Bw_a @ wa
                y = params[-12] @ x + params[-11] @ u + params[-10] @ wb + params[-9] @ wa  # y = Cy @ x + Dyu @ y + Dyw_b @ wb + Dyw_a @ wa
            else:
                x_next = ((self.W_mask["A"] * params[-16]) @ x + (self.W_mask["Bu"] * params[-15]) @ u +
                          (self.W_mask["Bw_b"] * params[-14]) @ wb + (self.W_mask["Bw_a"] * params[
                            -13]) @ wa)  # x+ = A @ x + Bu @ u + Bw_b @ wb + Bw_a @ wa
                y = ((self.W_mask["Cy"] * params[-12]) @ x + (self.W_mask["Dyu"] * params[-11]) @ u +
                     (self.W_mask["Dyw_b"] * params[-10]) @ wb + (self.W_mask["Dyw_a"] * params[-9]) @ wa)  # y = Cy @ x + Dyu @ y + Dyw_b @ wb + Dyw_a @ wa

            return x_next, y, iter_num, residual

        @jax.jit
        def model_step(x, u, params):
            x_next, y, iter_num, residual = self.model_step_with_iter_count(x, u, params)
            return x_next, y

        self.model_step_with_iter_count = model_step_with_iter_count
        return model_step

    def _add_lfr_mx_l1_reg(self, tau: float, reg_coeffs: Optional[Array]) -> Callable[[list[Array]], float]:
        if reg_coeffs is None:
            W_dim = 0
            for i in range(5, 17):
                W_dim += self.params[-i].reshape(-1).shape[0]
            W_dim += self.Dzw_dim1 * self.Dzw_dim2
            reg_coeffs = np.ones(shape=(W_dim,), dtype=np.float64)

        @jax.jit
        def LFR_matrix_l1_reg(params):
            A = params[-16].reshape(-1)
            Bu = params[-15].reshape(-1)
            Bw_b = params[-14].reshape(-1)
            Bw_a = params[-13].reshape(-1)
            Cy = params[-12].reshape(-1)
            Dyu = params[-11].reshape(-1)
            Dyw_b = params[-10].reshape(-1)
            Dyw_a = params[-9].reshape(-1)
            Cz_b = params[-8].reshape(-1)
            Cz_a = params[-7].reshape(-1)
            Dzu_b = params[-6].reshape(-1)
            Dzu_a = params[-5].reshape(-1)
            if self.Dzw_dim1 == self.Dzw_dim2:
                D_bar = simple_cayley(params[-4], params[-3])
                Dzw = nn.sigmoid(params[-1]) * D_bar / self.lipschitz_const
            elif self.Dzw_dim1 > self.Dzw_dim2:
                D_bar = general_cayley(params[-4], params[-3], params[-2])
                Dzw = nn.sigmoid(params[-1]) * D_bar / self.lipschitz_const
            else:
                D_bar = general_cayley(params[-4], params[-3], params[-2])
                Dzw = nn.sigmoid(params[-1]) * D_bar.T / self.lipschitz_const
            W_vec = jnp.hstack((A, Bu, Bw_b, Bw_a, Cy, Dyu, Dyw_b, Dyw_a, Cz_b, Cz_a, Dzu_b, Dzu_a, Dzw.reshape(-1)))
            return tau * jnp.sum(jnp.abs(W_vec * reg_coeffs))
        return LFR_matrix_l1_reg

    def _add_group_lasso_z(self, tau: float) -> Callable[[list[Array]], float]:
        @jax.jit
        def group_lasso_fun(th):
            cost = 0.
            Cz_a = th[-7]
            Dzu_a = th[-5]
            ANN_params = self.get_network_params(th)
            W0 = ANN_params[0]
            if self.Dzw_dim1 == self.Dzw_dim2:
                D_bar = simple_cayley(th[-4], th[-3])
                Dzw = nn.sigmoid(th[-1]) * D_bar / self.lipschitz_const
            elif self.Dzw_dim1 > self.Dzw_dim2:
                D_bar = general_cayley(th[-4], th[-3], th[-4])
                Dzw = nn.sigmoid(th[-1]) * D_bar / self.lipschitz_const
            else:
                D_bar = general_cayley(th[-4], th[-3], th[-2])
                Dzw = nn.sigmoid(th[-1]) * D_bar.T / self.lipschitz_const
            Dzw_ab_aa = Dzw[self.nx + self.nu:, :]
            for i in range(self.nz):
                cost += tau * jnp.sqrt(jnp.sum(Cz_a[i, :] ** 2) + jnp.sum(Dzu_a[i, :] ** 2) + jnp.sum(W0[:, i] ** 2) +
                                         jnp.sum(Dzw_ab_aa[i, :] ** 2))
            return cost
        return group_lasso_fun

    def _add_group_lasso_w(self, tau: float) -> Callable[[list[Array]], float]:
        @jax.jit
        def group_lasso_fun(th):
            cost = 0.
            ANN_params = self.get_network_params(th)
            W_last = ANN_params[-2]
            b_last = ANN_params[-1]
            Bw_a = th[-13]
            Dyw_a = th[-9]
            if self.Dzw_dim1 == self.Dzw_dim2:
                D_bar = simple_cayley(th[-4], th[-3])
                Dzw = nn.sigmoid(th[-1]) * D_bar / self.lipschitz_const
            elif self.Dzw_dim1 > self.Dzw_dim2:
                D_bar = general_cayley(th[-4], th[-3], th[-4])
                Dzw = nn.sigmoid(th[-1]) * D_bar / self.lipschitz_const
            else:
                D_bar = general_cayley(th[-4], th[-3], th[-2])
                Dzw = nn.sigmoid(th[-1]) * D_bar.T / self.lipschitz_const
            Dzw_ba_aa = Dzw[:, self.nx+self.ny:]
            for i in range(self.nw):
                cost += tau * jnp.sqrt(jnp.sum(W_last[i, :] ** 2) + b_last[i] ** 2 + jnp.sum(Bw_a[:, i] ** 2) +
                                         jnp.sum(Dyw_a[:, i] ** 2) + jnp.sum(Dzw_ba_aa[:, i] ** 2))
            return cost
        return group_lasso_fun

    def create_LFR_matrix_mask(self, th, eps):
        # TODO: implement in sub-classes as well
        raise NotImplementedError

    def save_LFR_matrices(self, filename):
        raise NotImplementedError

    def compute_new_l1_reg_weights(self, eps=1e-4):
        raise NotImplementedError


########################################################################################################################
#                                  STATIC CONTRACTING LFR-BASED STRUCTURE
########################################################################################################################
class StaticContractingLFRAugmentation(StaticWellPosedLFRAugmentation):
    """
    Static LFR-based model augmentation structure with guaranteed well-posedness and contraction parametrization.
    """
    def __init__(self,
                 known_sys: Any,
                 hidden_layers: int,
                 nodes_per_layer: int,
                 activation: str,
                 nz: int,
                 nw: int,
                 lipschitz_const: float,
                 x0: Optional[Union[Array, list[Array]]] = None,
                 seed: Union[int, list[int]] = 42,
                 std_x: Optional[np.ndarray] = None,
                 std_u: Optional[np.ndarray] = None,
                 std_y: Optional[np.ndarray] = None,
                 mu_x: Optional[np.ndarray] = None,
                 mu_u: Optional[np.ndarray] = None,
                 mu_y: Optional[np.ndarray] = None,
                 fpi_n_max: int = 100,
                 fpi_tol: float = 1e-3,
                 contraction_rate: float = 1.,
                 mask_params: Optional[list[Array]] = None,
                 mask_eps: float = 1e-4,
                 ) -> None:
        """
        Initializes the model structure.

        Parameters
        ----------
        known_sys: Any
            Baseline model (first-principle model) to be augmented.
        hidden_layers : int
            Number of hidden layers in the ANN.
        nodes_per_layer : int
            Neurons per hidden layer in the ANN.
        activation : str
            Activation function for the ANN.
        nz : int
            Dimension of the latent variable z_a.
        nw : int
            Dimension of the latent variable w_a.
        lipschitz_const : float
            Lipschitz constant of the baseline model (with normalization terms).
        x0 : array or list of arrays, optional
            Initial state(s).
        seed : int or list, optional
            Initialization seed(s).
        std_x, std_u, std_y : ndarray, optional
            Standard deviations for normalization.
        mu_x, mu_u, mu_y : ndarray, optional
            Means for normalization.
        fpi_n_max : int, optional
            Maximum iteration for the Fixed-Point Iterations during model evaluation. Defaults to 100.
        fpi_tol : float, optional
            Tolerance for the Fixed-Point Iterations during model evaluation. Defaults to 1e-3.
        contraction_rate : float, optional
            Maximum of the contraction rate between (0, 1]. The parametrization ensures that the tru contraction rate is
            less than this provided value. Defaults to 1.
        mask_params : list of ndarrays, optional
            Tuned parameters from a previous model training. Elements that are approximately zero in mask_params are kept
            as zeros in the new model instance. If None, no masking is applied. Defaults to None.
        mask_eps : float, optional
            Threshold to determine which variables are zeroed out based on the masking parameters. Parameters smaller than
            this in absolute value are viewed as 0. Defaults to 1e-4.
        """
        self.Bw_dim1 = known_sys.nx
        self.Bw_dim2 = known_sys.nx + known_sys.ny + nw
        self.Cz_dim1 = nz + known_sys.nu + known_sys.nx
        self.Cz_dim2 = known_sys.nx
        self.contraction_rate = contraction_rate
        if self.contraction_rate <= 0. or self.contraction_rate > 1.:
            raise ValueError("contraction_rate must be between 0 and 1")
        super().__init__(known_sys=known_sys, hidden_layers=hidden_layers, nodes_per_layer=nodes_per_layer,
                         activation=activation, nz=nz, nw=nw, lipschitz_const=lipschitz_const, x0=x0, seed=seed,
                         std_y=std_y, std_x=std_x, std_u=std_u, mu_y=mu_y, mu_x=mu_x, mu_u=mu_u, fpi_n_max=fpi_n_max,
                         fpi_tol=fpi_tol, mask_params=mask_params, mask_eps=mask_eps)

    def _initialize_parameters(self, known_sys: Any, hidden_layers: int, nodes_per_layer: int,
                               x0: Optional[Union[Array, list[Array]]], seed: int, activation: str) -> None:
        """Initializes the parameters of the static contracting LFR-based augmentation structure."""
        key = jax.random.key(seed)
        key_net, key_params = jax.random.split(key, 2)

        # init. network parameters for static additive case
        network_params = initialize_network(input_features=self.nz, output_features=self.nw, hidden_layers=hidden_layers,
                                            nodes_per_layer=nodes_per_layer, key=key_net, act_fun=activation)

        # add physical parameters to optimization variables (if necessary)
        if self.tune_physical_params:
            network_params.append(known_sys.init_params)  # theta = params[-23]
        self.physical_param_idx = -23

        n_B = max(self.Bw_dim1, self.Bw_dim2)
        m_B = min(self.Bw_dim1, self.Bw_dim2)
        n_C = self.Cz_dim1
        m_C = self.Cz_dim2
        n_D = max(self.Dzw_dim1, self.Dzw_dim2)
        m_D = min(self.Dzw_dim1, self.Dzw_dim2)

        keys = jax.random.split(key_params, 16)

        # generate matrix structure for initialization
        d_D = jnp.array([-5.], dtype=jnp.float64)
        X_A = jax.random.uniform(key=keys[0], shape=(self.nx, self.nx), minval=-1, maxval=1, dtype=jnp.float64)
        Y_A = jax.random.uniform(key=keys[1], shape=(self.nx, self.nx), minval=-1, maxval=1, dtype=jnp.float64)
        Bu = jax.random.uniform(key=keys[2], shape=(self.nx, self.nu), minval=-1e-3, maxval=1e-3, dtype=jnp.float64)
        X_B = jax.random.uniform(key=keys[3], shape=(m_B, m_B), minval=-1, maxval=1, dtype=jnp.float64)
        Y_B = jax.random.uniform(key=keys[4], shape=(m_B, m_B), minval=-1, maxval=1, dtype=jnp.float64)
        Z_B = jax.random.uniform(key=keys[5], shape=(n_B-m_B, m_B), minval=-1e-3, maxval=1e-3, dtype=jnp.float64)
        Cy = jax.random.uniform(key=keys[6], shape=(self.ny, self.nx), minval=-1e-3, maxval=1e-3, dtype=jnp.float64)
        Dyu = jax.random.uniform(key=keys[7], shape=(self.ny, self.nu), minval=-1e-3, maxval=1e-3, dtype=jnp.float64)
        Dyw_b = jnp.hstack((jnp.zeros((self.ny, self.nx), dtype=jnp.float64), jnp.eye(self.ny, dtype=jnp.float64)))
        Dyw_a = jax.random.uniform(key=keys[8], shape=(self.ny, self.nw), minval=-1e-3, maxval=1e-3, dtype=jnp.float64)
        X_C = jax.random.uniform(key=keys[9], shape=(m_C, m_C), minval=-1, maxval=1, dtype=jnp.float64)
        Y_C = jax.random.uniform(key=keys[10], shape=(m_C, m_C), minval=-1, maxval=1, dtype=jnp.float64)
        Z_C = jax.random.uniform(key=keys[11], shape=(n_C-m_C, m_C), minval=-1e-3, maxval=1e-3, dtype=jnp.float64)
        Dzu_b = jnp.vstack((jnp.zeros(shape=(self.nx, self.nu), dtype=jnp.float64), jnp.eye(self.nu, dtype=jnp.float64)))
        Dzu_a = jax.random.uniform(key=keys[12], shape=(self.nz, self.nu), minval=-1, maxval=1, dtype=jnp.float64)
        X_D = jax.random.uniform(key=keys[13], shape=(m_D, m_D), minval=-1., maxval=1., dtype=jnp.float64)
        Y_D = jax.random.uniform(key=keys[14], shape=(m_D, m_D), minval=-1., maxval=1., dtype=jnp.float64)
        Z_D = jax.random.uniform(key=keys[15], shape=(n_D - m_D, m_D), minval=-1., maxval=1., dtype=jnp.float64)
        alpha = jnp.array([-2.], dtype=jnp.float64)
        beta = jnp.array([0.], dtype=jnp.float64)
        gamma = jnp.array([-4.], dtype=jnp.float64)

        if self.W_mask is not None:
            Bu *= self.W_mask["Bu"]
            Cy *= self.W_mask["Cy"]
            Dyu *= self.W_mask["Dyu"]
            Dyw_b *= self.W_mask["Dyw_b"]
            Dyw_a *= self.W_mask["Dyw_a"]
            Dzu_b *= self.W_mask["Dzu_b"]
            Dzu_a *= self.W_mask["Dzu_a"]

        # add interconnection matrices to optimized variables
        network_params.append(d_D)  # d_D = params[-22]
        network_params.append(X_A)  # X_A = params[-21]
        network_params.append(Y_A)  # Y_B = params[-20]
        network_params.append(Bu)  # Bu = params[-19]
        network_params.append(X_B)  # X_B = params[-18]
        network_params.append(Y_B)  # Y_B = params[-17]
        network_params.append(Z_B)  # Z_B = params[-16]
        network_params.append(Cy)  # Cy = params[-15]
        network_params.append(Dyu)  # Dyu = params[-14]
        network_params.append(Dyw_b)  # Dyw_b = params[-13]
        network_params.append(Dyw_a)  # Dyw_a = params[-12]
        network_params.append(X_C)  # X_C = params[-11]
        network_params.append(Y_C)  # Y_C = params[-10]
        network_params.append(Z_C)  # Z_C = params[-9]
        network_params.append(Dzu_b)  # Dzu_b = params[-8]
        network_params.append(Dzu_a)  # Dzu_a = params[-7]
        network_params.append(X_D)  # X_D = params[-6]
        network_params.append(Y_D)  # Y_D = params[-5]
        network_params.append(Z_D)  # Z_D = params[-4]
        network_params.append(alpha)  # alpha = params[-3]
        network_params.append(beta)  # beta = params[-2]
        network_params.append(gamma)  # gamma = params[-1]
        self._init(params=network_params, x0=x0)

    def _create_jitted_model_step(self, known_sys: Any, hidden_layers: int, activation: str,
                                  ) -> Callable[[Array, Array, list[Array]], Tuple[Array, Array]]:
        """Creates JIT-compiled state transition and output functions according to the given augmentation structure."""

        learning_component = generate_simple_ann(hidden_layers, activation)

        @ jax.jit
        def nonlinear_components(z, params):
            zb = z[:self.nx+self.nu]
            za = z[self.nx+self.nu:]
            zb_x = zb[:self.nx]
            zb_u = zb[self.nx:]
            phys_params = self.get_physical_params(params)

            x_plus = (known_sys.f(zb_x * self.std_x + self.mu_x, zb_u * self.std_u + self.mu_u, phys_params) - self.mu_x) / self.std_x
            y = (known_sys.h(zb_x * self.std_x + self.mu_x, zb_u * self.std_u + self.mu_u, phys_params) - self.mu_y) / self.std_y

            w_a = learning_component(za, params)

            return jnp.concatenate((x_plus, y, w_a))

        @jax.jit
        def contractive_map(z, x, u, params):
            # 1-Lipschitz D_bar, then scaling with L
            if self.Dzw_dim1 == self.Dzw_dim2:
                D_bar = simple_cayley(params[-6], params[-5])
                Dzw = nn.sigmoid(params[-22]) * D_bar / self.lipschitz_const
            elif self.Dzw_dim1 > self.Dzw_dim2:
                D_bar = general_cayley(params[-6], params[-5], params[-4])
                Dzw = nn.sigmoid(params[-22]) * D_bar / self.lipschitz_const
            else:
                D_bar = general_cayley(params[-6], params[-5], params[-4])
                Dzw = nn.sigmoid(params[-22]) * D_bar.T / self.lipschitz_const

            C_bar = general_cayley(params[-11], params[-10], params[-9])
            # scaling factor sigma_C
            kappa = self.lipschitz_const / (1 - self.lipschitz_const * jnp.linalg.norm(Dzw, 2))
            kappa_sqrt = jnp.sqrt((1 - nn.sigmoid(params[-3])) / kappa)
            sigma_C = (1 / jnp.exp(params[-2])) * kappa_sqrt - nn.sigmoid(params[-1]) / (kappa * jnp.exp(params[-2]) * kappa_sqrt)
            Cz = C_bar * sigma_C * jnp.sqrt(self.contraction_rate)  # transformation + scaling with Lipschitz const.
            Cz_b = Cz[:(self.nx+self.nu), :]
            Cz_a = Cz[(self.nx+self.nu):, :]

            if self.W_mask is None:
                zb_feedthrough = Cz_b @ x + params[-8] @ u  # zb(x,u) = Cz_b @ x + Dzu_b @ u
                za_feedthrough = Cz_a @ x + params[-7] @ u  # za(x,u) = Cz_a @ x + Dzu_a @ u
                z_feedthrough = jnp.hstack((zb_feedthrough, za_feedthrough))
                z_next = Dzw @ nonlinear_components(z, params) + z_feedthrough
            else:
                zb_feedthrough = (self.W_mask["Cz_b"] * Cz_b) @ x + (self.W_mask["Dzu_b"] * params[-8]) @ u  # zb(x,u) = Cz_b @ x + Dzu_b @ u
                za_feedthrough = (self.W_mask["Cz_a"] * Cz_a) @ x + (self.W_mask["Dzu_a"] * params[-7]) @ u  # za(x,u) = Cz_a @ x + Dzu_a @ u
                z_feedthrough = jnp.hstack((zb_feedthrough, za_feedthrough))
                z_next = (self.W_mask["D_zw"] * Dzw) @ nonlinear_components(z, params) + z_feedthrough
            return z_next

        fpi = jaxopt.FixedPointIteration(fixed_point_fun=contractive_map, maxiter=self.fpi_n_max, implicit_diff=True,
                                         tol=self.fpi_tol)

        @jax.jit
        def model_step_with_iter_count(x, u, params):
            # f : (nx+nu) --> (nx)
            # h : (nx+nu) --> (ny)

            z0 = jnp.concatenate((x, u, jnp.zeros(self.nz, dtype=jnp.float64)))
            z_star, fpi_state = fpi.run(z0, x, u, params)
            iter_num = fpi_state.iter_num
            residual = fpi_state.error

            w = nonlinear_components(z_star, params)
            wb = w[:self.nx + self.ny]
            wa = w[self.nx + self.ny:]

            A_bar = simple_cayley(params[-21], params[-20])
            A = nn.sigmoid(params[-3]) * A_bar * self.contraction_rate  # scaling

            if self.Dzw_dim1 == self.Dzw_dim2:
                D_bar = simple_cayley(params[-6], params[-5])
                Dzw = nn.sigmoid(params[-22]) * D_bar / self.lipschitz_const
            elif self.Dzw_dim1 > self.Dzw_dim2:
                D_bar = general_cayley(params[-6], params[-5], params[-4])
                Dzw = nn.sigmoid(params[-22]) * D_bar / self.lipschitz_const
            else:
                D_bar = general_cayley(params[-6], params[-5], params[-4])
                Dzw = nn.sigmoid(params[-22]) * D_bar.T / self.lipschitz_const
                # scaling factor sigma_B
            kappa = self.lipschitz_const / (1 - self.lipschitz_const * jnp.linalg.norm(Dzw, 2))
            kappa_sqrt = jnp.sqrt((1 - nn.sigmoid(params[-3])) / kappa)

            B_bar = general_cayley(params[-18], params[-17], params[-16])
            Bw = B_bar.T * jnp.exp(params[-2]) * kappa_sqrt * jnp.sqrt(self.contraction_rate)
            Bw_b = Bw[:, :(self.nx + self.ny)]
            Bw_a = Bw[:, (self.nx + self.ny):]

            if self.W_mask is None:
                x_next = A @ x + params[-19] @ u + Bw_b @ wb + Bw_a @ wa  # x+ = A @ x + Bu @ u + Bw_b @ wb +  Bw_a @ wa
                y = params[-15] @ x + params[-14] @ u + params[-13] @ wb + params[-12] @ wa  # y = Cy @ x + Dyu @ y + Dyw_b @ wb + Dyw @ wa
            else:
                x_next = ((self.W_mask["A"] * A) @ x + (self.W_mask["Bu"] * params[-19]) @ u +
                          (self.W_mask["Bw_b"] * Bw_b) @ wb + (self.W_mask["Bw_a"] * Bw_a) @ wa)  # x+ = A @ x + Bu @ u + Bw_b @ wb + Bw_a @ wa
                y = ((self.W_mask["Cy"] * params[-15]) @ x + (self.W_mask["Dyu"] * params[-14]) @ u +
                     (self.W_mask["Dyw_b"] * params[-13]) @ wb + (self.W_mask["Dyw_a"] * params[-12]) @ wa)  # y = Cy @ x + Dyu @ y + Dyw_b @ wb + Dyw_a @ wa

            return x_next, y, iter_num, residual

        @jax.jit
        def model_step(x, u, params):
            x_next, y, iter_num, residual = self.model_step_with_iter_count(x, u, params)
            return x_next, y

        self.model_step_with_iter_count = model_step_with_iter_count
        return model_step

    def create_LFR_matrix_mask(self, th, eps):
        raise NotImplementedError

    def save_LFR_matrices(self, filename):
        raise NotImplementedError

    def compute_new_l1_reg_weights(self, eps=1e-4):
        raise NotImplementedError

    def sparsity_analysis(self):
        raise NotImplementedError

    def _add_lfr_mx_l1_reg(self, tau: float, reg_coeffs: Optional[Array]) -> Callable[[list[Array]], float]:
        raise NotImplementedError

    def _add_group_lasso_z(self, tau: float) -> Callable[[list[Array]], float]:
        raise NotImplementedError

    def _add_group_lasso_w(self, tau: float) -> Callable[[list[Array]], float]:
        raise NotImplementedError


########################################################################################################################
#                                       DYNAMIC LFR-BASED STRUCTURE
########################################################################################################################
class DynamicLFRAugmentation(AugmentationBase):
    def __init__(self,
                 known_sys: Any,
                 n_augm_states: int,
                 hidden_layers: int,
                 nodes_per_layer: int,
                 activation: str,
                 nz_a: int,
                 nw_a: int,
                 x0: Optional[Union[Array, list[Array]]] = None,
                 seed: Union[int, list[int]] = 42,
                 std_x: Optional[np.ndarray] = None,
                 std_u: Optional[np.ndarray] = None,
                 std_y: Optional[np.ndarray] = None,
                 mu_x: Optional[np.ndarray] = None,
                 mu_u: Optional[np.ndarray] = None,
                 mu_y: Optional[np.ndarray] = None,
                 Dzw_structure: Optional[str] = None,
                 ) -> None:
        """
        Initialize the static LFR-based model augmentation structure.

        Parameters
        ----------
        known_sys : object
            Baseline (first-principles) model.
        n_augm_states : int
            Number of augmented (hidden) states beyond the baseline states.
        hidden_layers : int
            Number of hidden layers in the ANN.
        nodes_per_layer : int
            Neurons per hidden layer in the ANN.
        activation : str
            Activation function for the ANN.
        nz_a : int
            Dimension of the latent variable z_a.
        nw_a : int
            Dimension of the latent variable w_a.
        x0 : array or list of arrays, optional
            Initial state(s).
        seed : int or list, optional
            Initialization seed(s).
        std_x, std_u, std_y : ndarray, optional
            Standard deviations for normalization.
        mu_x, mu_u, mu_y : ndarray, optional
            Means for normalization.
        Dzw_structure : str, optional
            If "lower", the $D_{zw}$ matrix is implemented as a strictly lower triangular matrix, if "upper", than a
            strictly lower triangular matrix. If None, $D_{zw}\equiv 0$ is applied.
        """
        self.nz = nz_a
        self.nw = nw_a
        if n_augm_states <= 0: raise ValueError(
            "n_augm_states must be > 0. For augmentation with no additional states, apply a static structure.")
        self.nx_b = known_sys.nx
        self.nx_a = n_augm_states
        self.Dzw_structure = Dzw_structure

        # combined normalization constants for base and augmented states (augmented states assumed to be zero-mean and std. of 1)
        self.std_xb = np.ones(known_sys.nx) if std_x is None else std_x
        self.mu_xb = np.zeros(known_sys.nx) if mu_x is None else mu_x
        std_x = np.hstack((self.std_xb.copy(), np.ones(self.nx_a)))
        mu_x = np.hstack((self.mu_xb.copy(), np.zeros(self.nx_a)))

        super().__init__(known_sys=known_sys, hidden_layers=hidden_layers, nodes_per_layer=nodes_per_layer,
                         activation=activation, x0=x0, seed=seed, std_y=std_y, std_x=std_x, std_u=std_u, mu_y=mu_y,
                         mu_x=mu_x, mu_u=mu_u)
        self.nx = known_sys.nx + n_augm_states

    def sparsity_analysis(self) -> Tuple[int, int, int]:
        """
        Provides a sparsity analysis of the LFR matrix.

        Returns
        -------
        z_reduction : int
            Redundant dimensions along latent variable z_a.
        w_reduction : int
            Redundant dimensions along latent variable w_a.
        xa_reduction : int
            Redundant dimensions along the added augmented states x_a.
        """
        th = self.params
        ANN_params = self.get_network_params(th)

        A_ab = np.array(th[-19])
        A_ba = np.array(th[-20])
        A_aa = np.array(th[-18])
        Bu_a = np.array(th[-16])
        Bw_ab = np.array(th[-13])
        Bw_aa = np.array(th[-12])
        Bw_ba = np.array(th[-14])
        Cy_a = np.array(th[-10])
        Cz_ab = np.array(th[-4])
        Cz_aa = np.array(th[-3])
        Cz_ba = np.array(th[-5])
        Dzu_a = np.array(th[-1])
        Dyw_a = np.array(th[-7])

        W0 = np.array(ANN_params[0])
        W_last = np.array(ANN_params[-2])
        b_last = np.array(ANN_params[-1])

        # zero-out coefficients smaller than self.zero_coeff
        A_ab[np.abs(A_ab) <= self.zero_coeff] = 0.
        A_ba[np.abs(A_ba) <= self.zero_coeff] = 0.
        A_aa[np.abs(A_aa) <= self.zero_coeff] = 0.
        Bu_a[np.abs(Bu_a) <= self.zero_coeff] = 0.
        Bw_ab[np.abs(Bw_ab) <= self.zero_coeff] = 0.
        Bw_aa[np.abs(Bw_aa) <= self.zero_coeff] = 0.
        Bw_ba[np.abs(Bw_ba) <= self.zero_coeff] = 0.
        Cy_a[np.abs(Cy_a) <= self.zero_coeff] = 0.
        Cz_ab[np.abs(Cz_ab) <= self.zero_coeff] = 0.
        Cz_aa[np.abs(Cz_aa) <= self.zero_coeff] = 0.
        Cz_ba[np.abs(Cz_ba) <= self.zero_coeff] = 0.
        Dzu_a[np.abs(Dzu_a) <= self.zero_coeff] = 0.
        Dyw_a[np.abs(Dyw_a) <= self.zero_coeff] = 0.
        W0[np.abs(W0) <= self.zero_coeff] = 0.
        W_last[np.abs(W_last) <= self.zero_coeff] = 0.
        b_last[np.abs(b_last) <= self.zero_coeff] = 0.

        if self.Dzw_structure == "lower":
            Dzw_ab = np.array(th[-22])
            Dzw_ab[np.abs(Dzw_ab) <= self.zero_coeff] = 0.

        z_reduction = 0
        w_reduction = 0
        xa_reduction = 0
        print("Sparsity analysis results:")

        for i in range(self.nz):
            if (np.max(np.abs(Cz_ab[i, :])) <= self.zero_coeff and np.max(np.abs(Cz_aa[i, :])) <= self.zero_coeff and np.max(np.abs(Dzu_a[i, :])) <= self.zero_coeff and
                    np.max(np.abs(W0[:, i])) <= self.zero_coeff):
                if self.Dzw_structure == "lower":
                    if np.max(np.abs(Dzw_ab[i, :])) <= self.zero_coeff:
                        print(f"z_{i + 1} can be eliminated")
                        z_reduction += 1
                else:
                    print(f"z_{i + 1} can be eliminated")
                    z_reduction += 1

        for i in range(self.nw):
            if (np.max(np.abs(W_last[i, :])) <= self.zero_coeff and np.abs(b_last[i]) <= self.zero_coeff and
                    np.max(np.abs(Bw_ba[:, i])) <= self.zero_coeff and np.max(np.abs(Bw_aa[:, i])) <= self.zero_coeff and
                    np.max(np.abs(Dyw_a[:, i])) <= self.zero_coeff):
                print(f"w_{i + 1} can be eliminated")
                w_reduction += 1

        for i in range(self.nx_a):
            if (np.max(np.abs(A_ab[i, :])) <= self.zero_coeff and np.max(np.abs(A_aa[i, :])) <= self.zero_coeff and
                    np.max(np.abs(Bu_a[i, :])) <= self.zero_coeff and np.max(np.abs(Bw_ab[i, :])) <= self.zero_coeff and
                    np.max(np.abs(Bw_aa[i, :])) <= self.zero_coeff and np.max(np.abs(Cz_ba[:, i])) <= self.zero_coeff and
                    np.max(np.abs(Cz_aa[:, i])) <= self.zero_coeff and np.max(np.abs(Cy_a[:, i])) <= self.zero_coeff and
                    np.max(np.abs(A_ba[:, i])) <= self.zero_coeff and np.max(np.abs(A_aa[:, i])) <= self.zero_coeff):
                print(f"xa_{i + 1} can be eliminated")
                xa_reduction += 1

        return z_reduction, w_reduction, xa_reduction

    def _initialize_parameters(self, known_sys: Any, hidden_layers: int, nodes_per_layer: int,
                               x0: Optional[Union[Array, list[Array]]], seed: int, activation: str) -> None:
        """Initializes the parameters according to the given augmentation structure."""

        key = jax.random.key(seed)
        key_net, key_params = jax.random.split(key, 2)

        if x0 is None:
            x0_init = np.zeros(self.nx_b + self.nx_a)
        else:
            x0_init = x0.copy()
            if isinstance(x0_init, list):
                for i in range(len(x0_init)):
                    x0_init[i] = np.concatenate((x0_init[i], np.zeros(self.nx_a)))
            else:
                x0_init = np.concatenate((x0_init, np.zeros(self.nx_a)))

        # init. network parameters for static additive case
        network_params = initialize_network(input_features=self.nz, output_features=self.nw,
                                            hidden_layers=hidden_layers, nodes_per_layer=nodes_per_layer, key=key_net,
                                            act_fun=activation)

        nx_a = self.nx_a
        nx_b = self.nx_b

        # add physical parameters to optimization variables (if necessary)
        if self.tune_physical_params:
            network_params.append(known_sys.init_params)  # theta = params[-22] OR if Dzw_structure == 'lower' params[-23]

        if self.Dzw_structure == 'lower':
            keys = jax.random.split(key_params, 17)
            Dzw_ab = jax.random.uniform(key=keys[16], shape=(self.nz, nx_b + self.ny), minval=-1e-3, maxval=1e-3,
                                        dtype=jnp.float64)
            self.physical_param_idx = -23
        else:
            keys = jax.random.split(key_params, 16)
            self.physical_param_idx = -22

        # generate matrix structure for initialization
        A_bb = jnp.zeros((nx_b, nx_b), dtype=jnp.float64)
        A_ba = jnp.zeros((nx_b, nx_a), dtype=jnp.float64)
        A_ab = jax.random.uniform(key=keys[2], shape=(nx_a, nx_b), minval=-1, maxval=1, dtype=jnp.float64)
        A_aa = jax.random.uniform(key=keys[3], shape=(nx_a, nx_a), minval=-1, maxval=1, dtype=jnp.float64)

        Bu_b = jnp.zeros((nx_b, self.nu), dtype=jnp.float64)
        Bu_a = jax.random.uniform(key=keys[5], shape=(nx_a, self.nu), minval=-1, maxval=1, dtype=jnp.float64)

        Bw_bb = jnp.hstack((jnp.eye(nx_b, dtype=jnp.float64), jnp.zeros((nx_b, self.ny), dtype=jnp.float64)))
        Bw_ba = jnp.zeros((nx_b, self.nw), dtype=jnp.float64)
        Bw_ab = jnp.zeros((nx_a, nx_b + self.ny), dtype=jnp.float64)
        Bw_aa = jax.random.uniform(key=keys[8], shape=(nx_a, self.nw), minval=-1, maxval=1, dtype=jnp.float64)

        Cy_b = jnp.zeros((self.ny, nx_b), dtype=jnp.float64)
        Cy_a = jnp.zeros((self.ny, nx_a), dtype=jnp.float64)
        Dyu = jnp.zeros((self.ny, self.nu), dtype=jnp.float64)

        Dyw_b = jnp.hstack((jnp.zeros((self.ny, nx_b), dtype=jnp.float64), jnp.eye(self.ny, dtype=jnp.float64)))
        Dyw_a = jnp.zeros((self.ny, self.nw), dtype=jnp.float64)

        Cz_bb = jnp.vstack((jnp.eye(nx_b, dtype=jnp.float64), jnp.zeros(shape=(self.nu, nx_b), dtype=jnp.float64)))
        Cz_ba = jnp.zeros((nx_b + self.nu, nx_a), dtype=jnp.float64)
        Cz_ab = jax.random.uniform(key=keys[13], shape=(self.nz, nx_b), minval=-1, maxval=1, dtype=jnp.float64)
        Cz_aa = jax.random.uniform(key=keys[14], shape=(self.nz, nx_a), minval=-1, maxval=1, dtype=jnp.float64)

        Dzu_b = jnp.vstack((jnp.zeros(shape=(nx_b, self.nu), dtype=jnp.float64), jnp.eye(self.nu, dtype=jnp.float64)))
        Dzu_a = jax.random.uniform(key=keys[15], shape=(self.nz, self.nu), minval=-1, maxval=1, dtype=jnp.float64)

        # add interconnection matrix to optimized variables
        if self.Dzw_structure == 'lower':
            network_params.append(Dzw_ab)  # Dzw_ab = params[-22]
        network_params.append(A_bb)  # A_bb = params[-21]
        network_params.append(A_ba)  # A_ba = params[-20]
        network_params.append(A_ab)  # A_ab = params[-19]
        network_params.append(A_aa)  # A_aa = params[-18]
        network_params.append(Bu_b)  # Bu_b = params[-17]
        network_params.append(Bu_a)  # Bu_a = params[-16]
        network_params.append(Bw_bb)  # Bw_bb = params[-15]
        network_params.append(Bw_ba)  # Bw_ba = params[-14]
        network_params.append(Bw_ab)  # Bw_ab = params[-13]
        network_params.append(Bw_aa)  # Bw_aa = params[-12]
        network_params.append(Cy_b)  # Cy_b = params[-11]
        network_params.append(Cy_a)  # Cy_a = params[-10]
        network_params.append(Dyu)  # Dyu = params[-9]
        network_params.append(Dyw_b)  # Dyw_b = params[-8]
        network_params.append(Dyw_a)  # Dyw_a = params[-7]
        network_params.append(Cz_bb)  # Cz_bb = params[-6]
        network_params.append(Cz_ba)  # Cz_ba = params[-5]
        network_params.append(Cz_ab)  # Cz_ab = params[-4]
        network_params.append(Cz_aa)  # Cz_aa = params[-3]
        network_params.append(Dzu_b)  # Dzu_b = params[-2]
        network_params.append(Dzu_a)  # Dzu_a = params[-1]
        self._init(params=network_params, x0=x0_init)

    def _create_jitted_model_step(self, known_sys: Any, hidden_layers: int, activation: str,
                                  ) -> Callable[[Array, Array, list[Array]], Tuple[Array, Array]]:
        """Creates JIT-compiled state transition and output functions according to the given augmentation structure."""

        learning_component = generate_simple_ann(hidden_layers, activation)

        @jax.jit
        def model_step(x, u, params):
            # f : (nx+nu) --> (nx)
            # h : (nx+nu) --> (ny)

            xb = x[:self.nx_b]
            xa = x[self.nx_b:]

            # zb = Cz_bb @ xb + Cz_ba @ xa + Dzu_b @ u
            zb = params[-6] @ xb + params[-5] @ xa + params[-2] @ u

            zb_x = zb[:self.nx_b]
            zb_u = zb[self.nx_b:]
            phys_params = self.get_physical_params(params)
            x_plus_fp = (known_sys.f(zb_x * self.std_xb + self.mu_xb, zb_u * self.std_u + self.mu_u, phys_params) - self.mu_xb) / self.std_xb
            y_fp = (known_sys.h(zb_x * self.std_xb + self.mu_xb, zb_u * self.std_u + self.mu_u, phys_params) - self.mu_y) / self.std_y
            wb = jnp.concatenate((x_plus_fp, y_fp))

            # za = Cz_ab @ xb + Cz_aa @ xa + Dzu_a @ u
            za = params[-4] @ xb + params[-3] @ xa + params[-1] @ u

            if self.Dzw_structure == "lower":
                za += params[-22] @ wb

            wa = learning_component(za, params)

            # xb+ = A_bb @ xb + A_ba @ xa + Bu_b @ u + Bw_bb @ wb + Bw_ba @ wa
            xb_next = params[-21] @ xb + params[-20] @ xa + params[-17] @ u + params[-15] @ wb + params[-14] @ wa

            # xa+ = A_ab @ xb + A_aa @ xa + Bu_a @ u + Bw_ab @ wb + Bw_aa @ wa
            xa_next = params[-19] @ xb + params[-18] @ xa + params[-16] @ u + params[-13] @ wb + params[-12] @ wa
            x_next = jnp.hstack((xb_next, xa_next))

            # y = Cy_b @ xb + Cy_a @ xa + Dyu @ u + Dyw_b @ wb + Dyw_a @ wa
            y = params[-11] @ xb + params[-10] @ xa + params[-9] @ u + params[-8] @ wb + params[-7] @ wa

            return x_next, y

        return model_step

    def _add_group_lasso_z(self, tau: float) -> Callable[[list[Array]], float]:
        @jax.jit
        def group_lasso_fun(th):
            cost = 0.
            Cz_ab = th[-4]
            Cz_aa = th[-3]
            Dzu_a = th[-1]
            if self.Dzw_structure == "lower":
                Dzw_ab = th[-22]
            ANN_params = self.get_network_params(th)
            W0 = ANN_params[0]
            b0 = ANN_params[1]
            for i in range(self.nz):
                cost += jnp.sum(Cz_ab[i, :] ** 2) + jnp.sum(Cz_aa[i, :] ** 2) + jnp.sum(Dzu_a[i, :] ** 2) + \
                        jnp.sum(W0[:, i] ** 2) + b0[i] ** 2
                if self.Dzw_structure == "lower":
                    cost += jnp.sum(Dzw_ab[i, :] ** 2)
            return tau * jnp.sqrt(cost)
        return group_lasso_fun

    def _add_group_lasso_w(self, tau: float) -> Callable[[list[Array]], float]:
        @jax.jit
        def group_lasso_fun(th):
            cost = 0.
            ANN_params = self.get_network_params(th)
            W_last = ANN_params[-2]
            b_last = ANN_params[-1]
            Bw_ba = th[-14]
            Bw_aa = th[-12]
            Dyw_a = th[-7]
            for i in range(self.nw):
                cost += tau * jnp.sqrt(jnp.sum(W_last[i, :] ** 2) + b_last[i] ** 2 + jnp.sum(Bw_ba[:, i] ** 2) +
                                         jnp.sum(Bw_aa[:, i] ** 2) + jnp.sum(Dyw_a[:, i] ** 2))
            return cost
        return group_lasso_fun

    def _add_group_lasso_x(self, tau: float) -> Callable[[list[Array], list[Array]], float]:
        @jax.jit
        def group_lasso_fun(th, x0):
            cost = 0.
            A_ab = th[-19]
            A_ba = th[-20]
            A_aa = th[-18]
            Bu_a = th[-16]
            Bw_ab = th[-13]
            Bw_aa = th[-12]
            Cy_a = th[-10]
            Cz_ba = th[-5]
            Cz_aa = th[-3]

            for i in range(self.nx_a):
                cost += tau * jnp.sqrt(jnp.sum(A_ab[i, :] ** 2) + jnp.sum(A_aa[i, :] ** 2) + jnp.sum(Bu_a[i, :] ** 2) +
                    jnp.sum(Bw_ab[i, :] ** 2) + jnp.sum(Bw_aa[i, :] ** 2) + sum([x0i[self.nx_b+i]**2 for x0i in x0]) +
                    jnp.sum(Cz_ba[:, i] ** 2) + jnp.sum(Cz_aa[:, i] ** 2) + jnp.sum(Cy_a[:, i] ** 2) +
                    jnp.sum(A_ba[:, i] ** 2) + jnp.sum(A_aa[:, i] ** 2) - A_aa[i, i] ** 2)
            return cost
        return group_lasso_fun

    def _add_lfr_mx_l1_reg(self, tau: float, reg_coeffs: Optional[Array]) -> Callable[[list[Array]], float]:
        raise NotImplementedError("L1 regularization-based augmentation structure discovery is only implemented for the full (well-posed) parametrization!")


########################################################################################################################
#                                    DYNAMIC WELL-POSED LFR-BASED STRUCTURE
########################################################################################################################
class DynamicWellPosedAugmentation(StaticWellPosedLFRAugmentation):
    def __init__(self,
                 known_sys: Any,
                 n_augm_states: int,
                 hidden_layers: int,
                 nodes_per_layer: int,
                 activation: str,
                 nz_a: int,
                 nw_a: int,
                 lipschitz_const : float,
                 x0: Optional[Union[Array, list[Array]]] = None,
                 seed: Union[int, list[int]] = 42,
                 std_x: Optional[np.ndarray] = None,
                 std_u: Optional[np.ndarray] = None,
                 std_y: Optional[np.ndarray] = None,
                 mu_x: Optional[np.ndarray] = None,
                 mu_u: Optional[np.ndarray] = None,
                 mu_y: Optional[np.ndarray] = None,
                 fpi_n_max: int = 100,
                 fpi_tol: float = 1e-3,
                 mask_params: Optional[list[Array]] = None,
                 mask_eps: float = 1e-4,
                 ) -> None:
        if n_augm_states <= 0: raise ValueError(
            "n_augm_states must be > 0. For augmentation with no additional states, apply a static structure.")
        self.nx_b = known_sys.nx
        self.nx_a = n_augm_states

        # combined normalization constants for base and augmented states (augmented states assumed to be zero-mean and std. of 1)
        self.std_xb = np.ones(known_sys.nx) if std_x is None else std_x
        self.mu_xb = np.zeros(known_sys.nx) if mu_x is None else mu_x
        std_x = np.hstack((self.std_xb.copy(), np.ones(self.nx_a)))
        mu_x = np.hstack((self.mu_xb.copy(), np.zeros(self.nx_a)))
        super().__init__(known_sys=known_sys, hidden_layers=hidden_layers, nodes_per_layer=nodes_per_layer,
                         activation=activation, nz=nz_a, nw=nw_a, lipschitz_const=lipschitz_const, x0=x0, seed=seed,
                         std_x=std_x, std_u=std_u, std_y=std_y, mu_x=mu_x, mu_u=mu_u, mu_y=mu_y, fpi_n_max=fpi_n_max,
                         fpi_tol=fpi_tol, mask_params=mask_params, mask_eps=mask_eps)
        self.nx = known_sys.nx + n_augm_states

    def sparsity_analysis(self) -> Tuple[int, int, int]:
        """
        Provides a sparsity analysis of LFR matrices.

        Returns
        -------
        z_reduction : int
            The number of redundant dimensions in the latent variable z_a.
        w_reduction : int
            The number of redundant dimensions in the latent variable w_a.
        xa_reduction : int
            The number of redundant dimensions in the augmented states x_a.
        """
        th = self.params
        ANN_params = self.get_network_params(th)

        A_ab = np.array(th[-23])
        A_ba = np.array(th[-24])
        A_aa = np.array(th[-22])
        Bu_a = np.array(th[-20])
        Bw_ab = np.array(th[-17])
        Bw_aa = np.array(th[-16])
        Bw_ba = np.array(th[-18])
        Cy_a = np.array(th[-14])
        Cz_ab = np.array(th[-8])
        Cz_aa = np.array(th[-7])
        Cz_ba = np.array(th[-9])
        Dzu_a = np.array(th[-5])
        Dyw_a = np.array(th[-11])

        if self.Dzw_dim1 == self.Dzw_dim2:
            D_bar = simple_cayley(th[-4], th[-3])
            Dzw = nn.sigmoid(th[-1]) * D_bar / self.Lipscitz_const
        elif self.Dzw_dim1 > self.Dzw_dim2:
            D_bar = general_cayley(th[-4], th[-3], th[-2])
            Dzw = nn.sigmoid(th[-1]) * D_bar / self.Lipscitz_const
        else:
            D_bar = general_cayley(th[-4], th[-3], th[-2])
            Dzw = nn.sigmoid(th[-1]) * D_bar.T / self.Lipscitz_const
        Dzw_ab_aa = np.array(Dzw[self.nx_b + self.nu:, :])
        Dzw_ba_aa = np.array(Dzw[:, self.nx_b + self.ny:])

        W0 = np.array(ANN_params[0])
        W_last = np.array(ANN_params[-2])
        b_last = np.array(ANN_params[-1])

        # zero-out coefficients smaller than self.zero_coeff
        A_ab[np.abs(A_ab) <= self.zero_coeff] = 0.
        A_ba[np.abs(A_ba) <= self.zero_coeff] = 0.
        A_aa[np.abs(A_aa) <= self.zero_coeff] = 0.
        Bu_a[np.abs(Bu_a) <= self.zero_coeff] = 0.
        Bw_ab[np.abs(Bw_ab) <= self.zero_coeff] = 0.
        Bw_aa[np.abs(Bw_aa) <= self.zero_coeff] = 0.
        Bw_ba[np.abs(Bw_ba) <= self.zero_coeff] = 0.
        Cy_a[np.abs(Cy_a) <= self.zero_coeff] = 0.
        Cz_ab[np.abs(Cz_ab) <= self.zero_coeff] = 0.
        Cz_aa[np.abs(Cz_aa) <= self.zero_coeff] = 0.
        Cz_ba[np.abs(Cz_ba) <= self.zero_coeff] = 0.
        Dzu_a[np.abs(Dzu_a) <= self.zero_coeff] = 0.
        Dyw_a[np.abs(Dyw_a) <= self.zero_coeff] = 0.
        # Dzw_ab_aa[np.abs(Dzw_ab_aa) <= self.zero_coeff] = 0.
        # Dzw_ba_aa[np.abs(Dzw_ba_aa) <= self.zero_coeff] = 0.
        # Temporarily disregard Dzw matrix until better solution (these values never reach 0)
        Dzw_ab_aa = np.zeros_like(Dzw_ba_aa)
        Dzw_ba_aa = np.zeros_like(Dzw_ba_aa)
        W0[np.abs(W0) <= self.zero_coeff] = 0.
        W_last[np.abs(W_last) <= self.zero_coeff] = 0.
        b_last[np.abs(b_last) <= self.zero_coeff] = 0.

        z_reduction = 0
        w_reduction = 0
        xa_reduction = 0
        print("Sparsity analysis results:")

        for i in range(self.nz):
            if (np.max(np.abs(Cz_ab[i, :])) <= self.zero_coeff and np.max(
                    np.abs(Cz_aa[i, :])) <= self.zero_coeff and np.max(np.abs(Dzu_a[i, :])) <= self.zero_coeff and
                    np.max(np.abs(W0[:, i])) <= self.zero_coeff and np.max(
                        np.abs(Dzw_ab_aa[i, :])) <= self.zero_coeff):
                print(f"z_{i + 1} can be eliminated")
                z_reduction += 1

        for i in range(self.nw):
            if (np.max(np.abs(W_last[i, :])) <= self.zero_coeff and np.abs(b_last[i]) <= self.zero_coeff and
                    np.max(np.abs(Bw_ba[:, i])) <= self.zero_coeff and np.max(
                        np.abs(Bw_aa[:, i])) <= self.zero_coeff and
                    np.max(np.abs(Dyw_a[:, i])) <= self.zero_coeff and np.max(
                        np.abs(Dzw_ba_aa[:, i])) <= self.zero_coeff):
                print(f"w_{i + 1} can be eliminated")
                w_reduction += 1

        for i in range(self.nx_a):
            if (np.max(np.abs(A_ab[i, :])) <= self.zero_coeff and np.max(np.abs(A_aa[i, :])) <= self.zero_coeff and
                    np.max(np.abs(Bu_a[i, :])) <= self.zero_coeff and np.max(
                        np.abs(Bw_ab[i, :])) <= self.zero_coeff and
                    np.max(np.abs(Bw_aa[i, :])) <= self.zero_coeff and np.max(
                        np.abs(Cz_ba[:, i])) <= self.zero_coeff and
                    np.max(np.abs(Cz_aa[:, i])) <= self.zero_coeff and np.max(
                        np.abs(Cy_a[:, i])) <= self.zero_coeff and
                    np.max(np.abs(A_ba[:, i])) <= self.zero_coeff and np.max(
                        np.abs(A_aa[:, i])) <= self.zero_coeff):
                print(f"xa_{i + 1} can be eliminated")
                xa_reduction += 1

        return z_reduction, w_reduction, xa_reduction

    def _initialize_parameters(self, known_sys: Any, hidden_layers: int, nodes_per_layer: int,
                              x0: Optional[Union[Array, list[Array]]], seed: int, activation: str) -> None:
        """Initializes the parameters according to the given augmentation structure."""
        if x0 is None:
            x0_init = np.zeros(self.nx_b+self.nx_a)
        else:
            x0_init = x0
            if isinstance(x0_init, list):
                for i in range(len(x0_init)):
                    x0_init[i] = np.concatenate((x0_init[i], np.zeros(self.nx_a)))
            else:
                x0_init = np.concatenate((x0_init, np.zeros(self.nx_a)))

        key = jax.random.key(seed)
        key_net, key_params = jax.random.split(key, 2)

        network_params = initialize_network(input_features=self.nz, output_features=self.nw, hidden_layers=hidden_layers,
                                            nodes_per_layer=nodes_per_layer, key=key_net, act_fun=activation)

        nx_a = self.nx_a
        nx_b = self.nx_b

        # add physical parameters to optimization variables (if necessary)
        self.physical_param_idx = -26
        if self.tune_physical_params:
            network_params.append(known_sys.init_params)  # theta = params[-26]

        n_D = max(self.Dzw_dim1, self.Dzw_dim2)
        m_D = min(self.Dzw_dim1, self.Dzw_dim2)

        keys = jax.random.split(key_params, 11)

        # generate matrix structure for initialization
        A_bb = jnp.zeros((nx_b, nx_b), dtype=jnp.float64)
        A_ba = jnp.zeros((nx_b, nx_a), dtype=jnp.float64)
        A_ab = jax.random.uniform(key=keys[0], shape=(nx_a, nx_b), minval=-1, maxval=1, dtype=jnp.float64)
        A_aa = jax.random.uniform(key=keys[1], shape=(nx_a, nx_a), minval=-1, maxval=1, dtype=jnp.float64)

        Bu_b = jnp.zeros((nx_b, self.nu), dtype=jnp.float64)
        Bu_a = jax.random.uniform(key=keys[2], shape=(nx_a, self.nu), minval=-1, maxval=1, dtype=jnp.float64)

        Bw_bb = jnp.hstack((jnp.eye(nx_b, dtype=jnp.float64), jnp.zeros((nx_b, self.ny), dtype=jnp.float64)))
        Bw_ba = jnp.zeros((nx_b, self.nw), dtype=jnp.float64)
        Bw_ab = jnp.zeros((nx_a, nx_b+self.ny), dtype=jnp.float64)
        Bw_aa = jax.random.uniform(key=keys[3], shape=(nx_a, self.nw), minval=-1, maxval=1, dtype=jnp.float64)

        Cy_b = jnp.zeros((self.ny, nx_b), dtype=jnp.float64)
        Cy_a = jnp.zeros((self.ny, nx_a), dtype=jnp.float64)
        Dyu =jnp.zeros((self.ny, self.nu), dtype=jnp.float64)

        Dyw_b = jnp.hstack((jnp.zeros((self.ny, nx_b), dtype=jnp.float64), jnp.eye(self.ny, dtype=jnp.float64)))
        Dyw_a = jnp.zeros((self.ny, self.nw), dtype=jnp.float64)

        Cz_bb = jnp.vstack((jnp.eye(nx_b, dtype=jnp.float64), jnp.zeros(shape=(self.nu, nx_b), dtype=jnp.float64)))
        Cz_ba = jnp.zeros((nx_b+self.nu,nx_a), dtype=jnp.float64)
        Cz_ab = jax.random.uniform(key=keys[4], shape=(self.nz, nx_b), minval=-1, maxval=1,dtype=jnp.float64)
        Cz_aa = jax.random.uniform(key=keys[5], shape=(self.nz, nx_a), minval=-1, maxval=1, dtype=jnp.float64)

        Dzu_b = jnp.vstack((jnp.zeros(shape=(nx_b, self.nu), dtype=jnp.float64), jnp.eye(self.nu, dtype=jnp.float64)))
        Dzu_a = jax.random.uniform(key=keys[6], shape=(self.nz, self.nu), minval=-1, maxval=1, dtype=jnp.float64)

        X_D = jax.random.uniform(key=keys[8], shape=(m_D, m_D), minval=-1., maxval=1., dtype=jnp.float64)
        Y_D = jax.random.uniform(key=keys[9], shape=(m_D, m_D), minval=-1., maxval=1., dtype=jnp.float64)
        Z_D = jax.random.uniform(key=keys[10], shape=(n_D - m_D, m_D), minval=-1., maxval=1., dtype=jnp.float64)
        d_D = jnp.array([-5.])

        if self.W_mask is not None:
            A_ab *= self.W_mask["A_ab"]
            A_aa *= self.W_mask["A_aa"]
            Bu_a *= self.W_mask["Bu_a"]
            Bw_bb *= self.W_mask["Bw_bb"]
            Bw_aa *= self.W_mask["Bw_aa"]
            Dyw_b *= self.W_mask["Dyw_b"]
            Cz_bb *= self.W_mask["Cz_bb"]
            Cz_ab *= self.W_mask["Cz_ab"]
            Cz_aa *= self.W_mask["Cz_aa"]
            Dzu_b *= self.W_mask["Dzu_b"]
            Dzu_a *= self.W_mask["Dzu_a"]

        # add interconnection matrix to optimized variables
        network_params.append(A_bb)  # A_bb = params[-25]
        network_params.append(A_ba)  # A_ba = params[-24]
        network_params.append(A_ab)  # A_ab = params[-23]
        network_params.append(A_aa)  # A_aa = params[-22]
        network_params.append(Bu_b)  # Bu_b = params[-21]
        network_params.append(Bu_a)  # Bu_a = params[-20]
        network_params.append(Bw_bb)  # Bw_bb = params[-19]
        network_params.append(Bw_ba)  # Bw_ba = params[-18]
        network_params.append(Bw_ab)  # Bw_ab = params[-17]
        network_params.append(Bw_aa)  # Bw_aa = params[-16]
        network_params.append(Cy_b)  # Cy_b = params[-15]
        network_params.append(Cy_a)  # Cy_a = params[-14]
        network_params.append(Dyu)  # Dyu = params[-13]
        network_params.append(Dyw_b)  # Dyw_b = params[-12]
        network_params.append(Dyw_a)  # Dyw_a = params[-11]
        network_params.append(Cz_bb)  # Cz_bb = params[-10]
        network_params.append(Cz_ba)  # Cz_ba = params[-9]
        network_params.append(Cz_ab)  # Cz_ab = params[-8]
        network_params.append(Cz_aa)  # Cz_aa = params[-7]
        network_params.append(Dzu_b)  # Dzu_b = params[-6]
        network_params.append(Dzu_a)  # Dzu_a = params[-5]
        network_params.append(X_D)  # X_D = params[-4]
        network_params.append(Y_D)  # Y_D = params[-3]
        network_params.append(Z_D)  # Z_D = params[-2]
        network_params.append(d_D)  # d_D = params[-1]

        self._init(params=network_params, x0=x0_init)

    def _create_jitted_model_step(self, known_sys: Any, hidden_layers: int, activation: str,
                                  ) -> Callable[[Array, Array, list[Array]], Tuple[Array, Array]]:
        """Creates JIT-compiled state transition and output functions according to the given augmentation structure."""

        learning_component = generate_simple_ann(hidden_layers, activation)

        @jax.jit
        def nonlinear_components(z, params):
            zb = z[:self.nx_b + self.nu]
            za = z[self.nx_b + self.nu:]
            zb_x = zb[:self.nx_b]
            zb_u = zb[self.nx_b:]
            phys_params = self.get_physical_params(params)

            x_plus = (known_sys.f(zb_x * self.std_xb + self.mu_xb, zb_u * self.std_u + self.mu_u,
                                  phys_params) - self.mu_xb) / self.std_xb
            y = (known_sys.h(zb_x * self.std_xb + self.mu_xb, zb_u * self.std_u + self.mu_u,
                             phys_params) - self.mu_y) / self.std_y

            w_a = learning_component(za, params)

            return jnp.concatenate((x_plus, y, w_a))

        @jax.jit
        def contractive_map(z, x, u, params):
            # 1-Lipschitz D_bar, then scaling with L
            if self.Dzw_dim1 == self.Dzw_dim2:
                D_bar = simple_cayley(params[-4], params[-3])
                Dzw = nn.sigmoid(params[-1]) * D_bar / self.lipschitz_const
            elif self.Dzw_dim1 > self.Dzw_dim2:
                D_bar = general_cayley(params[-4], params[-3], params[-2])
                Dzw = nn.sigmoid(params[-1]) * D_bar / self.lipschitz_const
            else:
                D_bar = general_cayley(params[-4], params[-3], params[-2])
                Dzw = nn.sigmoid(params[-1]) * D_bar.T / self.lipschitz_const

            xb = x[:self.nx_b]
            xa = x[self.nx_b:]

            if self.W_mask is None:
                zb_feedthrough = params[-10] @ xb + params[-9] @ xa + params[-6] @ u  # zb = Cz_bb @ xb + Cz_ba @ xa + Dzu_b @ u
                za_feedthrough = params[-8] @ xb + params[-7] @ xa + params[-5] @ u  # za = Cz_ab @ xb + Cz_aa @ xa + Dzu_a @ u
                z_feedthrough = jnp.hstack((zb_feedthrough, za_feedthrough))
                z_next = Dzw @ nonlinear_components(z, params) + z_feedthrough
            else:
                zb_feedthrough = ((self.W_mask["Cz_bb"] * params[-10]) @ xb + (self.W_mask["Cz_ba"] * params[-9]) @ xa
                                  + (self.W_mask["Dzu_b"] * params[-6]) @ u)  # zb = Cz_bb @ xb + Cz_ba @ xa + Dzu_b @ u
                za_feedthrough = ((self.W_mask["Cz_ab"] * params[-8]) @ xb + (self.W_mask["Cz_aa"] * params[-7]) @ xa +
                                  (self.W_mask["Dzu_a"] * params[-5]) @ u)  # za = Cz_ab @ xb + Cz_aa @ xa + Dzu_a @ u
                z_feedthrough = jnp.hstack((zb_feedthrough, za_feedthrough))
                z_next = (self.W_mask["Dzw"] * Dzw) @ nonlinear_components(z, params) + z_feedthrough
            return z_next

        fpi = jaxopt.FixedPointIteration(fixed_point_fun=contractive_map, maxiter=self.fpi_n_max, implicit_diff=True,
                                         tol=self.fpi_tol)

        @jax.jit
        def model_step_with_iter_count(x, u, params):
            # f : (nx+nu) --> (nx)
            # h : (nx+nu) --> (ny)

            xb = x[:self.nx_b]
            xa = x[self.nx_b:]

            z0 = jnp.concatenate((xb, u, jnp.zeros(self.nz)))
            z_star, fpi_state = fpi.run(z0, x, u, params)
            iter_num = fpi_state.iter_num
            residual = fpi_state.error

            w = nonlinear_components(z_star, params)
            wb = w[:self.nx_b + self.ny]
            wa = w[self.nx_b + self.ny:]
            if self.W_mask is None:
                # xb+ = A_bb @ xb + A_ba @ xa + Bu_b @ u + Bw_bb @ wb + Bw_ba @ wa
                xb_next = params[-25] @ xb + params[-24] @ xa + params[-21] @ u + params[-19] @ wb + params[-18] @ wa

                # xa+ = A_ab @ xb + A_aa @ xa + Bu_a @ u + Bw_ab @ wb + Bw_aa @ wa
                xa_next = params[-23] @ xb + params[-22] @ xa + params[-20] @ u + params[-17] @ wb + params[-16] @ wa
                x_next = jnp.hstack((xb_next, xa_next))

                # y = Cy_b @ xb + Cy_a @ xa + Dyu @ u + Dyw_b @ wb + Dyw_a @ wa
                y = params[-15] @ xb + params[-14] @ xa + params[-13] @ u + params[-12] @ wb + params[-11] @ wa
            else:
                # xb+ = A_bb @ xb + A_ba @ xa + Bu_b @ u + Bw_bb @ wb + Bw_ba @ wa
                xb_next = ((self.W_mask["A_bb"] * params[-25]) @ xb + (self.W_mask["A_ba"] * params[-24]) @ xa +
                           (self.W_mask["Bu_b"] * params[-21]) @ u + (self.W_mask["Bw_bb"] * params[-19]) @ wb +
                           (self.W_mask["Bw_ba"] * params[-18]) @ wa)

                # xa+ = A_ab @ xb + A_aa @ xa + Bu_a @ u + Bw_ab @ wb + Bw_aa @ wa
                xa_next = ((self.W_mask["A_ab"] * params[-23]) @ xb + (self.W_mask["A_aa"] * params[-22]) @ xa +
                           (self.W_mask["Bu_a"] * params[-20] @ u) + (self.W_mask["Bw_ab"] * params[-17]) @ wb +
                           (self.W_mask["Bw_aa"] * params[-16]) @ wa)
                x_next = jnp.hstack((xb_next, xa_next))

                # y = Cy_b @ xb + Cy_a @ xa + Dyu @ u + Dyw_b @ wb + Dyw_a @ wa
                y = ((self.W_mask["Cy_b"] * params[-15]) @ xb + (self.W_mask["Cy_a"] * params[-14]) @ xa +
                     (self.W_mask["Dyu"] * params[-13] @ u) + (self.W_mask["Dyw_b"] * params[-12]) @ wb +
                     (self.W_mask["Dyw_a"] * params[-11]) @ wa)

            return x_next, y, iter_num, residual

        @jax.jit
        def model_step(x, u, params):
            x_next, y, iter_num, residual = self.model_step_with_iter_count(x, u, params)
            return x_next, y

        self.model_step_with_iter_count = model_step_with_iter_count
        return model_step

    def _add_group_lasso_z(self, tau: float) -> Callable[[list[Array]], float]:

        @jax.jit
        def group_lasso_fun(th):
            cost = 0.
            Cz_ab = th[-8]
            Cz_aa = th[-7]
            Dzu_a = th[-5]
            ANN_params = self.get_network_params(th)
            W0 = ANN_params[0]
            if self.Dzw_dim1 == self.Dzw_dim2:
                D_bar = simple_cayley(th[-4], th[-3])
                Dzw = nn.sigmoid(th[-1]) * D_bar / self.lipschitz_const
            elif self.Dzw_dim1 > self.Dzw_dim2:
                D_bar = general_cayley(th[-4], th[-3], th[-2])
                Dzw = nn.sigmoid(th[-1]) * D_bar / self.lipschitz_const
            else:
                D_bar = general_cayley(th[-4], th[-3], th[-2])
                Dzw = nn.sigmoid(th[-1]) * D_bar.T / self.lipschitz_const
            Dzw_ab_aa = Dzw[self.nx_b + self.nu:, :]
            for i in range(self.nz):
                cost += tau * jnp.sqrt(
                    jnp.sum(Cz_ab[i, :] ** 2) + jnp.sum(Cz_aa[i, :] ** 2) + jnp.sum(Dzu_a[i, :] ** 2) +
                    jnp.sum(Dzw_ab_aa[i, :] ** 2) + jnp.sum(W0[:, i] ** 2))
            return cost

        return group_lasso_fun

    def _add_group_lasso_w(self, tau: float) -> Callable[[list[Array]], float]:

        @jax.jit
        def group_lasso_fun(th):
            cost = 0.
            ANN_params = self.get_network_params(th)
            W_last = ANN_params[-2]
            b_last = ANN_params[-1]
            Bw_ba = th[-18]
            Bw_aa = th[-16]
            Dyw_a = th[-11]
            if self.Dzw_dim1 == self.Dzw_dim2:
                D_bar = simple_cayley(th[-4], th[-3])
                Dzw = nn.sigmoid(th[-1]) * D_bar / self.lipschitz_const
            elif self.Dzw_dim1 > self.Dzw_dim2:
                D_bar = general_cayley(th[-4], th[-3], th[-2])
                Dzw = nn.sigmoid(th[-1]) * D_bar / self.lipschitz_const
            else:
                D_bar = general_cayley(th[-4], th[-3], th[-2])
                Dzw = nn.sigmoid(th[-1]) * D_bar.T / self.lipschitz_const
            Dzw_ba_aa = Dzw[:, self.nx_b + self.ny:]
            for i in range(self.nw):
                cost += tau * jnp.sqrt(jnp.sum(W_last[i, :] ** 2) + b_last[i] ** 2 + jnp.sum(Bw_ba[:, i] ** 2) +
                                         jnp.sum(Bw_aa[:, i] ** 2) + jnp.sum(Dyw_a[:, i] ** 2) + jnp.sum(
                    Dzw_ba_aa[:, i] ** 2))
            return cost

        return group_lasso_fun

    def _add_group_lasso_x(self, tau: float) -> Callable[[list[Array], list[Array]], float]:

        @jax.jit
        def group_lasso_fun(th, x0):
            cost = 0.
            A_ab = th[-23]
            A_ba = th[-24]
            A_aa = th[-22]
            Bu_a = th[-20]
            Bw_ab = th[-17]
            Bw_aa = th[-16]
            Cy_a = th[-14]
            Cz_ba = th[-9]
            Cz_aa = th[-7]

            for i in range(self.nx_a):
                cost += tau * jnp.sqrt(
                    jnp.sum(A_ab[i, :] ** 2) + jnp.sum(A_aa[i, :] ** 2) + jnp.sum(Bu_a[i, :] ** 2) +
                    jnp.sum(Bw_ab[i, :] ** 2) + jnp.sum(Bw_aa[i, :] ** 2) + sum(
                        [x0i[self.nx_b + i] ** 2 for x0i in x0]) +
                    jnp.sum(Cz_ba[:, i] ** 2) + jnp.sum(Cz_aa[:, i] ** 2) + jnp.sum(Cy_a[:, i] ** 2) +
                    jnp.sum(A_ba[:, i] ** 2) + jnp.sum(A_aa[:, i] ** 2) - A_aa[i, i] ** 2)
            return cost

        return group_lasso_fun

    def _add_lfr_mx_l1_reg(self, tau: float, reg_coeffs: Optional[Array]) -> Callable[[list[Array]], float]:
        raise NotImplementedError("Should be implemented in child class")

    def create_LFR_matrix_mask(self, th, eps):
        # TODO: implement in sub-classes as well
        raise NotImplementedError

    def save_LFR_matrices(self, filename):
        raise NotImplementedError

    def compute_new_l1_reg_weights(self, eps=1e-4):
        raise NotImplementedError