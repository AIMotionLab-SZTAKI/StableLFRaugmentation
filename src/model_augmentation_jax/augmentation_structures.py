from model_augmentation_jax.optimization_base import AugmentationBase
from model_augmentation_jax.networks import initialize_network, generate_simple_ann
import numpy as np
from jax import numpy as jnp
import jax

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

    def sparsity_analysis(self):
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

