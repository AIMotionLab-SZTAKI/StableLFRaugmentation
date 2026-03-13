import jax
import jax.numpy as jnp
import numpy as np
from jax_sysid.models import l2reg, xsat, l1reg, lbfgs_options, get_bounds, adam_solver
from jax_sysid.utils import vec_reshape
import jaxopt
from model_augmentation_jax.baseline_models import verify_known_sys
from joblib import Parallel, delayed, cpu_count
from functools import partial
import time

from typing import Any, Callable, Dict, Optional, Tuple, Union
Array = Union[np.ndarray, jnp.ndarray]

epsil_lasso = 1.e-16
default_small_tau_th = 1.e-8  # add some small L1-regularization.
epsilon_A = 1.e-3


########################################################################################################################
#                                       AUGMENTATION BASE
########################################################################################################################
class AugmentationBase(object):
    """
    Base class for model augmentation structures.

    Subclasses must implement:
        - _initialize_parameters()
        - _create_jitted_model_step()
        - _add_lfr_mx_l1_reg()
        - _add_group_lasso_z()
        - _add_group_lasso_w()
        - _add_group_lasso_x() -- only for dynamic structures
    """

    def __init__(self,
                 known_sys: Any,
                 hidden_layers: int,
                 nodes_per_layer: int,
                 activation: str,
                 x0: Optional[Union[Array, list[Array]]] = None,
                 seed: Union[int, list[int]] = 42,
                 std_x: Optional[np.ndarray] = None,
                 std_u: Optional[np.ndarray] = None,
                 std_y: Optional[np.ndarray] = None,
                 mu_x: Optional[np.ndarray] = None,
                 mu_u: Optional[np.ndarray] = None,
                 mu_y: Optional[np.ndarray] = None,
                 ) -> None:
        """
        Initialize model structure.

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
        """

        self.nx = known_sys.nx  # TODO: overwrite it for dynamic augmentation
        self.ny = known_sys.ny
        self.nu = known_sys.nu

        self.tune_physical_params, self.init_phys_params = verify_known_sys(known_sys)

        self.std_x = np.ones(known_sys.nx) if std_x is None else std_x
        self.std_u = np.ones(known_sys.nu) if std_u is None else std_u
        self.std_y = np.ones(known_sys.ny) if std_y is None else std_y

        self.mu_x = np.zeros(known_sys.nx) if mu_x is None else mu_x
        self.mu_u = np.zeros(known_sys.nu) if mu_u is None else mu_u
        self.mu_y = np.zeros(known_sys.ny) if mu_y is None else mu_y

        self.model_step = self._create_jitted_model_step(known_sys, hidden_layers, activation)

        @jax.jit
        def model_step_with_sat(x, u, th, sat):
            """Perform a forward pass of the nonlinear model. States are saturated to avoid possible explosion of state values in case the system is unstable."""

            x, y = self.model_step(x, u, th)

            # saturate states to avoid numerical issues due to instability
            x = xsat(x, sat)
            return x, y

        self.model_step_with_sat = model_step_with_sat

        # initialize attributes with default values
        self.x0 = None
        self.Jopt = None
        self.t_solve = None
        self.regularization_fun = None
        self.params = None
        self.params_min = None
        self.params_max = None
        self.x0_min = None
        self.x0_max = None
        self.isbounded = None
        self.physical_param_idx = 0
        self.l1_or_group_lasso_applied = False
        self.adam_epochs = 0
        self.adam_learning_rate = 1e-3
        self.lbfgs_epochs = 1000
        self.lbfgs_tol = 1e-16
        self.lbfgs_memory = 20
        self.zero_coeff = 0.
        self.xsat = 1000.
        self.train_x0 = False
        def output_loss_fun(Yhat, Y): return jnp.sum((Yhat - Y) ** 2) / Y.shape[0]
        self.output_loss = output_loss_fun
        self.verbosity = 1

        self._augm_struct_initialization(known_sys, hidden_layers, nodes_per_layer, x0, seed, activation)

    def get_physical_params(self, params: list[Array]) -> Array:
        """
        Selects the physical parameters from the combined parameter list.

        Parameters
        ----------
        params : list of ndararys
            Combined parameter list of the model structure.

        Returns
        -------
        param : ndarary
            Physical parameters.
        """
        if self.tune_physical_params:
            return params[self.physical_param_idx]
        else:
            return self.init_phys_params

    def get_network_params(self, params: list[Array]) -> list[Array]:
        """
        Selects the parameters from the combined parameter list that correspond to the ANN.

        Parameters
        ----------
        params : list of ndarrays
            Combined parameter list of the model structure.

        Returns
        -------
        params : list of ndarrays
            List os parameters for the ANN function.
        """
        phys_params_idx = self.physical_param_idx
        if not self.tune_physical_params:
            phys_params_idx -= 1
        return params[:-phys_params_idx]

    def set_regularization_terms(self,
                                 rho_base: float = 0.,
                                 rho_aug: float = 0.,
                                 tau_aug: float = 0.,
                                 rho_x0: float = 0.,
                                 tau_lfr: float = 0.,
                                 lfr_reg_coeffs: Optional[Array] = None,
                                 tau_z: float = 0.,
                                 tau_w: float = 0.,
                                 tau_x: float = 0.,
                                 ann_lipschitz_regul_coeff: float = 0.,
                                 ):
        """
        Sets up all regularization options for model training,including conventional elastic-type regularization,
        regularization of the baseline parameters around their nominal value, L1 regularization of the LFR matrix for
        automatic model structure discovery, group-lasso regularization and Lipschitz regularization of the ANN.

        Parameters
        ----------
           rho_base : float, optional
               Regularization coefficient to penalize tuning of the baseline parameters compared to their nominal values.
           rho_aug : float, optional
               L2 regularization coefficient for the ANN.
           tau_aug : float, optional
               L1 regularization coefficient for the ANN.
           rho_x0 : optional
               L2 regularization of the (tuned) initial states.
           tau_lfr : float, optional
               Global regularization coefficient on the LFR matrix to enable automatic structure discovery.
           lfr_reg_coeffs : ndarray, optional
               Individual coefficients for LFR matrix regularization. If None, lfr_reg_coeffs is initialized as ones.
           tau_z, tau_w, tau_x : float, optional
               Group-lasso regularization coefficients for variables z, w, and x, respectively.
           ann_lipschitz_regul_coeff : float, optional
               Regularization coefficient for the ANN Lipschitz bound.
           """

        if rho_base > 0:
            print(f"Deviations of the physical parameters from their nominal values are penalized with coefficient: alpha={rho_base}")

            @jax.jit
            def baseline_params_regul_fun(params):
                phys_params = self.get_physical_params(params)
                init_phys_params = self.init_phys_params
                cost_phys = 0.
                for i in range(init_phys_params.size):
                    cost_phys += (init_phys_params[i] - phys_params[i]) ** 2 / np.maximum(init_phys_params[i] ** 2,1e-6)
                return rho_base / 2 * cost_phys

        if rho_aug > 0:
            print(f"L2 regularization for the ANN parameters are applied with coefficient: lambda={rho_aug}")

        if tau_aug > 0:
            self.l1_or_group_lasso_applied = True
            print(f"L1 regularization for the ANN parameters are applied with coefficient: tau={tau_aug}")

        if rho_x0 > 0:
            print(f"L2 regularization for the initial states are applied with coefficient: rho={rho_x0}")

        if tau_lfr > 0:
            self.l1_or_group_lasso_applied = True
            print(f"L1 regularization for augmentation structure discovery is enabled.")
            lfr_mx_l1_reg_fun = self._add_lfr_mx_l1_reg(tau_lfr, lfr_reg_coeffs)

        if tau_z > 0:
            self.l1_or_group_lasso_applied = True
            print(f"Group lasso regularization is applied for variable z with coefficient: tau={tau_z}")
            group_lasso_z_fun = self._add_group_lasso_z(tau_z)

        if tau_w > 0:
            self.l1_or_group_lasso_applied = True
            print(f"Group lasso regularization is applied for variable w with coefficient: tau={tau_w}")
            group_lasso_w_fun = self._add_group_lasso_w(tau_w)

        if tau_x > 0:
            self.l1_or_group_lasso_applied = True
            print(f"Group lasso regularization is applied for variable x with coefficient: tau={tau_x}")
            group_lasso_x_fun = self._add_group_lasso_x(tau_x)

        if ann_lipschitz_regul_coeff > 0:
            if hasattr(self, 'lipschitz_const'):
                ann_lipschitz_bound = self.lipschitz_const
            else:
                ann_lipschitz_bound = 0.  # if we do not enforce a specific bound, then penalize the ANN Lipschitz constant

            print(f"ANN Lipschitz constant is regularized when exceeding bound {ann_lipschitz_bound} with coefficient: rho={ann_lipschitz_regul_coeff}")

            @jax.jit
            def ann_lipschitz_regul_fun(params):
                network_params = self.get_network_params(params)
                no_network_params = len(network_params)
                no_weights = int(no_network_params / 2)
                ann_lipschitz_const = 1.
                for i in range(no_weights):
                    ann_lipschitz_const *= jnp.linalg.norm(network_params[2 * i], 2)
                lipschitz_penalty = jnp.maximum(ann_lipschitz_const - ann_lipschitz_bound, 0.) ** 2
                return ann_lipschitz_regul_coeff * lipschitz_penalty

        @jax.jit
        def combined_regularization_terms(params, x0):
            network_params = self.get_network_params(params)
            cost = 0.
            if rho_base > 0:
                cost += baseline_params_regul_fun(params)
            if rho_aug > 0:
                cost += rho_aug / 2* l2reg(network_params)
            if tau_aug > 0:
                cost += tau_aug * l1reg(network_params)
            if rho_x0 > 0:
                cost += rho_x0 * sum([jnp.sum(x0i ** 2) for x0i in x0])
            if tau_lfr > 0:
                cost += lfr_mx_l1_reg_fun(params)
            if tau_z > 0:
                cost += group_lasso_z_fun(params)
            if tau_w > 0:
                cost += group_lasso_w_fun(params)
            if tau_x > 0:
                cost += group_lasso_x_fun(params, x0)
            if ann_lipschitz_regul_coeff > 0:
                cost += ann_lipschitz_regul_fun(params)
            return cost
        self.regularization_fun = combined_regularization_terms

    def set_optimization_parameters(self,
                                    adam_epochs: int,
                                    lbfgs_epochs: int,
                                    adam_learning_rate: float = 1e-3,
                                    lbfgs_tol: float = 1e-16,
                                    lbfgs_memory: int = 10,
                                    train_x0: bool = False,
                                    zero_coeff: float = 0.,
                                    params_min: Optional[list[Array]] = None,
                                    params_max: Optional[list[Array]] = None,
                                    x0_min: Optional[Union[Array, list[Array]]] = None,
                                    x0_max: Optional[Union[Array, list[Array]]] = None,
                                    output_loss_fun: Optional[Callable[[Array, Array], float]] = None,
                                    verbosity: int = 1,
                                    state_sat: Optional[float] = None,
                                    ) -> None:
        """
        Set up optimization parameters before model training.

        Parameters
        ----------
        adam_epochs : int
            Number of Adam iterations to "warm-start" the L-BFGS method.
        lbfgs_epochs : int
            Number of L-BFGS iterations to fine-tune models.
        adam_learning_rate : float, optional
            Learning rate for Adam optimizer. Defaults to 1e-3.
        lbfgs_tol : float, optional
            Tolerance for the L-BFGS optimizer. Defaults to 1e-16.
        lbfgs_memory : int, optional
            Memory for the Hessian approximations in the L-BFGS optimizer. Defaults to 10.
        train_x0 : bool, optional
            Whether to train the initial states or not. Defaults to False.
        zero_coeff : float, optional
            Parameters with smaller absolute value than this parameter are set to zero at the end of the training.
            Useful for L1 and group-lasso regularization. Defaults to 0.
        params_min : list, optional
            Lower bound for the optimization parameters (excluding x0). If None, no lower bound is applied. Defaults to None.
        params_max : list, optional
            Upper bound for the optimization parameters (excluding x0). If None, no upper bound is applied. Defaults to None.
        x0_min : ndarray or list of ndarrays, optional
            Lower bound for the initial state (if optimized). If None, no lower bound is applied. Defaults to None.
        x0_max : ndarray or list of ndarrays, optional
            Upper bound for the initial state (if optimized). If None, no upepr bound is applied. Defaults to None.
        output_loss_fun : callable, optional
            Loss function to be used for optimization (without additional regularization terms). If None, the RMSE loss
            function is used. Defaults to None.
        verbosity : int, optional
            Sets the verbosity level during optimization. Defaults to 1.
        state_sat : float, optional
            Sets the saturation level for the states during training to handle numerical stability losses that are
            mainly present during L-BFGS-B optimization. If None, the default value is set as xsat = 1000. Defaults to None.
        """
        # set optimization parameters (iteration length and optimization algorithm-specific options)
        self.adam_epochs = adam_epochs
        self.adam_learning_rate = adam_learning_rate
        self.lbfgs_epochs = lbfgs_epochs
        self.lbfgs_tol = lbfgs_tol
        self.lbfgs_memory = lbfgs_memory

        # set if initial states should be optimized or not
        self.train_x0 = train_x0

        # threshold for L1 and group-lasso regularization
        self.zero_coeff = zero_coeff  # parameters smaller in absolute value than this are set to zero after optimization

        # set parameter bounds
        self.params_min = params_min
        self.params_max = params_max
        self.x0_min = x0_min
        self.x0_max = x0_max

        # set output loss (default is RMSE)
        if output_loss_fun is None:
            def output_loss_fun(Yhat, Y): return jnp.sum((Yhat - Y) ** 2) / Y.shape[0]
        self.output_loss = output_loss_fun

        self.verbosity = verbosity
        if state_sat is not None:
            self.xsat = state_sat

    def fit(self,
            Y: Union[Array, list[Array]],
            U: Union[Array, list[Array]],
            ) -> None:
        """
        Trains the model according to all the pre-set options.

        Parameters
        ----------
        Y : ndarray or list of ndarrays
            Measured output values with shape (N, ny) or a list of ndarrays each with a shape of (Ni, ny).
        U : ndarray or list of ndarrays, optional
            Measured input values with shape (N, nu) or a list of ndarrays each with a shape of (Ni, nu).
        """

        jax.config.update('jax_platform_name', 'cpu')
        if not jax.config.jax_enable_x64:
            # Enable 64-bit computations
            jax.config.update("jax_enable_x64", True)

        data, x0, Nexp = self._prepare_training_data(U, Y)
        z = self.params

        if self.regularization_fun is None:
            self.set_regularization_terms()

        self.isbounded = (self.params_min is not None) or (self.params_max is not None) or (
                self.train_x0 and ((self.x0_min is not None) or (self.x0_max is not None)))
        if self.isbounded or self.l1_or_group_lasso_applied:
            # define default bounds, in case they are not provided
            if self.params_min is None:
                self.params_min = list()
                for i in range(len(z)):
                    self.params_min.append(-jnp.ones_like(z[i]) * np.inf)
            if self.params_max is None:
                self.params_max = list()
                for i in range(len(z)):
                    self.params_max.append(jnp.ones_like(z[i]) * np.inf)
            if self.train_x0:
                if self.x0_min is None:
                    self.x0_min = [-jnp.ones_like(self.x0) * np.inf] * Nexp
                if not isinstance(self.x0_min, list):
                    # repeat the same initial-state bound on all experiments
                    self.x0_min = [self.x0_min] * Nexp
                if len(self.x0_min) is not Nexp:
                    # number of experiments has changed, repeat the same initial-state bound on all experiments
                    self.x0_min = [self.x0_min[0]] * Nexp
                if self.x0_max is None:
                    self.x0_max = [jnp.ones_like(self.x0) * np.inf] * Nexp
                if not isinstance(self.x0_max, list):
                    self.x0_max = [self.x0_max] * Nexp
                if len(self.x0_max) is not Nexp:
                    self.x0_max = [self.x0_max[0]] * Nexp

        z, x0, Jopt, t_solve_1 = self._train_with_adam(data, z, x0, Nexp)
        z, x0, Jopt, t_solve_2 = self._train_with_lbfgs(data, z, x0, Nexp)
        t_solve = t_solve_1 + t_solve_2

        # Zero coefficients smaller than zero_coeff in absolute value
        for i in range(len(z)):
            z[i] = np.array(z[i])
            z[i][np.abs(z[i]) <= self.zero_coeff] = 0.

        x0 = [np.array(x0i) for x0i in x0]
        for i in range(Nexp):
            x0[i][np.abs(x0[i]) <= self.zero_coeff] = 0.0

        # Save optimal parameters and initial states
        self.params = z
        x0 = [np.array(x0i) * self.std_x + self.mu_x for x0i in x0]
        self.x0 = x0
        if Nexp == 1:
            self.x0 = self.x0[0]
        self.Jopt = Jopt
        self.t_solve = t_solve
        return

    def fit_parallel(self,
                     Y: Union[Array, list[Array]],
                     U: Union[Array, list[Array]],
                     seeds: list[int],
                     n_jobs: Optional[int] = None) -> list[Any]:
        """
        Fits the model in parallel using multiple seeds.

        Parameters
        ----------
        Y : ndarray or list of ndarrays
            Measured output values with shape (N, ny) or a list of ndarrays each with a shape of (Ni, ny).
        U : ndarray or list of ndarrays, optional
            Measured input values with shape (N, nu) or a list of ndarrays each with a shape of (Ni, nu).
        seeds : list
            The seeds used for initialization.
        n_jobs : int, optional
            The number of parallel jobs to run (default is None, which means using all available cores).

        Returns
        -------
        models : list
            A list of fitted models.
        """

        def single_fit(seed):
            if not jax.config.jax_enable_x64:
                # Enable 64-bit computations
                jax.config.update("jax_enable_x64", True)
            self.parallel_init_fun(seed)
            print("\033[1m" + f"Fitting model with seed = {seed} ... " + "\033[0m")
            self.fit(Y, U)
            print("\033[1m" + f"Seed = {seed}: done." + "\033[0m")
            return self

        if n_jobs is None:
            n_jobs = cpu_count()  # Use all available cores by default

        models = Parallel(n_jobs=n_jobs)(delayed(single_fit)(seed=seed) for seed in seeds)
        return models

    def simulate(self,
                 U: Union[Array, list[Array]],
                 X0: Optional[Union[Array, list[Array]]] = None,
                 ) -> Tuple[Union[Array, list[Array]], Union[Array, list[Array]]]:
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
            x_next, y = self.model_step(x, u, self.params)
            y = jnp.hstack((y, x))
            x_next = x_next.reshape(-1)
            return x_next, y

        if N_meas == 1:
            _, YX = jax.lax.scan(model_step_fixed_params, x0_scaled.reshape(-1), vec_reshape(U_scaled))
            Y = YX[:, 0:self.ny] * self.std_y + self.mu_y
            X = YX[:, self.ny:] * self.std_x + self.mu_x
        else:
            Y = []
            X = []
            for i in range(N_meas):
                _, YX = jax.lax.scan(model_step_fixed_params, x0_scaled[i].reshape(-1), vec_reshape(U_scaled[i]))
                Y.append(YX[:, 0:self.ny] * self.std_y + self.mu_y)
                X.append(YX[:, self.ny:] * self.std_x + self.mu_x)
        return Y, X

    def count_ann_zero_params(self) -> None:
        """
        Counts the number of zero parameters amongst the learning component (ANN) parameters. Useful when L1
        regularization is used for the ANN parameters.
        """
        params = self.get_network_params(self.params)
        n_params = len(params)

        zero_params = 0
        all_params = 0

        for i in range(n_params):
            param_i = np.array(params[i])
            all_params += vec_reshape(param_i).shape[0] * vec_reshape(param_i).shape[1]
            param_i[np.abs(param_i) <= self.zero_coeff] = 0.
            zero_params += np.count_nonzero(param_i == 0)

        print("Zero ANN parameters: " + str(zero_params) + " from" + str(all_params) + ".")

    def learn_x0(self,
                 U: Array,
                 Y: Array,
                 rho_x0: Optional[float] = 1e-4,
                 RTS_epochs: int = 1,
                 verbosity: bool = True,
                 lbfgs_refinement: bool = False,
                 lbfgs_epochs: int = 1000,
                 Q: Optional[Array] = None,
                 R: Optional[Array] = None,
                 x0_init: Optional[Array] = None,
                 ) -> Array:
        """
        EKF+RTS-based initial state estimation with L-BFGS refinement according to the jax-sysid toolbox.
        See: https://github.com/bemporad/jax-sysid

        Parameters
        ----------
        U : ndarray
            Input sequence based on which the state estimation is computed.
        Y : ndarray
            Corresponding true output sequence.
        rho_x0 : float, optional
            Regularization coefficient for the initial state. If NOne, no regularization is applied. Defaults to 1e-4.
        RTS_epochs : int, optional
            Number of EKF/RTS passes for initial estimate. Defaults to 1.
        verbosity : bool, optional
            Sets the verbosity level (True/False). Defaults to True.
        lbfgs_refinement : bool, optional
            Determines whether to refine the estimate obtained by the EKF/RTS passes with L-BFGS or not. Defaults to False.
        Q : ndarray, optional
            Process noise covariance matrix. If None, Q = 1e-5 * I is applied. Defaults to None.
        R : ndarray, optional
            Measurement noise covariance matrix. If None, R = I is applied. Defaults to None.
        x0_init : ndarray, optional
            Initial guess for x0. If None, the estimation starts from x0 = 0. Defaults to None.

        Returns
        -------
        x0 : ndarray
            Estimated initial state value.
        """
        nx = self.nx
        ny = self.ny
        N = U.shape[0]
        Y = (vec_reshape(Y.copy()) - self.mu_y) / self.std_y
        U = (vec_reshape(U.copy()) - self.mu_u) / self.std_u

        @jax.jit
        def state_fcn(x, u, params):
            x_plus, _ = self.model_step(x, u, params)
            return x_plus

        @jax.jit
        def output_fcn(x, u, params):
            _, y = self.model_step(x, u, params)
            return y

        @jax.jit
        def Ck(x, u):
            return jax.jacrev(output_fcn)(x, u=u, params=self.params)

        @jax.jit
        def Ak(x, u):
            return jax.jacrev(state_fcn)(x, u=u, params=self.params)

        if rho_x0 is None:
            rho_x0 = 0.
        if R is None:
            R = np.eye(ny)
        if Q is None:
            Q = 1.e-5 * np.eye(nx)

        # Forward EKF pass:
        @jax.jit
        def EKF_update(state, yuk):
            x, P, mse_loss = state
            yk = yuk[:ny]
            u = yuk[ny:]

            # measurement update
            y = output_fcn(x, u, self.params)
            Ckk = Ck(x, u)
            PC = P @ Ckk.T
            # M = PC / (R + C @PC) # this solves the linear system M*(R + C @PC) = PC
            # Note: Matlab's mrdivide A / B = (B'\A')' = np.linalg.solve(B.conj().T, A.conj().T).conj().T
            M = jax.scipy.linalg.solve((R+Ckk@PC), PC.T, assume_a='pos').T
            e = yk-y
            mse_loss += np.sum(e**2)  # just for monitoring purposes
            x1 = x + M@e  # x(k | k)

            # Standard Kalman measurement update
            # P -= M@PC.T
            # P = (P + P.T)/2. # P(k|k)

            # Joseph stabilized covariance update
            IKH = -M@Ckk
            IKH += jnp.eye(nx)
            P1 = IKH@P@IKH.T+M@R@M.T  # P(k|k)

            # Time update
            Akk = Ak(x1, u)
            P2 = Akk@P1@Akk.T+Q
            # P2 = (P2+P2.T)/2.
            x2 = state_fcn(x1, u, self.params)
            output = (x1, P1, x2, P2, Akk)

            return (x2, P2, mse_loss), output

        @jax.jit
        def RTS_update(state, input):
            x, P = state
            P1, P2, x1, x2, A = input

            # G=(PP1[k]@AA[k].T)/PP2[k]
            try:
                G = jax.scipy.linalg.solve(P2, (P1@A.T).T, assume_a='pos').T
            except:
                G = jax.scipy.linalg.solve(P2, (P1@A.T).T, assume_a='gen').T
            x = x1+G@(x-x2)
            P = P1+G@(P-P2)@G.T
            return (x, P), None

        # L2-regularization on initial state x0, 0.5*rho_x0*||x0||_2^2
        if rho_x0 > 0:
            P = np.eye(nx) / (rho_x0 * N)
        else:
            P = np.eye(nx)
        if x0_init is None:
            x = np.zeros(nx)
        else:
            x = (x0_init.copy().reshape(-1) - self.mu_x) / self.std_x

        for epoch in range(RTS_epochs):
            mse_loss = 0.

            # Forward EKF pass
            state = (x, P, mse_loss)
            state, output = jax.lax.scan(EKF_update, state, np.hstack((Y, U)))
            XX1, PP1, XX2, PP2, AA = output
            # PP1 = P(k | k)
            # PP2 = P(k + 1 | k)
            # XX1 = x(k | k)
            # XX2 = x(k + 1 | k)
            mse_loss = state[2]/N

            # RTS smoother pass:
            x = XX2[N-1]
            P = PP2[N-1]
            state = (x, P)
            input = (PP1[::-1], PP2[::-1], XX1[::-1], XX2[::-1], AA[::-1])
            state, _ = jax.lax.scan(RTS_update, state, input)
            x, P = state

            if verbosity:
                print(f"\nRTS smoothing, epoch: {epoch+1: 3d}/{RTS_epochs: 3d}, MSE loss = {mse_loss: 8.6f}")

        x = np.array(x)

        isstatebounded = self.x0_min is not None or self.x0_max is not None
        if isstatebounded:
            lb = self.x0_min
            if isinstance(lb, list):
                lb = lb[0]
            if lb is None:
                lb = -np.inf*np.ones(nx)
            ub = self.x0_max
            if isinstance(ub, list):
                ub = ub[0]
            if ub is None:
                ub = np.inf*np.ones(nx)
            if np.any(x < lb) or np.any(x > ub):
                lbfgs_refinement = True

        if lbfgs_refinement:
            # Refine via L-BFGS with very small penalty on x0
            options = lbfgs_options(
                iprint=-1, iters=lbfgs_epochs, lbfgs_tol=1.e-10, memory=100)

            @jax.jit
            def SS_step(x, u):
                x_next, y = self.model_step(x, u, self.params)
                return x_next.reshape(-1), y

            @jax.jit
            def J(x0):
                _, Yhat = jax.lax.scan(SS_step, x0, U)
                return jnp.sum((Yhat - Y) ** 2) / U.shape[0] + 0.5 * rho_x0 * jnp.sum(x0**2)
            if not isstatebounded:
                solver = jaxopt.ScipyMinimize(fun=J, tol=options["ftol"], method="L-BFGS-B", maxiter=options["maxfun"],
                                              options=options)
                x, state = solver.run(x)
            else:
                solver = jaxopt.ScipyBoundedMinimize(fun=J, tol=options["ftol"], method="L-BFGS-B",
                                                     maxiter=options["maxfun"], options=options)
                x, state = solver.run(x, bounds=(lb, ub))
            x = np.array(x)

            if verbosity:
                mse_loss = state.fun_val - 0.5 * rho_x0 * np.sum(x**2)
                print(f"\nFinal loss MSE (after LBFGS refinement) = {mse_loss: 8.6f}")
        return x * self.std_x + self.mu_x

    def _augm_struct_initialization(self, sys: Any, hidden_layers: int, nodes_per_layer: int,
                                    x0: Optional[Union[Array, list[Array]]], seed: int, activation: str) -> None:
        """Creates the ANN structure for augmentation and initializes all parameters."""
        if isinstance(seed, int):
            self._initialize_parameters(sys, hidden_layers, nodes_per_layer, x0, seed, activation)
        elif isinstance(seed, list):
            self._initialize_parameters(sys, hidden_layers, nodes_per_layer, x0, seed[0], activation)
            self.parallel_init_fun = lambda x: self._initialize_parameters(sys, hidden_layers, nodes_per_layer, x0, x,
                                                                           activation)
        else:
            raise TypeError("seed must be an int or a list")

    def _init(self, params: list[Array], x0: Optional[Union[Array, list[Array]]]) -> None:
        """Sets up model parameters and initial states according to the given model augmentation structure."""

        jax.config.update('jax_platform_name', 'cpu')
        if not jax.config.jax_enable_x64:
            # Enable 64-bit computations
            jax.config.update("jax_enable_x64", True)

        self.params = [jnp.array(th) for th in params]

        if x0 is not None:
            if isinstance(x0, list):
                Nexp = len(x0)
                self.x0 = [jnp.array(x0[i]) for i in range(Nexp)]
            else:
                self.x0 = jnp.array(x0)
        else:
            self.x0 = jnp.zeros(self.nx)

        return

    def _prepare_training_data(self, U: Union[np.ndarray, list[np.ndarray]], Y: Union[np.ndarray, list[np.ndarray]],
                               ) -> Tuple[Dict[str, list[np.ndarray]], list[np.ndarray], int]:
        """Normalize and package training data."""
        if isinstance(U, np.ndarray):
            U = [U]
            Y = [Y]
            x0 = [self.x0]
        else: # multiple experiments are provided
            if not isinstance(self.x0, list):
                x0 = [self.x0 for _ in range(len(U))]
            else:
                x0 = self.x0

        U_normed = []
        Y_normed = []
        x0_normed = []
        for i in range(len(U)):
            ui = (U[i] - self.mu_u) / self.std_u
            yi = (Y[i] - self.mu_y) / self.std_y
            x0i = (x0[i] - self.mu_x) / self.std_x
            U_normed.append(vec_reshape(ui))
            Y_normed.append(vec_reshape(yi))
            x0_normed.append(x0i)
        dataset = {"U": U_normed, "Y": Y_normed}
        return dataset, x0_normed, len(U)

    def _train_with_adam(self, dataset: Dict[str, list[Array]], z: list[Array], x0: Union[Array, list[Array]], Nexp: int,
                         ) -> Tuple[list[Array], Union[Array, list[Array]], float, float]:
        """Trains the model with the Adam optimizer."""

        n_params = len(z)  # number of parameters (excluding x0)
        if self.train_x0:
            for i in range(Nexp):
                # one initial state per experiment
                z.append(x0[i].reshape(-1))

        lb = None
        ub = None
        if self.isbounded:
            lb = self.params_min
            ub = self.params_max
            if self.train_x0:
                lb.append(self.x0_min)
                ub.append(self.x0_max)

        t_solve = time.time()  # include JIT time also in model training

        @jax.jit
        def loss_fn(th, x0):
            # Calculate the loss function for system identification. (RMSE)
            f = partial(self.model_step_with_sat, th=th, sat=self.xsat)
            cost = 0.
            for i in range(Nexp):
                _, Yhat = jax.lax.scan(f, x0[i], dataset["U"][i])
                cost += self.output_loss(Yhat, dataset["Y"][i])
            return cost

        if self.train_x0:
            @jax.jit
            def J(z):
                th = z[:n_params]
                x0 = z[n_params:]
                cost = loss_fn(th, x0)
                cost += self.regularization_fun(th, x0)
                return cost
        else:
            @jax.jit
            def J(z):
                cost = loss_fn(z, x0)
                cost += self.regularization_fun(z, x0)
                return cost

        def JdJ(z):
            return jax.value_and_grad(J)(z)

        z, Jopt = adam_solver(JdJ, z, self.adam_epochs, self.adam_learning_rate, 1, lb, ub)
        if self.train_x0:
            x0 = [z[n_params + i].reshape(-1) for i in range(Nexp)]
        t_solve = time.time() - t_solve
        return z[0:n_params], x0, Jopt, t_solve

    def _train_with_lbfgs(self, dataset: Dict[str, list[Array]], z: list[Array], x0: Union[Array, list[Array]], Nexp: int,
                          ) -> Tuple[list[Array], Union[Array, list[Array]], float, float]:
        """Trains the model with the L-BFGS-B method."""

        nth = len(z)  # model params only
        if self.l1_or_group_lasso_applied:
            # duplicate params to create positive and negative parts
            z.extend(z)
            for i in range(nth):
                zi = z[i].copy()
                # we could also consider bounds here, if present
                z[i] = jnp.maximum(zi, 0.) + epsil_lasso
                z[nth + i] = -jnp.minimum(zi, 0.) + epsil_lasso

        nzmx0 = len(z)
        if self.train_x0:
            for i in range(Nexp):
                # one initial state per experiment
                z.append(x0[i].reshape(-1))
            # in case of group-Lasso, if state nr. i is removed from A,B,C then the corresponding x0(i)=0
            # because of L2-regularization on x0.

        # total number of optimization variables
        nvars = sum([zi.size for zi in z])

        # L-BFGS-B params (no L1 regularization)
        options = lbfgs_options(min(self.verbosity, 90), self.lbfgs_epochs, self.lbfgs_tol, self.lbfgs_memory)

        print("Solving NLP with L-BFGS (%d optimization variables) ..." % nvars)

        if self.l1_or_group_lasso_applied:
            bounds = get_bounds(z[0:nth], epsil_lasso, self.params_min, self.params_max)
            if self.train_x0:
                bounds[0].append(self.x0_min)
                bounds[1].append(self.x0_max)
        elif self.isbounded:
            lb = self.params_min
            ub = self.params_max
            if self.train_x0:
                lb.append(self.x0_min)
                ub.append(self.x0_max)

        t_solve = time.time()  # include JIT time t_solve

        @jax.jit
        def loss_fn(th, x0):
            # Calculate the loss function for system identification. (RMSE)
            f = partial(self.model_step_with_sat, th=th, sat=self.xsat)
            cost = 0.
            for i in range(Nexp):
                _, Yhat = jax.lax.scan(f, x0[i], dataset["U"][i])
                cost += self.output_loss(Yhat, dataset["Y"][i])
            return cost

        if self.l1_or_group_lasso_applied:
            @jax.jit
            def J(z):
                # Optimize wrt to split positive and negative part of model parameters
                th = [z1 - z2 for z1, z2 in zip(z[0:nth], z[nth:2 * nth])]
                if self.train_x0:
                    x0_var = z[nzmx0:]
                else:
                    x0_var = x0
                cost = loss_fn(th, x0_var) + self.regularization_fun(th, x0_var)
                return cost

        else:
            @jax.jit
            def J(z):
                if self.train_x0:
                    th = z[:nzmx0]
                    x0_var = z[nzmx0:]
                else:
                    th = z
                    x0_var = x0
                cost = loss_fn(th, x0_var) + self.regularization_fun(th, x0_var)
                return cost

        if not self.l1_or_group_lasso_applied:
            if not self.isbounded:
                solver = jaxopt.ScipyMinimize(fun=J, tol=self.lbfgs_tol, method="L-BFGS-B", maxiter=self.lbfgs_epochs,
                                              options=options)
                z, state = solver.run(z)
            else:
                solver = jaxopt.ScipyBoundedMinimize(fun=J, tol=self.lbfgs_tol, method="L-BFGS-B",
                                                     maxiter=self.lbfgs_epochs, options=options)
                z, state = solver.run(z, bounds=(lb, ub))
        else:
            solver = jaxopt.ScipyBoundedMinimize(fun=J, tol=self.lbfgs_tol, method="L-BFGS-B",
                                                 maxiter=self.lbfgs_epochs, options=options)
            z, state = solver.run(z, bounds=bounds)
            z[0:nth] = [z1 - z2 for (z1, z2) in zip(z[0:nth], z[nth:2 * nth])]

        t_solve = time.time() - t_solve
        iter_num = state.iter_num
        Jopt = state.fun_val
        print('L-BFGS-B done in %d iterations.' % iter_num)

        if self.train_x0:
            x0 = [z[nzmx0 + i].reshape(-1) for i in range(Nexp)]

        return z[0:nth], x0, Jopt, t_solve

    def _initialize_parameters(self, known_sys: Any, hidden_layers: int, nodes_per_layer: int,
                              x0: Optional[Union[Array, list[Array]]], seed: int, activation: str) -> None:
        """Initializes the parameters according to the given augmentation structure."""
        self.physical_param_idx = 0  # should be correctly changed in subclass
        raise NotImplementedError("Should be implemented in child class")

    def _create_jitted_model_step(self, known_sys: Any, hidden_layers: int, activation: str,
                                  ) -> Callable[[Array, Array, list[Array]], Tuple[Array, Array]]:
        """Creates JIT-compiled state transition and output functions according to the given augmentation structure."""
        raise NotImplementedError("Should be implemented in child class")

    def _add_lfr_mx_l1_reg(self, tau: float, reg_coeffs: Optional[Array]) -> Callable[[list[Array]], float]:
        raise NotImplementedError("Should be implemented in child class")

    def _add_group_lasso_z(self, tau: float) -> Callable[[list[Array]], float]:
        raise NotImplementedError("Should be implemented in child class")

    def _add_group_lasso_w(self, tau: float) -> Callable[[list[Array]], float]:
        raise NotImplementedError("Should be implemented in child class")

    def _add_group_lasso_x(self, tau: float) -> Callable[[list[Array], list[Array]], float]:
        raise NotImplementedError("Should be implemented in child class")