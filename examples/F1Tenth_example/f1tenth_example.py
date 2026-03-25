import argparse
import jax
from jax import numpy as jnp
import numpy as np
from model_augmentation_jax import (StaticLFRAugmentation, StaticWellPosedLFRAugmentation, StaticContractingLFRAugmentation,
                                    DynamicLFRAugmentation, DynamicWellPosedAugmentation, DynamicContractingAugmentation,
                                    GeneralNonlinearSystem)
from model_augmentation_jax.utils import create_ndarray_from_list, NRMSE_loss
from matplotlib import pyplot as plt
import os
import yaml


class single_track_mdl(GeneralNonlinearSystem):
    def __init__(self, params):
        super().__init__(params=params, nx=3, ny=3, nu=2, ts=0.025, tune_params=True)

    def f(self, x, u, params):
        # Parameters
        #  x - current state (1D array with shape nx)
        #  u - current input (1D array with shape nu)
        #  params - physical parameters (1D array)
        # Returns
        #  x_plus - next state (1D array with shape nx)

        # state values
        v_xi, v_eta, omega = x

        # inputs
        delta, d = u

        # physical parameters
        m, Jz, lr, lf, Cm1, Cm2, Cm3, Cr, Cf = params

        # Longitudinal tire force
        Fxi = Cm1 * d - Cm2 * v_xi - jnp.sign(v_xi) * Cm3

        # Lateral tire forces
        alpha_r = (-v_eta + lr * omega) / jnp.maximum(v_xi, 0.1 * jnp.ones_like(v_xi))
        alpha_f = delta - (v_eta + lf * omega) / jnp.maximum(v_xi, 0.1 * jnp.ones_like(v_xi))
        Fr_eta = Cr * alpha_r
        Ff_eta = Cf * alpha_f

        # CT dynamics
        v_xid = 1 / m * (Fxi + Fxi * jnp.cos(delta) - Ff_eta * jnp.sin(delta) + m * v_eta * omega)
        v_etad = 1 / m * (Fr_eta + Fxi * jnp.sin(delta) + Ff_eta * jnp.cos(delta) - m * v_xi * omega)
        omega_d = 1 / Jz * (Ff_eta * lf * jnp.cos(delta) + Fxi * lf * jnp.sin(delta) - Fr_eta * lr)

        # discretization
        return jnp.array([v_xi + self.ts * v_xid, v_eta + self.ts * v_etad, omega + self.ts * omega_d])

    def h(self, x, u, params):
        return x


def load_data_set(folder_name):
    input_data_list = []
    output_data_list = []
    data_names_list = os.listdir(folder_name)
    for name in data_names_list:
        input_file_name = os.path.join(folder_name, name, "input.csv")
        with open(input_file_name, 'r') as in_file:
            input_data = np.genfromtxt(in_file, delimiter=",")
        output_file_name = os.path.join(folder_name, name, "output.csv")
        with open(output_file_name, 'r') as in_file:
            output_data = np.genfromtxt(in_file, delimiter=",")
        idxCut = np.where(np.abs(output_data[:, 1]) > 0.25)[0][0]
        input_data_list.append(input_data[idxCut:, 1:])
        output_data_list.append(output_data[idxCut:, 1:])
    return input_data_list, output_data_list

def add_noise(Y_train, SNR_level):
    np.random.seed(0)  # for reproducibility
    if SNR_level == 0:
        return Y_train
    else:
        if SNR_level == 15:
            sigma_n = np.array([0.12, 0.017, 0.1])
        elif SNR_level == 20:
            sigma_n = np.array([0.07, 0.0095, 0.06])
        elif SNR_level == 25:
            sigma_n = np.array([0.037, 0.0055, 0.033])
        elif SNR_level == 30:
            sigma_n = np.array([0.022, 0.003, 0.018])
        elif SNR_level == 40:
            sigma_n = np.array([0.007, 0.0009, 0.0057])
        else:
            raise NotImplementedError

        y_noisy = []
        noise = []
        for y in Y_train:
            n = np.random.normal(np.zeros(3), sigma_n, size=y.shape)
            noise.append(n)
            y_noisy.append(y+n)

        Py = np.sum(np.square(create_ndarray_from_list(Y_train)), axis=0)
        Pn = np.sum(np.square(create_ndarray_from_list(noise)), axis=0)
        SNR = 10 * np.log10(Py / Pn)
        print(f"Training SNR: {SNR}")
        return y_noisy


def parse_args():
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(
        description = "Example script for testing LFR-based model augmentation on the F1Tenth identification example."
    )

    parser.add_argument(
        "--seed",
        type = int,
        default = 0,
        help = "Random seed."
    )

    parser.add_argument(
        "--SNR",
        type = int,
        default = 0,
        choices = [0, 40, 30, 20],
        help = "Added Gaussian noise to training with specified sensor-to-noise ratio (SNR). If 0, no noise is added."
    )

    parser.add_argument(
        "--state_augm_type",
        type = str,
        default = "static",
        choices = ["static", "dynamic"],
        help = "Type of state augmentation. Options: [static, dynamic]"
    )

    parser.add_argument(
        "--LFR_struct",
        type = str,
        default = "zero",
        choices = ["zero", "lower-triang", "WP", "contr"],
        help = "LFR matrix parametrization. Options: [zero, lower-triang, WP, contr]. 'zero' for Dzw=0, 'lower-triang'"
               "for Dzw strictly lower triangular, 'WP' for well-posed parametrization, 'contr' for contracting parametrization."
    )

    return parser.parse_args()


def main():

    jax.config.update('jax_platform_name', 'cpu')
    if not jax.config.jax_enable_x64:
        jax.config.update("jax_enable_x64", True)  # Enable 64-bit computations

    args = parse_args()

    print("Running experiment with:")
    print(args)

    seed = args.seed
    SNR = args.SNR
    state_augm_type = args.state_augm_type
    LFR_struct = args.LFR_struct

    cwd = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(cwd, "hyperparam_config.yaml"), 'r') as file:
        hyperparams = yaml.safe_load(file)

    nominal_phys_params = np.array([2.93, 0.0796, 0.168, 0.163, 41.796, 2.0152, 0.4328, 41.7372, 29.4662])
    fp_model = single_track_mdl(params=nominal_phys_params)

    # load data
    cwd = os.path.dirname(os.path.abspath(__file__))
    U_train, Y_train = load_data_set(os.path.join(cwd, 'train_data'))
    Y_train = add_noise(Y_train, SNR)
    U_test, Y_test = load_data_set(os.path.join(cwd, 'test_data'))

    std_y = np.std(create_ndarray_from_list(Y_train), axis=0)
    mu_y = np.mean(create_ndarray_from_list(Y_train), axis=0)
    std_u = np.std(create_ndarray_from_list(U_train), axis=0)
    mu_u = np.mean(create_ndarray_from_list(U_train), axis=0)
    norm = {"std_x": std_y, "std_y": std_y, "mean_x": mu_y, "mean_y": mu_y, "std_u": std_u, "mean_u": mu_u}

    # x0 is known:
    x0_train = [Y_train[i][0, :] for i in range(len(Y_train))]
    x0_test = [Y_test[i][0, :] for i in range(len(Y_test))]

    if state_augm_type == "static" and LFR_struct == "zero":
        model = StaticLFRAugmentation(known_sys=fp_model, hidden_layers=hyperparams["hidden_layers"],
                                      nodes_per_layer=hyperparams["nodes_per_layer"], activation=hyperparams["act_fun"],
                                      nz=hyperparams["nz"], nw=hyperparams["nw_static"], x0=x0_train, seed=seed,
                                      norm_dict=norm, Dzw_structure=None)
        model.set_regularization_terms(rho_base=hyperparams["rho_base"])
        model.set_optimization_parameters(adam_epochs=hyperparams["adam_epochs"],
                                          lbfgs_epochs=hyperparams["lbfgs_epochs"],
                                          adam_learning_rate=hyperparams["lr_static"],
                                          lbfgs_memory=hyperparams["lbfgs_memory"], verbosity=hyperparams["verbosity"],
                                          train_x0=False)
    elif state_augm_type == "static" and LFR_struct == "lower-triang":
        model = StaticLFRAugmentation(known_sys=fp_model, hidden_layers=hyperparams["hidden_layers"],
                                      nodes_per_layer=hyperparams["nodes_per_layer"], activation=hyperparams["act_fun"],
                                      nz=hyperparams["nz"], nw=hyperparams["nw_static"], x0=x0_train, seed=seed,
                                      norm_dict=norm, Dzw_structure="lower")
        model.set_regularization_terms(rho_base=hyperparams["rho_base"])
        model.set_optimization_parameters(adam_epochs=hyperparams["adam_epochs"],
                                          lbfgs_epochs=hyperparams["lbfgs_epochs"],
                                          adam_learning_rate=hyperparams["lr_static"],
                                          lbfgs_memory=hyperparams["lbfgs_memory"], verbosity=hyperparams["verbosity"],
                                          train_x0=False)
    elif state_augm_type == "static" and LFR_struct == "WP":
        model = StaticWellPosedLFRAugmentation(known_sys=fp_model, hidden_layers=hyperparams["hidden_layers"],
                                               nodes_per_layer=hyperparams["nodes_per_layer"],
                                               activation=hyperparams["act_fun"], nz=hyperparams["nz"],
                                               nw=hyperparams["nw_static"], lipschitz_const=2.5, x0=x0_train, seed=seed,
                                               norm_dict=norm, fpi_n_max=hyperparams["FPI_max"],
                                               fpi_tol=hyperparams["FPI_tol"])
        model.set_regularization_terms(rho_base=hyperparams["rho_base"],
                                       ann_lipschitz_regul_coeff=hyperparams["ann_lipschitz_regul"])
        model.set_optimization_parameters(adam_epochs=hyperparams["adam_epochs"],
                                          lbfgs_epochs=hyperparams["lbfgs_epochs"],
                                          adam_learning_rate=hyperparams["lr_static"],
                                          lbfgs_memory=hyperparams["lbfgs_memory"], verbosity=hyperparams["verbosity"],
                                          train_x0=False)
    elif state_augm_type == "static" and LFR_struct == "contr":
        model = StaticContractingLFRAugmentation(known_sys=fp_model, hidden_layers=hyperparams["hidden_layers"],
                                                 nodes_per_layer=hyperparams["nodes_per_layer"],
                                                 activation=hyperparams["act_fun"], nz=hyperparams["nz"],
                                                 nw=hyperparams["nw_static"], lipschitz_const=2.5, x0=x0_train,
                                                 seed=seed, norm_dict=norm, fpi_n_max=hyperparams["FPI_max"],
                                                 fpi_tol=hyperparams["FPI_tol"],
                                                 contraction_rate=hyperparams["contraction_rate"])
        model.set_regularization_terms(rho_base=hyperparams["rho_base"],
                                       ann_lipschitz_regul_coeff=hyperparams["ann_lipschitz_regul"])
        model.set_optimization_parameters(adam_epochs=hyperparams["adam_epochs"],
                                          lbfgs_epochs=hyperparams["lbfgs_epochs"],
                                          adam_learning_rate=hyperparams["lr_static"],
                                          lbfgs_memory=hyperparams["lbfgs_memory"], verbosity=hyperparams["verbosity"],
                                          train_x0=False)
    elif state_augm_type == "dynamic" and LFR_struct == "zero":
        model = DynamicLFRAugmentation(known_sys=fp_model, n_augm_states=hyperparams["nx_a"],
                                       hidden_layers=hyperparams["hidden_layers"],
                                       nodes_per_layer=hyperparams["nodes_per_layer"], activation=hyperparams["act_fun"],
                                       nz_a=hyperparams["nz"], nw_a=hyperparams["nw_dynamic"], x0=x0_train, seed=seed,
                                       norm_dict=norm, Dzw_structure=None)
        model.set_regularization_terms(rho_base=hyperparams["rho_base"], rho_x0=hyperparams["rho_x0"])
        model.set_optimization_parameters(adam_epochs=hyperparams["adam_epochs"],
                                          lbfgs_epochs=hyperparams["lbfgs_epochs"],
                                          adam_learning_rate=hyperparams["lr_dynamic"],
                                          lbfgs_memory=hyperparams["lbfgs_memory"], verbosity=hyperparams["verbosity"],
                                          train_x0=True)
    elif state_augm_type == "dynamic" and LFR_struct == "lower-triang":
        model = DynamicLFRAugmentation(known_sys=fp_model, n_augm_states=hyperparams["nx_a"],
                                       hidden_layers=hyperparams["hidden_layers"],
                                       nodes_per_layer=hyperparams["nodes_per_layer"], activation=hyperparams["act_fun"],
                                       nz_a=hyperparams["nz"], nw_a=hyperparams["nw_dynamic"], x0=x0_train, seed=seed,
                                       norm_dict=norm, Dzw_structure="lower")
        model.set_regularization_terms(rho_base=hyperparams["rho_base"], rho_x0=hyperparams["rho_x0"])
        model.set_optimization_parameters(adam_epochs=hyperparams["adam_epochs"],
                                          lbfgs_epochs=hyperparams["lbfgs_epochs"],
                                          adam_learning_rate=hyperparams["lr_dynamic"],
                                          lbfgs_memory=hyperparams["lbfgs_memory"], verbosity=hyperparams["verbosity"],
                                          train_x0=True)
    elif state_augm_type == "dynamic" and LFR_struct == "WP":
        model = DynamicWellPosedAugmentation(known_sys=fp_model, n_augm_states=hyperparams["nx_a"],
                                             hidden_layers=hyperparams["hidden_layers"],
                                             nodes_per_layer=hyperparams["nodes_per_layer"],
                                             activation=hyperparams["act_fun"], nz_a=hyperparams["nz"],
                                             nw_a=hyperparams["nw_dynamic"], lipschitz_const=2.5, x0=x0_train, seed=seed,
                                             norm_dict=norm, fpi_n_max=hyperparams["FPI_max"],
                                             fpi_tol=hyperparams["FPI_tol"],)
        model.set_regularization_terms(rho_base=hyperparams["rho_base"], rho_x0=hyperparams["rho_x0"],
                                       ann_lipschitz_regul_coeff=hyperparams["ann_lipschitz_regul"])
        model.set_optimization_parameters(adam_epochs=hyperparams["adam_epochs"],
                                          lbfgs_epochs=hyperparams["lbfgs_epochs"],
                                          adam_learning_rate=hyperparams["lr_dynamic"],
                                          lbfgs_memory=hyperparams["lbfgs_memory"], verbosity=hyperparams["verbosity"],
                                          train_x0=True)
    elif state_augm_type == "dynamic" and LFR_struct == "contr":
        model = DynamicContractingAugmentation(known_sys=fp_model, n_augm_states=hyperparams["nx_a"],
                                               hidden_layers=hyperparams["hidden_layers"],
                                               nodes_per_layer=hyperparams["nodes_per_layer"],
                                               activation=hyperparams["act_fun"], nz_a=hyperparams["nz"],
                                               nw_a=hyperparams["nw_dynamic"], lipschitz_const=2.5, x0=x0_train, seed=seed,
                                               norm_dict=norm, fpi_n_max=hyperparams["FPI_max"],
                                               fpi_tol=hyperparams["FPI_tol"])
        model.set_regularization_terms(rho_base=hyperparams["rho_base"], rho_x0=hyperparams["rho_x0"],
                                       ann_lipschitz_regul_coeff=hyperparams["ann_lipschitz_regul"])
        model.set_optimization_parameters(adam_epochs=hyperparams["adam_epochs"],
                                          lbfgs_epochs=hyperparams["lbfgs_epochs"],
                                          adam_learning_rate=hyperparams["lr_dynamic"],
                                          lbfgs_memory=hyperparams["lbfgs_memory"], verbosity=hyperparams["verbosity"],
                                          train_x0=True)
    else:
        raise ValueError

    model.fit(Y_train, U_train)
    print(f"Model training finished in {model.t_solve} seconds.")
    sim_results_train = model.simulate(U_train, model.x0)

    if state_augm_type == "static":
        start_idx = 0
        sim_results_test = model.simulate(U_test, x0_test)
    else:
        start_idx = hyperparams["n_lag"]  # only calculate errors after the state initialization window
        X0_test = []
        for Yi, Ui, x0i in zip(Y_test, U_test, x0_test):
            x0_init = np.hstack((x0i, np.zeros(hyperparams["nx_a"])))
            x0_test_i = model.learn_x0(U=Ui[:hyperparams["n_lag"], :], Y=Yi[:hyperparams["n_lag"], :], RTS_epochs=3,
                                       x0_init=x0_init, verbosity=False)
            X0_test.append(x0_test_i)
        sim_results_test = model.simulate(U_test, X0_test)

    Y_train = create_ndarray_from_list(Y_train)
    Yhat_train = create_ndarray_from_list(sim_results_train[0])
    Y_test = create_ndarray_from_list([yi[start_idx:, :] for yi in Y_test])
    Yhat_test = create_ndarray_from_list([yi[start_idx:, :] for yi in sim_results_test[0]])

    # baseline params
    baseline_params_final = model.get_physical_params(model.params)
    print(f"Baseline parameter deviation (from init.): {np.linalg.norm(nominal_phys_params - baseline_params_final, 2)}")

    nrmse_train = NRMSE_loss(Yhat_train, Y_train)
    print(f"Training NRMSE: {nrmse_train:.2%}")

    nrmse_test = NRMSE_loss(Yhat_test, Y_test)
    print(f"Testing NRMSE: {nrmse_test:.2%}")

    if LFR_struct == "WP" or LFR_struct == "contr":
        fig, ax = plt.subplots(2, 1, layout="tight", sharex=True)
        ax[0].plot(create_ndarray_from_list(sim_results_test[2]), '.')
        ax[0].axhline(hyperparams["FPI_max"], linestyle='--', color='k')
        ax[0].set_ylabel("FPI count")
        ax[0].grid(True)

        ax[1].semilogy(create_ndarray_from_list(sim_results_test[3]), '.')
        ax[1].axhline(hyperparams["FPI_tol"], linestyle='--', color='k')
        ax[1].set_ylabel("Residual")
        ax[1].set_xlabel("Sim. index")
        ax[1].grid(True)
        plt.show(block=False)

    fig, ax = plt.subplots(2, 3, layout="tight")
    ax[0, 0].plot(Y_train[:, 0], label="Meas.")
    ax[0, 0].plot(Y_train[:, 0] - Yhat_train[:, 0], label="Error")
    ax[0, 0].set_xlabel("Time index")
    ax[0, 0].set_ylabel(r"$v_\xi$")
    ax[0, 0].legend()

    ax[0, 1].plot(Y_train[:, 1])
    ax[0, 1].plot(Y_train[:, 1] - Yhat_train[:, 1])
    ax[0, 1].set_xlabel("Time index")
    ax[0, 1].set_ylabel(r"$v_\eta$")
    ax[0, 1].set_title("Train")

    ax[0, 2].plot(Y_train[:, 2])
    ax[0, 2].plot(Y_train[:, 2] - Yhat_train[:, 2])
    ax[0, 2].set_xlabel("Time index")
    ax[0, 2].set_ylabel(r"$\omega$")

    ax[1, 0].plot(Y_test[:, 0])
    ax[1, 0].plot(Y_test[:, 0] - Yhat_test[:, 0])
    ax[1, 0].set_xlabel("Time index")
    ax[1, 0].set_ylabel(r"$v_\xi$")

    ax[1, 1].plot(Y_test[:, 1])
    ax[1, 1].plot(Y_test[:, 1] - Yhat_test[:, 1])
    ax[1, 1].set_xlabel("Time index")
    ax[1, 1].set_ylabel(r"$v_\eta$")
    ax[1, 1].set_title("Test")

    ax[1, 2].plot(Y_test[:, 2])
    ax[1, 2].plot(Y_test[:, 2] - Yhat_test[:, 2])
    ax[1, 2].set_xlabel("Time index")
    ax[1, 2].set_ylabel(r"$\omega$")

    plt.show()


if __name__ == "__main__":
    main()