import argparse
import jax
from jax import numpy as jnp
import numpy as np
from model_augmentation_jax import StaticWellPosedLFRAugmentation, StaticContractingLFRAugmentation, GeneralNonlinearSystem
from matplotlib import pyplot as plt
import nonlinear_benchmarks as nlb
import os
import yaml


class baseline_model(GeneralNonlinearSystem):
    def __init__(self, params):
        super().__init__(params=params, nx=2, ny=1, nu=1, ts=4., tune_params=True)

    def f(self, x, u, params):
        x1_plus = x[0] + self.ts * (-params[0] * jnp.sqrt(x[0]) + params[3] * u[0])
        x2_plus = x[1] + self.ts * (params[1] * jnp.sqrt(x[0]) - params[2] * jnp.sqrt(x[1]))
        return jnp.array([x1_plus, x2_plus], dtype=jnp.float64)

    def h(self, x, u, params):
        x = x[:, None]
        return x[1]


def parse_args():
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(
        description = "Example script for testing LFR-based model augmentation on the Cascaded Tanks benchmark identification problem."
    )

    parser.add_argument(
        "--seed",
        type = int,
        default = 3,
        help = "Random seed."
    )

    parser.add_argument(
        "--LFR_struct",
        type = str,
        default = "WP",
        choices = ["WP", "contr"],
        help = "LFR matrix parametrization. Options: [WP, contr]. 'WP' for well-posed parametrization, 'contr' for contracting parametrization."
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
    LFR_struct = args.LFR_struct

    cwd = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(cwd, "hyperparams.yaml"), 'r') as file:
        hyperparams = yaml.safe_load(file)

    # load data
    data = nlb.Cascaded_Tanks()
    train_u = data[0].u
    train_y = data[0].y
    test_u = data[1].u
    test_y = data[1].y

    # baseline model
    nominal_params = np.array([0.05, 0.05, 0.05, 0.05])
    x0_init = np.array([train_y[0], train_y[0]])
    baseline_mdl = baseline_model(nominal_params)

    # simulate baseline model for normalization constants
    Yhat_fp, Xhat_fp = baseline_mdl.simulate(test_u, x0_init)

    # normalization
    std_y = np.std(train_y, axis=0)
    mu_y = np.mean(train_y, axis=0)
    std_u = np.std(train_u, axis=0)
    mu_u = np.mean(train_u, axis=0)
    std_x = np.std(Xhat_fp, axis=0)
    mu_x = np.mean(Xhat_fp, axis=0)
    norm = {"std_y": std_y, "mean_y": mu_y, "std_u": std_u, "mean_u": mu_u, "std_x": std_x, "mean_x": mu_x}

    if LFR_struct == "WP":
        model = StaticWellPosedLFRAugmentation(known_sys=baseline_mdl, hidden_layers=hyperparams['hidden_layers'],
                                               nodes_per_layer=hyperparams["n_nodes"],
                                               activation=hyperparams["activation"], nz=hyperparams["nz"],
                                               nw=hyperparams["nw"], lipschitz_const=2., seed=seed, x0=x0_init,
                                               norm_dict=norm, fpi_n_max=hyperparams["fpi_n_max"],
                                               fpi_tol=hyperparams["fpi_tol"])
        model.set_regularization_terms(rho_base=hyperparams["rho_base"], rho_aug=hyperparams["rho_aug"],
                                       ann_lipschitz_regul_coeff=hyperparams["ann_lipschitz_regul_coeff"])
        model.set_optimization_parameters(adam_epochs=hyperparams["adam_epochs"],
                                          lbfgs_epochs=hyperparams["lbfgs_epochs"],
                                          lbfgs_memory=hyperparams["lbfgs_memory"], train_x0=True)
    elif LFR_struct == "contr":
        model = StaticContractingLFRAugmentation(known_sys=baseline_mdl, hidden_layers=hyperparams['hidden_layers'],
                                                 nodes_per_layer=hyperparams["n_nodes"],
                                                 activation=hyperparams["activation_contr"], nz=hyperparams["nz"],
                                                 nw=hyperparams["nw"], lipschitz_const=2., seed=seed, x0=x0_init,
                                                 norm_dict=norm, fpi_n_max=hyperparams["fpi_n_max"],
                                                 fpi_tol=hyperparams["fpi_tol"])
        model.set_regularization_terms(rho_base=hyperparams["rho_base"], rho_aug=hyperparams["rho_aug"],
                                       ann_lipschitz_regul_coeff=hyperparams["ann_lipschitz_regul_coeff_contr"])
        model.set_optimization_parameters(adam_epochs=hyperparams["adam_epochs_contr"],
                                          lbfgs_epochs=hyperparams["lbfgs_epochs"],
                                          lbfgs_memory=hyperparams["lbfgs_memory_contr"], train_x0=True)
    else:
        raise ValueError

    model.fit(train_y, train_u)

    Yhat_train, _, _, _ = model.simulate(train_u, model.x0)
    Yhat_train_fp, _ = baseline_mdl.simulate(train_u, model.x0)

    # state initialization window is 5 according to benchmark rules
    x0_test = model.learn_x0(U=test_u[:5], Y=test_y[:5], rho_x0=None, RTS_epochs=0, lbfgs_refinement=True,
                             x0_init=np.array([test_y[0], test_y[0]]))
    Yhat_test, _, iters, residuals = model.simulate(test_u, x0_test)
    Yhat_test_fp, _ = baseline_mdl.simulate(test_u, x0_test[:2])

    rmse_train_fp = nlb.error_metrics.RMSE(train_y, Yhat_train_fp[:, 0])
    rmse_train = nlb.error_metrics.RMSE(train_y, Yhat_train[:, 0])
    print(f"Training RMSE: {rmse_train:.4} (baseline model: {rmse_train_fp:.4})")

    rmse_test = nlb.error_metrics.RMSE(test_y[5:], Yhat_test[5:, 0])
    rmse_test_fp = nlb.error_metrics.RMSE(test_y[5:], Yhat_test_fp[5:, 0])
    print(f"Testing RMSE: {rmse_test:.2} (baseline model: {rmse_test_fp:.4})")

    fig, ax = plt.subplots(2, 1, layout="tight")
    ax[0].plot(train_y, label="Meas.")
    ax[0].plot(Yhat_train[:, 0], label="Model")
    ax[0].set_title("Training")
    ax[0].set_xlabel("Time index")
    ax[0].set_ylabel(r"$y$")
    ax[0].legend()

    ax[1].plot(test_y)
    ax[1].plot(Yhat_test[:, 0])
    ax[1].set_title("Testing")
    ax[1].set_xlabel("Time index")
    ax[1].set_ylabel(r"$y$")

    plt.show(block=False)

    fig, ax = plt.subplots(2, 1, layout="tight", sharex=True)
    ax[0].plot(iters, '.')
    ax[0].axhline(hyperparams["fpi_n_max"], linestyle='--', color='k')
    ax[0].set_title("Fixed-point iterations required")
    ax[0].set_ylabel("Iteration count")
    ax[0].grid(True)

    ax[1].semilogy(residuals, '.')
    ax[1].set_title("Final residuals of the fixed-point iteration")
    ax[1].axhline(hyperparams["fpi_tol"], linestyle='--', color='k')
    ax[1].set_ylabel("Residual")
    ax[1].set_xlabel("Time index")
    ax[1].grid(True)

    plt.show()

if __name__ == "__main__":
    main()