import jax
import numpy as np
from jax import numpy as jnp
from model_augmentation_jax.augmentation_structures import DynamicLFRAugmentation, StaticLFRAugmentation
from model_augmentation_jax.baseline_models import GeneralNonlinearSystem
from model_augmentation_jax.utils import BestFitRatio
from matplotlib import pyplot as plt
import os
import argparse
import yaml


def F_fric(v):
    # friction force is implemented as an ANN, but in reality is a Stribeck-like force with additional viscous components
    vs = (v - fric_params["umean"]) / fric_params["ustd"]
    y_next = np.tanh(np.dot(vs, fric_params["W_in"].T) + fric_params["b_in"])
    y_out = jnp.dot(y_next, fric_params["W_out"].T) + fric_params["b_out"]
    return (y_out * fric_params["ystd"] + fric_params["umean"]).reshape(-1)


def truesystem(U):
    # simulates the data-generating system
    m = 1.
    c = 90.
    ks = 1000.
    Ts = 0.01
    # system generating the training and test dataset
    N_train = U.shape[0]
    x = np.zeros(2)
    Y = np.empty((N_train, 1))
    X = np.empty((N_train, 2))
    for k in range(N_train):
        X[k, :] = x
        Y[k, 0] = x[0]  #+ np.random.normal(loc=0, scale=0.01, size=2)
        u = U[k, 0]
        x1_plus = x[0] + Ts * x[1]
        fric = F_fric(x[1])[0]
        x2_plus = x[1] + (Ts / m) * (-ks * x[0] - c * x[1] - fric + u)
        x = np.array([x1_plus, x2_plus])
    return Y, X


# baseline model with known physical parameters
class known_sys(GeneralNonlinearSystem):
    def __init__(self):
        super().__init__(nx=2, ny=1, nu=1)

    def f(self, x, u, params):
        m = 1.
        c = 90.
        ks = 1000.
        Ts = 0.01

        x1_plus = x[0] + Ts * x[1]
        x2_plus = x[1] + (Ts / m) * (-ks * x[0] - c * x[1] + u[0])
        return jnp.array([x1_plus, x2_plus])

    def h(self, x, u, params):
        return x[0:1]


def parse_args():
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(
        description = "Example script for testing the group-lasso regularization option for LFR-based model augmentation."
    )

    parser.add_argument(
        "--seed",
        type = int,
        default = 0,
        help = "Random seed."
    )

    parser.add_argument(
        "--nxa",
        type = int,
        default = 2,
        help = "Dimension of the augmented state. If >0 dynamic augmentation, if ==0 static augmentation."
    )

    parser.add_argument(
        "--nwa",
        type = int,
        default = 3,
        help = "Dimension of the latent variable w_a."
    )

    parser.add_argument(
        "--nza",
        type = int,
        default = 3,
        help = "Dimension of the latent variable z_a."
    )

    parser.add_argument(
        "--var",
        type = str,
        choices = ["x", "w", "z"],
        default = "x",
        help = "Select the variable on which the group-lasso regularization is applied.."
    )

    parser.add_argument(
        "--rho",
        type = float,
        default = 1e-3,
        help = "Group-lasso regularization coefficient."
    )

    return parser.parse_args()


if __name__ == "__main__":

    jax.config.update('jax_platform_name', 'cpu')
    if not jax.config.jax_enable_x64:
        jax.config.update("jax_enable_x64", True)  # Enable 64-bit computations

    args = parse_args()

    print("Running experiment with:")
    print(args)

    seed = args.seed
    nxa = args.nxa
    nwa = args.nwa
    nza = args.nza
    gl_var = args.var
    rho_gl = args.rho

    N_train = 10000
    N_test = 1000

    # load friction parameters
    cwd = os.path.dirname(os.path.abspath(__file__))
    fric_params = np.load(os.path.join(cwd, "..", "iterative_l1_reg_example", "fric_params.npz"))
    with open(os.path.join(cwd, "hyperparams.yaml"), 'r') as file:
        hyperparams = yaml.safe_load(file)

    # data generation
    np.random.seed(seed=0)
    U_train = np.random.uniform(low=-1e2, high=1e2, size=(N_train, 1))
    Y_train, X_train = truesystem(U_train)
    U_test = np.random.uniform(low=-1e2, high=1e2, size=(N_test, 1))
    Y_test, X_test = truesystem(U_test)

    fp_model = known_sys()

    Yhat_train_fp, Xhat_train_fp = fp_model.simulate(U_train)

    # normalization
    std_y = np.std(Y_train, axis=0)
    mu_y = np.mean(Y_train, axis=0)
    std_u = np.std(U_train, axis=0)
    mu_u = np.mean(U_train, axis=0)
    std_x = np.std(Xhat_train_fp, axis=0)
    mu_x = np.mean(Xhat_train_fp, axis=0)

    if nxa > 0:
        model = DynamicLFRAugmentation(known_sys=fp_model, n_augm_states=nxa, hidden_layers=hyperparams["hl"],
                                       nodes_per_layer=hyperparams["nodes"], activation=hyperparams["act_fun"],
                                       nz_a=nza, nw_a=nwa, seed=seed, std_x=std_x, std_u=std_u, std_y=std_y, mu_x=mu_x,
                                       mu_u=mu_u, mu_y=mu_y)
        if gl_var == "x":
            model.set_regularization_terms(rho_x0=hyperparams["rho_x0"], tau_x=rho_gl)
            model.set_optimization_parameters(adam_epochs=hyperparams["adam_epochs"],
                                              lbfgs_epochs=hyperparams["lbfgs_epochs"], train_x0=True,
                                              zero_coeff=hyperparams["zc_x"], verbosity=50)
        elif gl_var == "w":
            model.set_regularization_terms(rho_x0=hyperparams["rho_x0"], tau_z=rho_gl)
            model.set_optimization_parameters(adam_epochs=hyperparams["adam_epochs"],
                                              lbfgs_epochs=hyperparams["lbfgs_epochs"], train_x0=True,
                                              zero_coeff=hyperparams["zc_w"], verbosity=50)
        elif gl_var == "z":
            model.set_regularization_terms(rho_x0=hyperparams["rho_x0"], tau_z=rho_gl)
            model.set_optimization_parameters(adam_epochs=hyperparams["adam_epochs"],
                                              lbfgs_epochs=hyperparams["lbfgs_epochs"], train_x0=True,
                                              zero_coeff=hyperparams["zc_z"], verbosity=50)
    else:
        model = StaticLFRAugmentation(known_sys=fp_model, hidden_layers=hyperparams["hl"],
                                      nodes_per_layer=hyperparams["nodes"], activation=hyperparams["act_fun"], nz=nza,
                                      nw=nwa, seed=seed, std_x=std_x, std_u=std_u, std_y=std_y, mu_x=mu_x, mu_u=mu_u,
                                      mu_y=mu_y)
        if gl_var == "x":
            raise ValueError("Regularizing augmented states is not possible for static structures!")
        elif gl_var == "w":
            model.set_regularization_terms(tau_w=rho_gl)
            model.set_optimization_parameters(adam_epochs=hyperparams["adam_epochs"],
                                              lbfgs_epochs=hyperparams["lbfgs_epochs"], train_x0=False,
                                              zero_coeff=hyperparams["zc_w"], verbosity=50)
        elif gl_var == "z":
            model.set_regularization_terms(tau_z=rho_gl)
            model.set_optimization_parameters(adam_epochs=hyperparams["adam_epochs"],
                                              lbfgs_epochs=hyperparams["lbfgs_epochs"], train_x0=False,
                                              zero_coeff=hyperparams["zc_z"], verbosity=50)

    model.fit(Y_train, U_train)
    t0 = model.t_solve
    print(f"Elapsed time: {t0} s")

    model.sparsity_analysis()

    Yhat_train, _ = model.simulate(U_train, model.x0)

    if nxa > 0:
        x0_test = model.learn_x0(U_test[:hyperparams["state_init_window"]], Y_test[:hyperparams["state_init_window"]],
                                 rho_x0=hyperparams["rho_x0"], RTS_epochs=hyperparams["RTS_iters"],
                                 lbfgs_refinement=True, verbosity=False)
    else:
        x0_test = np.zeros(2)
    Yhat_test, _ = model.simulate(U_test, x0_test)

    bfr_train = BestFitRatio(Y=Y_train, Yhat=Yhat_train)
    print(f"Training NRMSE: {bfr_train:.4%}")

    bfr_test = BestFitRatio(Y=Y_test, Yhat=Yhat_test)
    print(f"Testing NRMSE: {bfr_test:.2%}")

    plt.figure(layout="tight")
    plt.plot(Y_test, label="True")
    plt.plot(Y_test - Yhat_test, label="Augmented model error")
    plt.legend()
    plt.show(block=False)
