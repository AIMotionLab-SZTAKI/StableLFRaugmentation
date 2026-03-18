import jax
import numpy as np
from jax import numpy as jnp
from model_augmentation_jax.augmentation_structures import StaticWellPosedLFRAugmentation
from model_augmentation_jax.baseline_models import GeneralNonlinearSystem
from model_augmentation_jax.utils import NRMSE_loss
from matplotlib import pyplot as plt
import os
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
    Ffric = np.empty((N_train, 1))
    for k in range(N_train):
        X[k, :] = x
        Y[k, 0] = x[0]  #+ np.random.normal(loc=0, scale=0.01, size=2)
        u = U[k, 0]
        x1_plus = x[0] + Ts * x[1]
        fric = F_fric(x[1])[0]
        Ffric[k , 0] = fric
        x2_plus = x[1] + (Ts / m) * (-ks * x[0] - c * x[1] - fric + u)
        x = np.array([x1_plus, x2_plus])
    return Y, X, Ffric


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


if __name__ == "__main__":

    jax.config.update('jax_platform_name', 'cpu')
    if not jax.config.jax_enable_x64:
        jax.config.update("jax_enable_x64", True)  # Enable 64-bit computations

    N_train = 10000
    N_test = 1000

    # load friction parameters
    cwd = os.path.dirname(os.path.abspath(__file__))
    fric_params = np.load(os.path.join(cwd, "fric_params.npz"))
    with open(os.path.join(cwd, "hyperparams.yaml"), 'r') as file:
        hyperparams = yaml.safe_load(file)

    # data generation
    np.random.seed(seed=0)
    U_train = np.random.uniform(low=-1e2, high=1e2, size=(N_train, 1))
    Y_train, X_train, F_fric_train = truesystem(U_train)
    U_test = np.random.uniform(low=-1e2, high=1e2, size=(N_test, 1))
    Y_test, X_test, F_fric_test = truesystem(U_test)

    fp_model = known_sys()

    Yhat_train_fp, Xhat_train_fp = fp_model.simulate(U_train)

    # normalization
    std_y = np.std(Y_train, axis=0)
    mu_y = np.mean(Y_train, axis=0)
    std_u = np.std(U_train, axis=0)
    mu_u = np.mean(U_train, axis=0)
    std_x = np.std(Xhat_train_fp, axis=0)
    mu_x = np.mean(Xhat_train_fp, axis=0)

    # create storage for iterative L1 regularization
    e_base = float(jnp.sum(((Yhat_train_fp - Y_train)/std_y) ** 2) / Y_train.shape[0])
    Lambda = None
    Lambda_diffs = []
    Zero_elements = []

    #Iterative re-weighting L1 regularization
    for i in range(hyperparams["max_iters"]):
        # initialize model
        model = StaticWellPosedLFRAugmentation(known_sys=fp_model, hidden_layers=hyperparams["hidden_layers"],
                                               nodes_per_layer=hyperparams["nodes_per_layer"],
                                               activation=hyperparams["activation"], nz=hyperparams["nz_a"],
                                               nw=hyperparams["nw_a"], seed=hyperparams["seed"], std_y=std_y,
                                               std_x=std_x, std_u=std_u, mu_y=mu_y, mu_x=mu_x, mu_u=mu_u,
                                               fpi_n_max=hyperparams["fpi_max"], fpi_tol=hyperparams["fpi_tol"],
                                               lipschitz_const=10.05)
        if i > 0:
            # use final parameters from the last iteration
            model._init(params=params, x0=np.zeros(2))
        # set regularization and training parameters
        model.set_regularization_terms(ann_lipschitz_regul_coeff=hyperparams["ann_lipschitz"],
                                       tau_lfr=hyperparams["epsilon"]*e_base, lfr_reg_coeffs=Lambda)
        model.set_optimization_parameters(adam_epochs=hyperparams["adam_epochs"],
                                          lbfgs_epochs=hyperparams["lbfgs_epochs"],
                                          lbfgs_memory=hyperparams["lbfgs_memory"], train_x0=False, verbosity=-1)
        # train the model
        model.fit(Y_train, U_train)
        print(f"Model #{i + 1} finished training in {model.t_solve:.2} seconds.")

        # compute new L1 weight for each element of the LFR matrix
        params = model.params
        Lambda_new, zeros, W_LFR = model.compute_new_l1_reg_weights(eps=hyperparams["epsilon"])
        Zero_elements.append(zeros)

        # Compare the relative change of the regularization weight compared to previous iteration
        if i == 0:
            Lambda = np.ones_like(Lambda_new)
        Lambda_diff = np.linalg.norm(Lambda_new - Lambda, 2) / np.linalg.norm(Lambda_new, 2)
        Lambda_diffs.append(Lambda_diff)
        if Lambda_diff < hyperparams["termination_rel_cond"]:
            # if the weights converged --> terminate
            break
        else:
            Lambda = Lambda_new

    # re-train model with masked W elements
    model = StaticWellPosedLFRAugmentation(known_sys=fp_model, hidden_layers=hyperparams["hidden_layers"],
                                           nodes_per_layer=hyperparams["nodes_per_layer"],
                                           activation=hyperparams["activation"], nz=hyperparams["nz_a"],
                                           nw=hyperparams["nw_a"], seed=hyperparams["seed"], std_y=std_y,
                                           std_x=std_x, std_u=std_u, mu_y=mu_y, mu_x=mu_x, mu_u=mu_u,
                                           fpi_n_max=hyperparams["fpi_max"], fpi_tol=hyperparams["fpi_tol"],
                                           lipschitz_const=10.05, mask_params=params, mask_eps=hyperparams["epsilon"])
    model.set_regularization_terms(ann_lipschitz_regul_coeff=hyperparams["ann_lipschitz"])
    model.set_optimization_parameters(adam_epochs=hyperparams["adam_epochs"],
                                      lbfgs_epochs=hyperparams["lbfgs_epochs"],
                                      lbfgs_memory=hyperparams["lbfgs_memory"], train_x0=False, verbosity=-1)
    model.fit(Y_train, U_train)

    Yhat_test, Xhat_test, _, _ = model.simulate(U_test)
    Yhat_test_fp, _ = fp_model.simulate(U_test)

    nrmse_test = NRMSE_loss(Yhat_test, Y_test)
    nrmse_test_fp = NRMSE_loss(Yhat_test_fp, Y_test)

    # train a naive model with no regularization
    model = StaticWellPosedLFRAugmentation(known_sys=fp_model, hidden_layers=hyperparams["hidden_layers"],
                                           nodes_per_layer=hyperparams["nodes_per_layer"],
                                           activation=hyperparams["activation"], nz=hyperparams["nz_a"],
                                           nw=hyperparams["nw_a"], seed=hyperparams["seed"], std_y=std_y,
                                           std_x=std_x, std_u=std_u, mu_y=mu_y, mu_x=mu_x, mu_u=mu_u,
                                           fpi_n_max=hyperparams["fpi_max"], fpi_tol=hyperparams["fpi_tol"],
                                           lipschitz_const=10.05)
    model.set_regularization_terms(ann_lipschitz_regul_coeff=hyperparams["ann_lipschitz"])
    model.set_optimization_parameters(adam_epochs=hyperparams["adam_epochs"],
                                      lbfgs_epochs=hyperparams["lbfgs_epochs"],
                                      lbfgs_memory=hyperparams["lbfgs_memory"], train_x0=False, verbosity=-1)
    model.fit(Y_train, U_train)

    Yhat_test2, _, _, _ = model.simulate(U_test)
    nrmse_test2 = NRMSE_loss(Yhat_test2, Y_test)

    print(f"Test NRMSE: {100*nrmse_test}%")
    print(f"Test NRMSE (baseline model): {100*nrmse_test_fp}%")
    print(f"Test NRMSE (naive approach): {100 * nrmse_test2}%")

    plt.figure(layout="tight")
    plt.plot(Y_test, label="True")
    plt.plot(Y_test - Yhat_test, label="Augmented model error")
    plt.plot(Y_test - Yhat_test_fp, label="FP model error")
    plt.legend()
    plt.show(block=False)

    fig, ax = plt.subplots(2, 1, sharex=True, layout="tight")
    ax[0].plot(Zero_elements)
    ax[0].set_ylabel(r"Zeros in $W$ [\%]")
    ax[0].grid()
    ax[1].plot(Lambda_diffs)
    ax[1].set_ylabel(r"$\Delta \Lambda$")
    ax[1].set_xlabel("Iterations")
    ax[1].grid()
    plt.show()
