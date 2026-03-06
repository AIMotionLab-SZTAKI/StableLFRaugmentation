import jax
from jax._src.prng import PRNGKeyArray
import jax.numpy as jnp
import numpy as np
import flax.linen as nn

from typing import Callable, List, Optional, Union
Array = Union[np.ndarray, jnp.ndarray]


relu = lambda x: jnp.maximum(jnp.zeros_like(a=x), x)


def generic_activation_fun(
        x: Array,
        activation: Optional[str] = None,
) -> Array:
    """
    Activation function for ANNs.

    Parameters
    ----------
    x : ndarray
        Input of the activation function.
    activation : str, optional
        Activation function to be applied to the input. If None, linear activation is used.

    Returns
    -------
    y : ndarray
        Output of the activation function.
    """
    if activation is None:
        y =  x
    elif activation == 'relu':
        y = relu(x)
    elif activation == 'tanh':
        y = jnp.tanh(x)
    elif activation == 'sigmoid':
        y = nn.sigmoid(x)
    elif activation == 'swish':
        y = nn.swish(x)
    else:
        raise NotImplementedError('Further activation functions should be implemented by user!')
    return y


def generate_simple_ann(
        hidden_layers: int,
        act_fun: str,
) -> Callable[[Array, List[Array]], Array]:
    """
    Generates a JIT-compatible ANN function

    Parameters
    ----------
    hidden_layers : int
        Number of hidden layers to be applied.
    act_fun : str
        Activation function for the hidden layers.

    Returns
    -------
    net : Callable
        ANN function.
    """
    def net(net_in, params):
        # first layer
        W = params[0]
        b = params[1]
        y_next = generic_activation_fun(jnp.dot(net_in, W.T) + b, act_fun)
        for i in range(hidden_layers - 1):
            # using Python for loop with relatively few iteration does not influence the compile time that much
            # for LARGE neural networks (>10 hidden layers) this should be written with jax.lax.fori_loop
            W = params[2 * i + 2]
            b = params[2 * i + 3]
            y_next = generic_activation_fun(jnp.dot(y_next, W.T) + b, act_fun)
        W = params[2 * hidden_layers]
        b = params[2 * hidden_layers + 1]
        # output with linear activation
        y_out = jnp.dot(y_next, W.T) + b
        return y_out
    return net


def initialization_gain(
        act_fun: str,
) -> float:
    """
    Provides a gain for weight initialization according to the Xavier method. (Matches the pytorch implementation).

    Parameters
    ----------
    act_fun : str
        Activation function.

    Returns
    -------
    gain : float
        Gain for the weight initialization.
    """
    if act_fun == "linear" or act_fun == "sigmoid":
        gain = 1.
    elif act_fun == "relu" or act_fun == "swish":
        gain = jnp.sqrt(2.)
    elif act_fun == "tanh":
        gain = 5. / 3.
    else:
        raise NotImplementedError("Initialization is not implemented for further activation functions.")
    return gain


def initialize_weights_and_biases(
        layer_units: list[int],
        input_features: int,
        key: PRNGKeyArray,
        act_fun: str,
) -> list[Array]:
    """
    Implements a modified Xavier initialization scheme for the ANN that parametrizes the learning component of the
    augmentation structure. The modification means that the weight matrix of the last layer is initialized as a near
    zero matrox so that the behavior of the augmented model matches the behavior of the baseline model.

    Parameters
    ----------
    layer_units : list
        Lists the number of neurons in each layer, as: [hidden_dim, ..., hidden_dim, output_dim].
    input_features : int
        Number of input features (input dimension).
    key : PRNGKeyArray
        Random key for initialization.
    act_fun : str
        Activation function.

    Returns
    -------
    weights_and_biases : list
        A list of arrays containing the initialized values of the weights and biases.
    """

    weights_and_bias = []

    for i, units in enumerate(layer_units):
        if i == 0:
            # first layer weight has dim (num_units, input shape)
            key_carry, key_w = jax.random.split(key, 2)
            a = initialization_gain(act_fun) * jnp.sqrt(6. / (input_features + units))
            w = jax.random.uniform(key=key_w, shape=(units, input_features), minval=-a, maxval=a, dtype=jnp.float64)
            b = jnp.zeros(shape=(units,), dtype=jnp.float64)
        elif i == len(layer_units)-1:
            # Xavier init. (for last layer: the weight is initialized as a near zero matrix)
            a = jnp.sqrt(6. / (layer_units[i-1] + units)) * 1e-3
            w = jax.random.uniform(key=key_carry, shape=(units, layer_units[i-1]), minval=-a, maxval=a, dtype=jnp.float64)
            b = jnp.zeros(shape=(units,), dtype=jnp.float64)
        else:
            # if not first and not last layer
            key_carry, key_w = jax.random.split(key_carry, 2)
            a = jnp.sqrt(6. / (layer_units[i-1] + units))
            w = jax.random.uniform(key=key_w, shape=(units, layer_units[i-1]), minval=-a, maxval=a, dtype=jnp.float64)
            b = jnp.zeros(shape=(units,), dtype=jnp.float64)
        # append weights
        weights_and_bias.append(w)
        weights_and_bias.append(b)
    return weights_and_bias


def initialize_network(
        input_features: int,
        output_features: int,
        hidden_layers: int,
        nodes_per_layer: int,
        key: PRNGKeyArray,
        act_fun: str,
) -> list[Array]:
    """
    Returns the initialized weight and bias values for the learning component.

    Parameters
    ----------
    input_features : int
        Input dimension.
    output_features : int
        Output dimension.
    hidden_layers : int
        Number of hidden layers in the ANN.
    nodes_per_layer : int
        Nodes per (hidden) layers in the ANN.
    key : PRNGKeyArray
        Random key for initialization.
    act_fun : str
        Activation function.

    Returns
    -------
    parameters : list
        List of initialized network parameters.
    """
    # list network architecture
    net_units = [nodes_per_layer]
    for i in range(hidden_layers-1):
        net_units.append(nodes_per_layer)
    net_units.append(output_features)
    parameters = initialize_weights_and_biases(layer_units=net_units, input_features=input_features, key=key,
                                               act_fun=act_fun)
    return parameters


if __name__ == '__main__':
    # initialize a simple feedforward fully connected ANN with 2 hidden layers and 8 nodes per layer
    nu = 2
    ny = 1
    hidden_layers = 2
    nodes_per_layer = 8
    activation = 'tanh'

    ANN = jax.jit(generate_simple_ann(hidden_layers, activation))

    jax.config.update('jax_platform_name', 'cpu')
    if not jax.config.jax_enable_x64:
        jax.config.update("jax_enable_x64", True)  # Enable 64-bit computations

    params_init = initialize_network(nu, ny, hidden_layers, nodes_per_layer, jax.random.key(0), activation)

    random_input = np.random.uniform(low=-100.0, high=100.0, size=nu)
    print(f"Random input: {random_input}")
    output = ANN(random_input, params_init)
    print(f"ANN output: {output}")  # this value will be quite small as the last layer's weight and bias values are near zero
