import optax
import jax.numpy as jnp
import jax
import numpy as np

BATCH_SIZE = 32
NUM_TRAIN_STEPS = 1000
# https://numpy.org/doc/2.1/reference/random/generated/numpy.random.randint.html
RAW_TRAINING_DATA =  jnp.array(np.random.randint(2, size=(NUM_TRAIN_STEPS, BATCH_SIZE, 2)))
print(RAW_TRAINING_DATA)

# 16/02/25 https://numpy.org/doc/2.2/reference/generated/numpy.bitwise_xor.html
XOR_Labels = RAW_TRAINING_DATA[:, :, 0] ^ RAW_TRAINING_DATA[:, :, 1]  # compare first with second column, deconstructs

# https://docs.jax.dev/en/latest/_autosummary/jax.nn.one_hot.html
LABELS = jax.nn.one_hot(XOR_Labels, 2)

# https://docs.jax.dev/en/latest/_autosummary/jax.random.normal.html 18/02/2025

initial_params = {
    'hidden': jax.random.normal(jax.random.PRNGKey(0), shape=(2, 32)),
    'output': jax.random.normal(jax.random.PRNGKey(1), shape=(32, 2))
}


def net(inputX: jnp.ndarray, paramWeights: optax.Params) -> jnp.ndarray:
    inputX = jnp.dot(inputX, paramWeights['hidden'])  #  https://docs.jax.dev/en/latest/_autosummary/jax.numpy.dot.html 18/02/
    inputX = jax.nn.relu(inputX)  # https://docs.jax.dev/en/latest/_autosummary/jax.nn.sigmoid.html
    inputX = jnp.dot(inputX, paramWeights['output'])  #
    return inputX

def loss(params: optax.Params, batch: jnp.ndarray, labels: jnp.ndarray) -> jnp.ndarray:
    y_hat = net(batch, params)
    loss_value = optax.sigmoid_binary_cross_entropy(y_hat, labels).sum(axis=-1)
    return loss_value.mean()

def fit(params: optax.Params, optimizer: optax.GradientTransformation) -> optax.Params:
    opt_state = optimizer.init(params)

    @jax.jit
    def step(params, opt_state, batch, labels):
        loss_value, grads = jax.value_and_grad(loss)(params, batch, labels)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss_value

    for i, (batch, labels) in enumerate(zip(RAW_TRAINING_DATA, LABELS)):
        params, opt_state, loss_value = step(params, opt_state, batch, labels)
        if i % 100 == 0:
            print(f'step {i}, loss: {loss_value}')
    return params


optimizer = optax.adam(learning_rate=1e-2)
params = fit(initial_params, optimizer)