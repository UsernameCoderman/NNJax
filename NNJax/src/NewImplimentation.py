import optax
import jax.numpy as jnp
import jax
import numpy as np
from jax import tree_util
import json


BATCH_SIZE = 10000
NUM_OF_BATCHES = 50
BITS=2

filename = "2Bit10kBatch50Samples.json"
with open(filename, "r") as file:
    data = json.load(file)

TRAINING_DATA = jnp.array(data)


#Generates the correct labels using native XOR funtionality
XOR_LABELS = TRAINING_DATA[:, :, 0] ^ TRAINING_DATA[:, :, 1]  # compare first with second column, deconstructs

#One hot encodes 1 to [1,0] and 0 to [0,1]
LABELS = jax.nn.one_hot(XOR_LABELS, 2)

#print statment to visualise the data and the labels
#for i in range(len(TRAINING_DATA[1])):
   # print(f"DATA: {TRAINING_DATA[1][i]} LABEL: {LABELS[1][i]}")
#
#neural network with activation functions, for sigmoid
def forward_pass(params, input_array):
    hidden_layer = jax.nn.sigmoid(jnp.dot(input_array, params["weight_hidden"]) + params["bias_hidden"])
    output_layer = jnp.dot(hidden_layer, params["weight_output"]) + params["bias_output"]
    return output_layer

#loss function, uses cross entropy to calulcate loss
def loss(params, inputs, labels):
    predicted = forward_pass(params, inputs)
    loss_value = optax.sigmoid_binary_cross_entropy(predicted, labels)  # https://optax.readthedocs.io/en/latest/api/losses.html#optax.losses.sigmoid_binary_cross_entropy
    return jnp.mean(loss_value)

#training function,
def train_model(training_data, labels, traning_steps, epochs=10, learning_rate=0.1):


    neurons = int(round(BATCH_SIZE/10,0))

    params = { # https://docs.jax.dev/en/latest/_autosummary/jax.numpy.array.html
        "weight_hidden": jnp.array(np.random.randn(2, neurons) * 0.1, dtype=jnp.float32),   # https://numpy.org/doc/2.1/reference/random/generated/numpy.random.rand.html
        "bias_hidden": jnp.array(np.zeros(neurons), dtype=jnp.float32),
        "weight_output": jnp.array(np.random.randn(neurons, 2) * 0.1, dtype=jnp.float32),
        "bias_output": jnp.array(np.zeros(2), dtype=jnp.float32),# https://numpy.org/devdocs/reference/generated/numpy.zeros.html
    }

    #print(f"{params}")
    optimizer = optax.adam(learning_rate)
    opt_state = optimizer.init(tree_util.tree_map(jnp.array, params)) # https://docs.jax.dev/en/latest/working-with-pytrees.html

    #step function
    @jax.jit
    def training_step(params, batch, labels, opt_state):
        loss_value, grad = jax.value_and_grad(loss)(params, batch, labels)  # https://docs.jax.dev/en/latest/_autosummary/jax.value_and_grad.html
        params_update, new_opt_state = optimizer.update(grad, opt_state, params)  # https://optax.readthedocs.io/en/latest/api/optimizers.html#adam
        new_parameters = optax.apply_updates(params, params_update)  # https://optax.readthedocs.io/en/latest/api/apply_updates.html
        return new_parameters, new_opt_state, loss_value

    # training loop
    for epoch in range(epochs):
        for i in range(traning_steps):
            batch = training_data[i]
            current_labels = labels[i]
            params, opt_state, loss_value = training_step(params, batch, current_labels, opt_state)
        print(f"Epoch {epoch + 1}, Loss: {loss_value.item():.4f}")


train_model(TRAINING_DATA, LABELS, NUM_OF_BATCHES)