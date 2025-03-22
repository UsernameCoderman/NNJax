import optax
import jax.numpy as jnp
import jax
import numpy as np
from jax import tree_util
import json
import re

K_FOLD = 5
NEURONS = 9
#Filename for dataset
filename = "3Bit89Batches1000SamplesPerBatch.json"


# Extract dataset properties from filename
setUp = re.search(r'(\d+)Bit(\d+)Batches(\d+)SamplesPerBatch', filename)
if setUp:
    BITS, NUM_OF_BATCHES, BATCH_SIZE = map(int, setUp.groups())
    print("BITS:", BITS)
    print("NUM_OF_BATCHES:", NUM_OF_BATCHES)
    print("BATCH_SIZE:", BATCH_SIZE)

#Get data from file
with open(filename, "r") as file:
    data = json.load(file)
TRAINING_DATA = jnp.array(data)

# Generates the correct labels by checking if the sum is odd or even
XOR_LABELS = (jnp.sum(TRAINING_DATA, axis=-1) % 2 != 0).astype(jnp.int32)
# One hot encodes 1 to [1,0] and 0 to [0,1]
LABELS = jax.nn.one_hot(XOR_LABELS, 2)


# print statment to visualise the data and the labels
# for i in range(len(TRAINING_DATA[1])):
# print(f"DATA: {TRAINING_DATA[1][i]} LABEL: {LABELS[1][i]}")
#
# neural network with activation functions, for sigmoid
def forward_pass(params, input_array):
    hidden_layer = jax.nn.sigmoid(jnp.dot(input_array, params["weight_hidden"]) + params["bias_hidden"])
    output_layer = jnp.dot(hidden_layer, params["weight_output"]) + params["bias_output"]
    return output_layer


# loss function, uses cross entropy to calulcate loss
def loss(params, inputs, labels):
    predicted = forward_pass(params, inputs)
    loss_value = optax.sigmoid_binary_cross_entropy(predicted, labels)  # https://optax.readthedocs.io/en/latest/api/losses.html#optax.losses.sigmoid_binary_cross_entropy
    return jnp.mean(loss_value)


# training function,
def train_model(training_data, labels, batch_size, traning_steps, epochs=10, learning_rate=0.1):

    params = {  # https://docs.jax.dev/en/latest/_autosummary/jax.numpy.array.html
        "weight_hidden": jnp.array(np.random.randn(BITS, NEURONS) * 0.1, dtype=jnp.float32),
        # https://numpy.org/doc/2.1/reference/random/generated/numpy.random.rand.html
        "bias_hidden": jnp.array(np.zeros(NEURONS), dtype=jnp.float32),
        "weight_output": jnp.array(np.random.randn(NEURONS, 2) * 0.1, dtype=jnp.float32),
        "bias_output": jnp.array(np.zeros(2), dtype=jnp.float32),
        # https://numpy.org/devdocs/reference/generated/numpy.zeros.html
    }

    # print(f"{params}")
    optimizer = optax.adam(learning_rate)
    opt_state = optimizer.init(
        tree_util.tree_map(jnp.array, params))  # https://docs.jax.dev/en/latest/working-with-pytrees.html

    # step function
    @jax.jit
    def training_step(params, batch, labels, opt_state):
        loss_value, grad = jax.value_and_grad(loss)(params, batch, labels)  # https://docs.jax.dev/en/latest/_autosummary/jax.value_and_grad.html
        params_update, new_opt_state = optimizer.update(grad, opt_state, params)  # https://optax.readthedocs.io/en/latest/api/optimizers.html#adam
        new_parameters = optax.apply_updates(params, params_update)  # https://optax.readthedocs.io/en/latest/api/apply_updates.html
        return new_parameters, new_opt_state, loss_value

    # training loop
    for epoch in range(epochs):
        for i in range(0, len(training_data), batch_size):  # Iterate in batches
            batch = training_data[i:i + batch_size]
            current_labels = labels[i:i + batch_size]

            params, opt_state, loss_value = training_step(params, batch, current_labels, opt_state)
        print(f"Epoch {epoch + 1}, Loss: {loss_value.item():.4f}")

    return loss_value.item()




fold_size = len(TRAINING_DATA)//K_FOLD
fold_losses = []

#divide dataset into K number of folds
for fold in range(K_FOLD):
    start_of_fold, end_of_fold = fold * fold_size, (fold+1) * fold_size
    val_data, val_labels = TRAINING_DATA[start_of_fold:end_of_fold], LABELS[start_of_fold:end_of_fold]
    train_data = jnp.concatenate([TRAINING_DATA[:start_of_fold], TRAINING_DATA[end_of_fold:]], axis=0)
    train_labels = jnp.concatenate([LABELS[:start_of_fold], LABELS[end_of_fold:]], axis=0)
    print(f"Training Fold {fold + 1}/{K_FOLD}")

    fold_loss = train_model(train_data, train_labels, batch_size=BATCH_SIZE, traning_steps=len(train_data) // BATCH_SIZE)
    fold_losses.append(fold_loss)


average_kfold_loss = np.mean(fold_losses)
print(f"Average K-Fold Loss: {average_kfold_loss:.4f}")




