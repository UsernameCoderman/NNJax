import optax
import jax.numpy as jnp
import jax
import numpy as np
from keras.src.legacy.backend import update
from tensorflow.python.ops.gen_batch_ops import batch

from classRewriteXOR import optimizer, params

BATCH_SIZE = 32
NUM_TRAIN_STEPS = 1000
# https://numpy.org/doc/2.1/reference/random/generated/numpy.random.randint.html
RAW_TRAINING_DATA =  jnp.array(np.random.randint(2, size=(NUM_TRAIN_STEPS, BATCH_SIZE, 2)))
print(RAW_TRAINING_DATA)


# 16/02/25 https://numpy.org/doc/2.2/reference/generated/numpy.bitwise_xor.html
XOR_Labels = RAW_TRAINING_DATA[:, :, 0] ^ RAW_TRAINING_DATA[:, :, 1]  # compare first with second column, deconstructs

# https://docs.jax.dev/en/latest/_autosummary/jax.nn.one_hot.html
LABELS = jax.nn.one_hot(XOR_Labels, 2)


def neuralNet(params, input_array):
    hidden_layer= jax.nn.sigmoid(jnp.dot(input_array, params["weightHidden"]) + params["biasHidden"])
    output_layer= jax.nn.sigmoid(jnp.dot(hidden_layer, params["weightOutput"]) + params["biasOutput"])
    return output_layer

def loss(params,input,labels):
    predicted = neuralNet(params, input)
    calc_loss= optax.sigmoid_binary_cross_entropy(predicted, labels) #https://optax.readthedocs.io/en/latest/api/losses.html#optax.losses.sigmoid_binary_cross_entropy
    return jnp.mean(calc_loss)

def training_step(params, batch, labels, optimizer: optax.GradientTransformation, optimizerState):
    loss_value, grad = jax.value_and_grad(loss)(params, batch, labels) # https://docs.jax.dev/en/latest/_autosummary/jax.value_and_grad.html
    updateForParam, newOptimizerState = optimizer.update(grad,optimizerState, params) #https://optax.readthedocs.io/en/latest/api/optimizers.html#adam
    new_parameters = optax.apply_updates(params,updateForParam) #https://optax.readthedocs.io/en/latest/api/apply_updates.html
    return  new_parameters, newOptimizerState, loss_value


def fit(trainingdata,traininglabels,  epochs, learning_rate):

    optimizer = optax.adam(learning_rate)

    params = {
        "weightHidden" : 0.1,
        "biasHidden" : 0.1,
        "weightOutput" : 0.1,
        "biasOutput" : 0.1
    }

    optimize_state = optimizer.init(params)

    #training loop
    for epoch in range(epochs):
        for batch, labels in zip(trainingdata, traininglabels): # https://www.w3schools.com/python/ref_func_zip.asp
            params, optimize_state, loss_value = training_step(params, batch, labels, optimizer, optimize_state)

        print(f"Epoch {epoch+1}, Loss: {loss_value.item():.4f}")



