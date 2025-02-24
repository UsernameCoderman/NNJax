import optax
import jax.numpy as jnp
import jax
import numpy as np

from classRewriteXOR import optimizer

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
    return calc_loss

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

        print("loss: ")



def training_steps(params, batch, labels):
   loss_value, grad = jax.value_and_grad(loss)(params, batch, labels) # https://docs.jax.dev/en/latest/_autosummary/jax.value_and_grad.html
