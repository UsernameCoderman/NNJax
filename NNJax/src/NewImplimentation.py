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


def neuralNet(parampsInput, paramsOutput, input_array):
    hidden_layer= jax.nn.sigmoid(jnp.dot(input_array, parampsInput))
    output_layer= jax.nn.sigmoid(jnp.dot(hidden_layer, paramsOutput))
    return output_layer

def loss(paramsInput , paramsOutput,input,labels):
    predicted = neuralNet(paramsInput, paramsOutput, input)
    calc_loss= optax.sigmoid_binary_cross_entropy(predicted, labels) #https://optax.readthedocs.io/en/latest/api/losses.html#optax.losses.sigmoid_binary_cross_entropy
    return calc_loss

def fit(trainingdata,trianinglabels,  epochs, learning_rate)


    params_input=

    params_output=
    optimizer =

    #training loop
    for epoch in range(epochs)

        print("loss: ")



def training_steps()
