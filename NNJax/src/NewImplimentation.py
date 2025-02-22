import optax
import jax.numpy as jnp
import jax
import numpy as np

from classRewriteXOR import params

BATCH_SIZE = 32
NUM_TRAIN_STEPS = 1000
# https://numpy.org/doc/2.1/reference/random/generated/numpy.random.randint.html
RAW_TRAINING_DATA =  jnp.array(np.random.randint(2, size=(NUM_TRAIN_STEPS, BATCH_SIZE, 2)))
print(RAW_TRAINING_DATA)


# 16/02/25 https://numpy.org/doc/2.2/reference/generated/numpy.bitwise_xor.html
XOR_Labels = RAW_TRAINING_DATA[:, :, 0] ^ RAW_TRAINING_DATA[:, :, 1]  # compare first with second column, deconstructs

# https://docs.jax.dev/en/latest/_autosummary/jax.nn.one_hot.html
LABELS = jax.nn.one_hot(XOR_Labels, 2)


def neuralNet(params, inputs)
    hidden_layer=
    output_layer=
    return output_layer

def loss(params,input,labels)
    predicted = neuralNet()
    calc_loss= loss fuciton
    return calc_loss

def fit(trainingdata,trianinglabels,  epochs, learning_rate))


    params=

    optimizer =

    #training loop
    for epoch in range(epochs)




def training_steps()
