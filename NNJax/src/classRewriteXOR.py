# Importing required libraries

import random  # Standard Python library for generating random numbers (not used in this script).
from typing import Tuple  # Provides type hints for functions (Tuple not used in this script).

import optax  # Library for optimization algorithms (used for Adam optimizer).
import jax.numpy as jnp  # JAX's version of NumPy, optimized for GPUs/TPUs.
import jax  # Core JAX library, used for just-in-time (JIT) compilation and automatic differentiation.
import numpy as np  # Standard NumPy library for handling arrays and numerical computations.

# Defining constants for training

BATCH_SIZE = 2  # Number of samples processed in each training batch.
NUM_TRAIN_STEPS = 1000  # Total number of training steps (iterations over dataset).

# Generate raw training data

RAW_TRAINING_DATA = np.random.randint(2, size=(NUM_TRAIN_STEPS, BATCH_SIZE, 1))
print(RAW_TRAINING_DATA)
###
# Generates a 3D NumPy array of shape (1000, 5, 1) filled with random integers between 0 and 254.
# Each value represents an 8-bit grayscale intensity (1-byte integer).
# - NUM_TRAIN_STEPS (1000): Number of training steps (i.e., different batches).
# - BATCH_SIZE (5): Number of examples in each batch.
# - Last dimension (1): Holds a single integer per sample, making it easier to reshape later.

# Convert raw training data into binary representation (8-bit format)

#TRAINING_DATA = np.unpackbits(RAW_TRAINING_DATA.astype(np.uint8), axis=-1)
# - Converts each 8-bit integer into a binary array of shape (NUM_TRAIN_STEPS, BATCH_SIZE, 8).
# - Example: 5 → [0, 0, 0, 0, 0, 1, 0, 1] (8-bit binary).
# - This transformation allows the neural network to process the data as binary inputs.

# Create one-hot encoded labels for even/odd classification

# 16/02/25 https://numpy.org/doc/2.2/reference/generated/numpy.bitwise_xor.html

xor_labels = RAW_TRAINING_DATA[:, :, 0] ^ RAW_TRAINING_DATA[:, :, 1]  # XOR first bit with second bit

#
LABEL =


#LABELS = jax.nn.one_hot(RAW_TRAINING_DATA % 2, 2).astype(jnp.float32).reshape(NUM_TRAIN_STEPS, BATCH_SIZE, 2)
# - Computes remainder when dividing each number by 2 (RAW_TRAINING_DATA % 2).
# - Converts 0 (even) to [1, 0] and 1 (odd) to [0, 1] using one-hot encoding.
# - Reshapes the labels into (1000, 5, 2), ensuring compatibility with the model.

# Initializing neural network parameters (weights)

initial_params = {
    'hidden': jax.random.normal(shape=[8, 32], key=jax.random.PRNGKey(0)),
    # - Initializes random weights for the hidden layer (8 inputs → 32 neurons).
    # - Uses PRNGKey(0) for reproducibility.

    'output': jax.random.normal(shape=[32, 2], key=jax.random.PRNGKey(1))
    # - Initializes random weights for the output layer (32 neurons → 2 output classes).
    # - Uses PRNGKey(1) to ensure different random values from the hidden layer.
}

# Define the neural network forward pass function

def net(inputX: jnp.ndarray, paramWeights: optax.Params) -> jnp.ndarray:
    """
    Simple neural network with one hidden layer.
    Args:
        inputX: Input tensor (batch of binary feature vectors).
        paramWeights: Dictionary containing the model's weights.
    Returns:
        Output tensor (logits before softmax).
    """
    inputX = jnp.dot(inputX, paramWeights['hidden'])  # Perform matrix multiplication with hidden layer weights.
    inputX = jax.nn.relu(inputX)  # Apply ReLU activation function (introduces non-linearity).
    inputX = jnp.dot(inputX, paramWeights['output'])  # Compute final layer logits (before activation).
    return inputX  # Output shape: (batch_size, 2) - raw scores for the even/odd classes.

# Define the loss function

def loss(params: optax.Params, batch: jnp.ndarray, labels: jnp.ndarray) -> jnp.ndarray:
    """
    Computes binary cross-entropy loss for classification.
    Args:
        params: Model parameters (weights).
        batch: Input training data (binary representations).
        labels: One-hot encoded labels.
    Returns:
        Mean loss value across the batch.
    """
    y_hat = net(batch, params)  # Get predictions from the network.
    loss_value = optax.sigmoid_binary_cross_entropy(y_hat, labels).sum(axis=-1)
    # - Computes binary cross-entropy loss for each sample.
    # - Summed across the last dimension (-1) because we have two output classes.

    return loss_value.mean()  # Compute the mean loss over the batch.

# Define the training function

def fit(params: optax.Params, optimizer: optax.GradientTransformation) -> optax.Params:
    """
    Trains the neural network using the given optimizer.
    Args:
        params: Initial model parameters.
        optimizer: Optimization algorithm (Adam in this case).
    Returns:
        Updated model parameters after training.
    """
    opt_state = optimizer.init(params)  # Initialize optimizer state.

    @jax.jit  # JIT-compiles the function for efficient execution.
    def step(params, opt_state, batch, labels):
        """
        Performs one training step (forward pass, loss computation, backward pass, and parameter update).
        Args:
            params: Model parameters.
            opt_state: Current optimizer state.
            batch: Training data batch.
            labels: Corresponding labels.
        Returns:
            Updated parameters, new optimizer state, and loss value.
        """
        loss_value, grads = jax.value_and_grad(loss)(params, batch, labels)
        # - Computes the loss and its gradients with respect to the parameters.

        updates, opt_state = optimizer.update(grads, opt_state, params)
        # - Computes parameter updates using gradients.

        params = optax.apply_updates(params, updates)
        # - Applies the computed updates to the model parameters.

        return params, opt_state, loss_value  # Return updated parameters and loss.

    # Training loop
    for i, (batch, labels) in enumerate(zip(TRAINING_DATA, LABELS)):
        params, opt_state, loss_value = step(params, opt_state, batch, labels)
        # - Runs the `step` function to perform one gradient update.

        if i % 100 == 0:  # Print loss every 100 iterations.
            print(f'step {i}, loss: {loss_value}')

    return params  # Return trained model parameters.

# Initialize the optimizer

optimizer = optax.adam(learning_rate=1e-2)
# - Creates an Adam optimizer with a learning rate of 0.01.
# - Adam is an adaptive learning rate optimization algorithm that generally performs well in deep learning.

# Train the model

params = fit(initial_params, optimizer)
# - Calls the `fit` function to train the model.
# - Returns the final trained model parameters.



#comments from chatGPT