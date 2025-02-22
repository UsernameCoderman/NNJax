# This is an example provided by Roland Ohlsson at Østfold University College

import random
from typing import Tuple

import optax
import jax.numpy as jnp
import jax
import numpy as np

BATCH_SIZE = 5
NUM_TRAIN_STEPS = 1000
RAW_TRAINING_DATA = np.random.randint( 255, size=( NUM_TRAIN_STEPS, BATCH_SIZE, 1 ) )

TRAINING_DATA = np.unpackbits(RAW_TRAINING_DATA.astype(np.uint8), axis=-1)
LABELS = jax.nn.one_hot(RAW_TRAINING_DATA % 2, 2).astype(jnp.float32).reshape(NUM_TRAIN_STEPS, BATCH_SIZE, 2)


#key can be state of the random generator

initial_params = {
    'hidden': jax.random.normal(jax.random.PRNGKey( 0 ), shape = [8, 32]),
    'output': jax.random.normal( key = jax.random.PRNGKey( 1 ) ,shape = [32, 2])
}


def net( x : jnp.ndarray, params : optax.Params ) -> jnp.ndarray:
    x = jnp.dot( x, params[ 'hidden' ] )
    x = jax.nn.relu( x )
    x = jnp.dot( x, params[ 'output' ] )
    return x


def loss( params : optax.Params, batch : jnp.ndarray, labels : jnp.ndarray ) -> jnp.ndarray:
    y_hat = net( batch, params )
    loss_value = optax.sigmoid_binary_cross_entropy( y_hat, labels ).sum( axis = -1 )
    return loss_value.mean()


def fit( params : optax.Params, optimizer : optax.GradientTransformation ) -> optax.Params:
    opt_state = optimizer.init( params )

    @jax.jit
    def step( params, opt_state, batch, labels ):
        loss_value, grads = jax.value_and_grad( loss )( params, batch, labels )
        updates, opt_state = optimizer.update( grads, opt_state, params )
        params = optax.apply_updates( params, updates )
        return params, opt_state, loss_value

    for i, ( batch, labels ) in enumerate( zip( TRAINING_DATA, LABELS ) ):
        params, opt_state, loss_value = step( params, opt_state, batch, labels )
        if i % 100 == 0:
            print(f'step {i}, loss: {loss_value}')
    return params

optimizer = optax.adam( learning_rate = 1e-2 )
params = fit( initial_params, optimizer )

