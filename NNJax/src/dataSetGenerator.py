import jax.numpy as jnp
import numpy as np
import json



BATCH_SIZE = 2
NUM_OF_BATCHES = 3
BITS = 4
#Generates random pairs of 1 and 0 into an array based on batch size and number of batches, and has 2 possible values
TRAINING_DATA = jnp.array(np.random.randint(2, size=(NUM_OF_BATCHES, BATCH_SIZE, BITS)))

print(f"{TRAINING_DATA}")

filename = f"{BITS}Bit{NUM_OF_BATCHES}Batches{BATCH_SIZE}SamplesPerBatch.json"
# Convert JAX array to a list and save as JSON
with open(filename, "w") as file:
    json.dump(TRAINING_DATA.tolist(), file)

print(f"File '{filename}' has been created successfully.")
print(f"{filename}")