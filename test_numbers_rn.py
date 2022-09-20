# coding=<UTF-8>
from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf
import tensorflow_datasets as tfds

import math
import numpy as np
import matplotlib.pyplot as plt
import logging

logger = tf.get_logger()

logger.setLevel(logging.ERROR)

# We get data set and metadata from set
dataset, metadata = tfds.load("mnist", as_supervised=True, with_info=True)
# 60,000 data for training and 10,000 data for validation
train_dataset, test_dataset = dataset["train"], dataset["test"]

# We define text labels for each possible response from the network
class_names = [
    "Cero",
    "Uno",
    "Dos",
    "Tres",
    "Cuatro",
    "Cinco",
    "Seis",
    "Siete",
    "Ocho",
    "Nueve",
]

# We obtain the number of examples in variables for later use.
num_train_examples = metadata.splits["train"].num_examples
num_test_examples = metadata.splits["test"].num_examples

# Normalize numbers 0-255
def normalize(images, labels):
    images = tf.cast(images, tf.float32)
    images /= 255
    return images, labels


train_dataset = train_dataset.map(normalize)
test_dataset = train_dataset.map(normalize)

# Network structure
model = tf.keras.Sequential(
    [
        # Input layer 784 neurons specifying that it will arrive in a 28x28 square layer.
        tf.keras.layers.Flatten(input_shape=(28, 28, 1)),
        # Two dense hidden layers of 64 each
        tf.keras.layers.Dense(64, activation=tf.nn.relu),
        tf.keras.layers.Dense(64, activation=tf.nn.relu),
        # Output layer
        tf.keras.layers.Dense(10, activation=tf.nn.softmax),
    ]
)

# Compile model specifying cost function
# Indicate functions to be used
model.compile(
    optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"]
)

# Batch learning of 32 each batch
BATCHSIZE = 32
# Randomly reorder data in batches of 32
train_dataset = train_dataset.repeat().shuffle(num_train_examples).batch(BATCHSIZE)
test_dataset = test_dataset.batch(BATCHSIZE)

# Perform apprenticeship
model.fit(
    # Specify times: number of training laps
    train_dataset,
    epochs=5,
    steps_per_epoch=math.ceil(num_train_examples / BATCHSIZE),
)

# Evaluating the input model against the test dataset
test_loss, test_accuracy = model.evaluate(
    test_dataset, steps=math.ceil(num_test_examples / 32)
)

print("Result: ", test_accuracy)