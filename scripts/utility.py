# Importing necessary libraries
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


def create_uncompiled_model():
    """
    This function creates an uncompiled simple neural network model
    with two hidden layers and returns it.
    """

    # Defining a simple neural network model
    model = tf.keras.Sequential(
        [
            # Input layer with 1 input feature
            tf.keras.layers.InputLayer(shape=[1]),
            # First hidden layer with 16 neurons and ReLU activation function
            tf.keras.layers.Dense(units=16, activation="relu"),
            # Second hidden layer with 8 neurons and ReLU activation function
            tf.keras.layers.Dense(units=8, activation="relu"),
            # Output layer with 1 neuron
            tf.keras.layers.Dense(units=1),
        ]
    )

    # Print the model summary
    print(model.summary())

    # Returning the uncompiled model
    return model


def adjust_learning_rate(training_dataset):
    """
    This function adjusts the learning rate for the model training process.
    It creates an uncompiled model, sets a learning rate schedule, compiles the model,
    and then fits the model to the provided dataset.

    Args:
        training_dataset: The dataset to be used for training the model.

    Returns:
        The history of the model training process.
    """

    # Create an uncompiled model
    model = create_uncompiled_model()

    # Define a learning rate schedule. The learning rate changes with each epoch.
    learning_rate_schedule = tf.keras.callbacks.LearningRateScheduler(
        lambda epoch: 1e-4 * 10 ** (epoch / 20)
    )

    # Select the Adam optimizer for the model
    adam_optimizer = tf.keras.optimizers.Adam()

    # Select the Huber loss for the model
    huber_loss = tf.keras.losses.Huber()

    # Select the Mean Squared Error as the metric for the model
    mse_metric = tf.keras.metrics.MeanSquaredError()

    # Compile the model with the Mean Squared Error loss, Adam optimizer, and accuracy as a metric
    model.compile(loss=huber_loss, optimizer=adam_optimizer, metrics=[mse_metric])

    # Fit the model to the dataset for 100 epochs, adjusting the learning rate according to the schedule
    training_history = model.fit(
        training_dataset, epochs=100, callbacks=[learning_rate_schedule]
    )

    return training_history
