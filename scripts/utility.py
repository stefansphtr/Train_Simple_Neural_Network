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
    model = tf.keras.Sequential([
        # Input layer with 1 input feature
        tf.keras.layers.InputLayer(shape=[1]),
        
        # First hidden layer with 16 neurons and ReLU activation function
        tf.keras.layers.Dense(units=16, activation='relu'),
        
        # Second hidden layer with 8 neurons and ReLU activation function
        tf.keras.layers.Dense(units=8, activation='relu'),
        
        # Output layer with 1 neuron
        tf.keras.layers.Dense(units=1)
    ])
    
    # Print the model summary
    print(model.summary())
    
    # Returning the uncompiled model
    return model