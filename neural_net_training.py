#!/usr/bin/env python3

# Import necessary libraries
import tensorflow as tf

# Enable eager execution (for testing only)
tf.enable_eager_execution()

# Check that it worked
print(tf.executing_eagerly())

# Import Data
from import_matfile import *
train_data = {
    "input": f["imInt"].value,
    "parameters": [f["x"].value, f["y"].value], 
    "output": f["pixID"].value}
print(train_data)

# Define neural net architecture
class DigitSeisModel(tf.keras.Model):
    def __init__(self):
        super(DigitSeisModel, self).__init__()
        self.input_layer = tf.keras.layers.Dense(units=2000000)
        self.dense1 = tf.keras.layers.Dense(units=1000000)
        self.output_layer = tf.keras.layers.Dense(units=8000000, activation=tf.nn.softmax)

        def call(self, input):
            """Runs the model."""
            result = self.input_layer(input)
            result = self.dense1(result)
            result = self.output_layer(result)
            return result

# Create neural net
neuralNet = DigitSeisModel()

# Define loss function
def loss(model, input, targets):
    prediction = model.call(input)
    return tf.losses.categorical_cross_entropy(y_true=targets, y_predict=prediction)

# Define function to measure gradient
def grad(model, inputs, targets):
    with tf.GradientTape() as tape:
        loss_value = loss(model, inputs, targets)
        return tape.gradient(loss_value, model.variables)

# Define optimization function
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)

# Time to Train!
for (i, (x, y)) in enumerate(dataset_train):
    # Calculate derivates of the input function with respect to its parameters
    grads = grad(neuralNet, x, y)
    # Apply the gradient to the model
    optimizer.apply_gradients(zip(grads, neuralNet.variables),
        global_step=tf.train.get_or_create_global_step())
    print("Loss at step {:04d}: {:.3f}".format(i, loss(neural_net, x, y)))

print("Final Loss: {:.3f}".format(loss(neuralNet, x, y)))



