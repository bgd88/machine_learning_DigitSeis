#!/usr/bin/env python3

# Import necessary libraries
import tensorflow as tf
import itertools

# Enable eager execution (for testing only)
tf.enable_eager_execution()

# Check that it worked
print(tf.executing_eagerly())

# Import Data
from import_matfile import *
train_data = {
    "input": f["imInt"].value.astype('int'),
    "parameters": [f["x"].value, f["y"].value, f["nc"].value, f["nr"].value], 
    "output": f["pixID"].value.astype('int')}
print(train_data)

# Convert to Tensor
training_tensor = tf.data.Dataset.from_tensor_slices((train_data["input"], train_data["output"]))
# labels_tensor = tf.manip.reshape(tf.convert_to_tensor(train_data["output"]),[2000000])
print("DEBUG:")
print(training_tensor)

# Package Tensors
# training_tensor = [tf.data.Dataset.from_tensor_slices(input_tensor),tf.data.Dataset.from_tensor_slices(labels_tensor)]

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
for (i, (x, y)) in enumerate(training_tensor):
    # Calculate derivates of the input function with respect to its parameters
    print("DEBUG: ")
    print(x)
    print(y)
    grads = grad(neuralNet, x, y)
    # Apply the gradient to the model
    optimizer.apply_gradients(zip(grads, neuralNet.variables),
        global_step=tf.train.get_or_create_global_step())
    print("Loss at step {:04d}: {:.3f}".format(i, loss(neural_net, x, y)))

print("Final Loss: {:.3f}".format(loss(neuralNet, x, y)))



