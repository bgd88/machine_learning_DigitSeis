#!/usr/bin/env python3

# Import tensorflow package
import tensorflow as tf

# Enable eager execution--runs and returns values now
tf.enable_eager_execution()

# check that it worked
print(tf.executing_eagerly())

# # matmul is matrix multiplication
# x = [[2.]]
# m = tf.matmul(x, x)
# print("hello, {}".format(m))

# # constant creates a matrix/tensor
# a = tf.constant([[1, 2],
#                  [3, 4]])
# print(a)


# # Broadcasting support
# b = tf.add(a, 1)
# print(b)

# # Operator overloading is supported
# print(a * b)

# # Use NumPy values
# import numpy as np

# c = np.multiply(a, b)
# print(c)

# # Obtain numpy value from a tensor:
# print(a.numpy())

# tfe = tf.contrib.eager

# ################
# # Dynamic Control Flow
# ################

# def fizzbuzz(max_num):
#   counter = tf.constant(0)
#   max_num = tf.convert_to_tensor(max_num)
#   for num in range(max_num.numpy()):
#     num = tf.constant(num)
#     if int(num % 3) == 0 and int(num % 5) == 0:
#       print('FizzBuzz')
#     elif int(num % 3) == 0:
#       print('Fizz')
#     elif int(num % 5) == 0:
#       print('Buzz')
#     else:
#       print(num)
#     counter += 1
#   return counter

# ################
# # Build a Model
# ################

# class MySimpleLayer(tf.keras.layers.Dense):
#   def __init__(self, output_units):
#     super(MySimpleLayer, self).__init__()
#     self.output_units = output_units

#   def build(self, input_shape):
#     # The build method gets called the first time your layer is used.
#     # Creating variables on build() allows you to make their shape depend
#     # on the input shape and hence removes the need for the user to specify
#     # full shapes. It is possible to create variables during __init__() if
#     # you already know their full shapes.
#     self.kernel = self.add_variable(
#       "kernel", [input_shape[-1], self.output_units])

#   def call(self, input):
#     # Override call() instead of __call__ so we can perform some bookkeeping.
#     return tf.matmul(input, self.kernel)



# model = tf.keras.Sequential([
#   tf.keras.layers.Dense(10, input_shape=(784,)),  # must declare input shape
#   tf.keras.layers.Dense(10)
# ])

# class MNISTModel(tf.keras.Model):
#   def __init__(self):
#     super(MNISTModel, self).__init__()
#     self.dense1 = tf.keras.layers.Dense(units=10)
#     self.dense2 = tf.keras.layers.Dense(units=10)

#   def call(self, input):
#     """Run the model."""
#     result = self.dense1(input)
#     result = self.dense2(result)
#     result = self.dense2(result)  # reuse variables from dense2 layer
#     return result

# model = MNISTModel()

# ################
# # Eager Training: Computing Gradients
# ################

# w = tf.contrib.eager.Variable([[1.0]])
# with tf.GradientTape() as tape:
#   loss = w * w

# grad = tape.gradient(loss, w)
# print(grad)

# # A toy dataset of points around 3 * x + 2
# NUM_EXAMPLES = 1000
# training_inputs = tf.random_normal([NUM_EXAMPLES])
# noise = tf.random_normal([NUM_EXAMPLES])
# training_outputs = training_inputs * 3 + 2 + noise

# def prediction(input, weight, bias):
#   return input * weight + bias

# # A loss function using mean-squared error
# def loss(weights, biases):
#   error = prediction(training_inputs, weights, biases) - training_outputs
#   return tf.reduce_mean(tf.square(error))

# # Return the derivative of loss with respect to weight and bias
# def grad(weights, biases):
#   with tf.GradientTape() as tape:
#     loss_value = loss(weights, biases)
#   return tape.gradient(loss_value, [weights, biases])

# train_steps = 200
# learning_rate = 0.01
# # Start with arbitrary values for W and B on the same batch of data
# W = tf.contrib.eager.Variable(5.)
# B = tf.contrib.eager.Variable(10.)

# print("Initial loss: {:.3f}".format(loss(W, B)))

# for i in range(train_steps):
#   dW, dB = grad(W, B)
#   W.assign_sub(dW * learning_rate)
#   B.assign_sub(dB * learning_rate)
#   if i % 20 == 0:
#     print("Loss at step {:03d}: {:.3f}".format(i, loss(W, B)))

# print("Final loss: {:.3f}".format(loss(W, B)))
# print("W = {}, B = {}".format(W.numpy(), B.numpy()))


# ################
# # Eager Training: Train a Model
# ################

# # Create a tensor representing a blank image
# batch = tf.zeros([1, 1, 784])
# print(batch.shape)

# result = model(batch)
# print(result)

# import dataset  # download dataset.py file
# dataset_train = dataset.train('./datasets').shuffle(60000).repeat(4).batch(32)

# def loss(model, x, y):
#   prediction = model(x)
#   return tf.losses.sparse_softmax_cross_entropy(labels=y, logits=prediction)

# def grad(model, inputs, targets):
#   with tf.GradientTape() as tape:
#     loss_value = loss(model, inputs, targets)
#   return tape.gradient(loss_value, model.variables)

# optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)

# x, y = iter(dataset_train).next()
# print("Initial loss: {:.3f}".format(loss(model, x, y)))

# # Training loop

# # Slow version
# for (i, (x, y)) in enumerate(dataset_train):
#   # Calculate derivatives of the input function with respect to its parameters.
#   grads = grad(model, x, y)
#   # Apply the gradient to the model
#   optimizer.apply_gradients(zip(grads, model.variables),
#                             global_step=tf.train.get_or_create_global_step())

# # Fast version, doesn't work on Mac
# # with tf.device("/gpu:0"):
# #   for (i, (x, y)) in enumerate(dataset_train):
# #     # minimize() is equivalent to the grad() and apply_gradients() calls.
# #     optimizer.minimize(lambda: loss(model, x, y),
# #                        global_step=tf.train.get_or_create_global_step())
#   if i % 200 == 0:
#     print("Loss at step {:04d}: {:.3f}".format(i, loss(model, x, y)))

# print("Final loss: {:.3f}".format(loss(model, x, y)))

################
# Eager Training: Variables and optimizers
################

class Model(tf.keras.Model):
  def __init__(self):
    super(Model, self).__init__()
    self.W = tf.contrib.eager.Variable(5., name='weight')
    self.B = tf.contrib.eager.Variable(10., name='bias')
  def call(self, inputs):
    return inputs * self.W + self.B

# A toy dataset of points around 3 * x + 2
NUM_EXAMPLES = 2000
training_inputs = tf.random_normal([NUM_EXAMPLES])
noise = tf.random_normal([NUM_EXAMPLES])
training_outputs = training_inputs * 3 + 2 + noise

# The loss function to be optimized
def loss(model, inputs, targets):
  error = model(inputs) - targets
  return tf.reduce_mean(tf.square(error))

def grad(model, inputs, targets):
  with tf.GradientTape() as tape:
    loss_value = loss(model, inputs, targets)
  return tape.gradient(loss_value, [model.W, model.B])

# Define:
# 1. A model.
# 2. Derivatives of a loss function with respect to model parameters.
# 3. A strategy for updating the variables based on the derivatives.
model = Model()
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)

print("Initial loss: {:.3f}".format(loss(model, training_inputs, training_outputs)))

# Training loop
for i in range(300):
  grads = grad(model, training_inputs, training_outputs)
  optimizer.apply_gradients(zip(grads, [model.W, model.B]),
                            global_step=tf.train.get_or_create_global_step())
  if i % 20 == 0:
    print("Loss at step {:03d}: {:.3f}".format(i, loss(model, training_inputs, training_outputs)))

print("Final loss: {:.3f}".format(loss(model, training_inputs, training_outputs)))
print("W = {}, B = {}".format(model.W.numpy(), model.B.numpy()))


################
# Use objects for state during eager execution
################

x = tf.contrib.eager.Variable(10.)

checkpoint = tf.train.Checkpoint(x=x)  # save as "x"

x.assign(2.)   # Assign a new value to the variables and save.
save_path = checkpoint.save('./ckpt/')

x.assign(11.)  # Change the variable after saving.

# Restore values from the checkpoint
checkpoint.restore(save_path)

print(x)


