#!/usr/bin/env python3
import glob
from myPath import dataDir
import numpy as np
import csv
import tensorflow as tf
import re
import math
from tensorflow.python import debug as tf_debug
import os
from PIL import Image


BATCH_SIZE=128
GLOBAL_STEP=50
EVAL_PERIOD=1
TRAIN_DATA_SIZE = 5000
RESCALED_X = 20
RESCALED_Y = 20
NUM_CLASSES = 3
TEST_DATA_SIZE = 1000

tf.logging.set_verbosity(tf.logging.INFO)


def cnn_model_fn(features, labels, mode):
  """Model function for CNN."""
  # Input Layer
  # Reshape X to 4-D tensor: [batch_size, width, height, channels]
  # JPG images of boudning box are 100x200 pixels, and have one color channel
  input_layer = tf.reshape(features, [-1, RESCALED_X, RESCALED_Y, 1])

  # Convolutional Layer #1
  # Computes 32 features using a 5x5 filter with ReLU activation.
  # Padding is added to preserve width and height.
  # Input Tensor Shape: [batch_size, 100, 200, 1]
  # Output Tensor Shape: [batch_size, 100, 200, 32]
  conv1 = tf.layers.conv2d(
      inputs=input_layer,
      filters=32,
      kernel_size=[5, 5],
      padding="same",
      activation=tf.nn.relu)

  # Pooling Layer #1
  # First max pooling layer with a 2x2 filter and stride of 2
  # Input Tensor Shape: [batch_size, 100, 200, 32]
  # Output Tensor Shape: [batch_size, 50, 100, 32]
  pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)

  # Convolutional Layer #2
  # Computes 64 features using a 5x5 filter.
  # Padding is added to preserve width and height.
  # Input Tensor Shape: [batch_size, 50, 100, 32]
  # Output Tensor Shape: [batch_size, 50, 100, 64]
  conv2 = tf.layers.conv2d(
      inputs=pool1,
      filters=64,
      kernel_size=[5, 5],
      padding="same",
      activation=tf.nn.relu)

  # Pooling Layer #2
  # Second max pooling layer with a 2x2 filter and stride of 2
  # Input Tensor Shape: [batch_size, 25, 50, 64]
  # Output Tensor Shape: [batch_size, 25, 50, 64]
  pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)

  # Flatten tensor into a batch of vectors
  # Input Tensor Shape: [batch_size, 25, 50, 64]
  # Output Tensor Shape: [batch_size, 25 * 50 * 64]
  pool2_flat = tf.reshape(pool2, [-1, int((RESCALED_X/4)*(RESCALED_Y/4)*64)])

  # Dense Layer
  # Densely connected layer with 1024 neurons
  # Input Tensor Shape: [batch_size, 25 * 50 * 64]
  # Output Tensor Shape: [batch_size, 1024]
  dense = tf.layers.dense(inputs=pool2_flat, units=1000, activation=tf.nn.relu)

  # Add dropout operation; 0.6 probability that element will be kept
  dropout = tf.layers.dropout(
      inputs=dense, rate=0.4, training=mode == tf.estimator.ModeKeys.TRAIN)

  # Logits layer
  # Input Tensor Shape: [batch_size, 1024]
  # Output Tensor Shape: [batch_size, 3]
  logits = tf.layers.dense(inputs=dropout, units=NUM_CLASSES)

  predictions = {
      # Generate predictions (for PREDICT and EVAL mode)
      "classes": tf.argmax(input=logits, axis=1),
      # Add `softmax_tensor` to the graph. It is used for PREDICT and by the
      # `logging_hook`.
      "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
  }
  if mode == tf.estimator.ModeKeys.PREDICT:
    return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

  # Calculate Loss (for both TRAIN and EVAL modes)
  loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)

  # Compute evaluation metrics
  accuracy = tf.metrics.accuracy(labels=labels,
                                 predictions=predictions["classes"],
                                 name='acc_op')
  metrics = {'accuracy': accuracy}
  tf.summary.scalar('accuracy', accuracy[1])

  # Configure the Training Op (for TRAIN mode)
  if mode == tf.estimator.ModeKeys.TRAIN:
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
    train_op = optimizer.minimize(
        loss=loss,
        global_step=tf.train.get_global_step())
    return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

  # Add evaluation metrics (for EVAL mode)
  return tf.estimator.EstimatorSpec(
      mode=mode, loss=loss, eval_metric_ops=metrics)

def train_input_fn(batch_size=BATCH_SIZE):
  """An input function for training"""

  # Get the path to the jpg files
  current_dir = os.getcwd()
  train_0 = glob.glob(current_dir + "train_jpg_0/*")
  train_1 = glob.glob(current_dir + "train_jpg_1/*")
  train_2 = glob.glob(current_dir + "train_jpg_2/*")

  x_train, y_train = get_train_data(TRAIN_DATA_SIZE, train_0, train_1, train_2)



  x_train = x_train.reshape(x_train.shape[0], RESCALED_X, RESCALED_Y, 1)
  input_shape = (RESCALED_X, RESCALED_Y, 1)

  x_train = x_train.astype('float32')
  x_train /= 255
  y_train = y_train.astype('int32')

  # Assume that each row of `features` corresponds to the same row as `labels`.
  assert x_train.shape[0] == y_train.shape[0]

  x_train_placeholder = tf.placeholder(x_train.dtype, x_train.shape)
  y_train_placeholder = tf.placeholder(y_train.dtype, y_train.shape)

  dataset = tf.data.Dataset.from_tensor_slices((x_train_placeholder, y_train_placeholder))
  shuffle_function = tf.contrib.data.shuffle_and_repeat(TRAIN_DATA_SIZE*NUM_CLASSES)
  dataset = dataset.apply(shuffle_function).batch(batch_size)
  iterator = dataset.make_initializable_iterator()

  with tf.Session() as sess:
    sess.run(iterator.initializer, feed_dict={x_train_placeholder: x_train,
                                          y_train_placeholder: y_train})

  # Return the dataset.
  return dataset

def eval_input_fn(batch_size=BATCH_SIZE):
  """A function to evaluate how well model perform"""
  test = glob.glob(current_dir + "test_jpg/*")

  x_test, y_test = get_test_data(TEST_DATA_SIZE, test)

  x_test = t_test.reshape(x_test.shape[0], RESCALED_X, RESCALED_Y, 1)
  input_shape = (RESCALED_X, RESCALED_Y, 1)
  x_test = x_test.astype('float32')
  x_test /= 255
  y_test = y_test.astype('int32')

  # Assume that each row of `features` corresponds to the same row as `labels`.
  assert x_test.shape[0] == y_test.shape[0]

  x_test_placeholder = tf.placeholder(x_test.dtype, x_test.shape)
  y_test_placeholder = tf.placeholder(y_test.dtype, y_test.shape)

  dataset = tf.data.Dataset.from_tensor_slices((x_test_placeholder, y_test_placeholder))
  dataset = dataset.batch(batch_size)
  iterator = dataset.make_initializable_iterator()

  with tf.Session() as sess:
    sess.run(iterator.initializer, feed_dict={x_test_placeholder: x_test,
                                          y_test_placeholder: y_test})

  # Return the dataset.
  return dataset



def get_image_data(filename):
  image = Image.open(filename)
  image = image.resize((RESCALED_X, RESCALED_Y))
  return np.array(image)

def get_train_data(n, train_0, train_1, train_2):
  train_data = []
  train_labels = []
  for f in train_0[:n]:
    train_data.append(get_image_data(f))
    train_labels.append(0)
  for f in train_1[:n]:
    train_data.append(get_image_data(f))
    train_labels.append(1)
  for f in train_2[:n]:
    train_data.append(get_image_data(f))
    train_labels.append(2)
  return np.asarray(train_data), np.asarray(train_labels)

def get_test_data(n, test):
  test_data = []
  test_labels = []
  for f in test[:n]:
    test_data.append(get_image_data(f))
    label = (f.split('/')[-1]).split('_')[0]
    test_labels.append(int(label))
  return np.asarray(test_data), np.asarray(test_labels)



def main(unused_argv):

  # Create the Estimator
  digitSeis_classifier = tf.estimator.Estimator(
      model_fn=cnn_model_fn, model_dir="/tmp/digitSeis_convnet_model")

  # Set up logging for predictions
  # Log the values in the "Softmax" tensor with label "probabilities"
  tensors_to_log = {"probabilities": "softmax_tensor"}
  logging_hook = tf.train.LoggingTensorHook(
      tensors=tensors_to_log, every_n_iter=50)

  for i in range(0, math.ceil(GLOBAL_STEP/EVAL_PERIOD)):
    # Train the model
    digitSeis_classifier.train(
        input_fn=train_input_fn,
        steps=EVAL_PERIOD,
        hooks=[logging_hook])

    # Evaluate the model and print results
    eval_results = digitSeis_classifier.evaluate(input_fn=eval_input_fn)
    print(eval_results)


if __name__ == "__main__":
  tf.app.run()



