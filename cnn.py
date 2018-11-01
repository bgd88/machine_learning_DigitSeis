#!/usr/bin/env python3
import h5py
import glob
import pandas as pd
from myPath import dataDir
import matplotlib.pyplot as plt
import numpy as np
import csv
import tensorflow as tf
import re
import math


BATCH_SIZE=200
GLOBAL_STEP=100000
EVAL_PERIOD=250

tf.logging.set_verbosity(tf.logging.INFO)


def cnn_model_fn(features, labels, mode):
  """Model function for CNN."""
  # Input Layer
  # Reshape X to 4-D tensor: [batch_size, width, height, channels]
  # JPG images of boudning box are 100x200 pixels, and have one color channel
  input_layer = tf.reshape(features, [-1, 100, 200, 1])

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
  pool2_flat = tf.reshape(pool2, [-1, 25 * 50 * 64])

  # Dense Layer
  # Densely connected layer with 1024 neurons
  # Input Tensor Shape: [batch_size, 25 * 50 * 64]
  # Output Tensor Shape: [batch_size, 1024]
  dense = tf.layers.dense(inputs=pool2_flat, units=1024, activation=tf.nn.relu)

  # Add dropout operation; 0.6 probability that element will be kept
  dropout = tf.layers.dropout(
      inputs=dense, rate=0.4, training=mode == tf.estimator.ModeKeys.TRAIN)

  # Logits layer
  # Input Tensor Shape: [batch_size, 1024]
  # Output Tensor Shape: [batch_size, 3]
  logits = tf.layers.dense(inputs=dropout, units=3)

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
  loss = tf.losses.softmax_cross_entropy(labels=labels, logits=logits)

  # Configure the Training Op (for TRAIN mode)
  if mode == tf.estimator.ModeKeys.TRAIN:
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
    train_op = optimizer.minimize(
        loss=loss,
        global_step=tf.train.get_global_step())
    return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

  # Add evaluation metrics (for EVAL mode)
  eval_metric_ops = {
      "accuracy": tf.metrics.accuracy(
          labels=labels, predictions=predictions["classes"])}
  return tf.estimator.EstimatorSpec(
      mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)

# Reads an image from a file, decodes it into a dense tensor, and resizes it
# to a fixed shape.
def _parse_function(filename, label):
  image_string = tf.read_file(filename)
  image_decoded = tf.image.decode_jpeg(image_string)
  image_resized = tf.image.resize_images(image_decoded, [100, 200])
  return image_resized, label

def train_input_fn(batch_size=BATCH_SIZE):
  """An input function for training"""
  # Get the path to the jpg files
  filenames = glob.glob(dataDir + "train_jpg/*")

  # `labels[i]` is the label for the image in `filenames[i].
  labels = tf.constant([int(re.search('/([0-2])_', filename).group(1)) for filename in filenames])
  print(labels)

  dataset = tf.data.Dataset.from_tensor_slices((filenames, labels))
  dataset = dataset.map(_parse_function)

  # Shuffle, repeat, and batch the examples.
  dataset = dataset.shuffle(1000).repeat().batch(batch_size)

  print(dataset.output_shapes)

  # Return the dataset.
  return dataset

def eval_input_fn(batch_size=BATCH_SIZE):
  """A function to evaluate how well model perform"""
  filenames = glob.glob(dataDir + "test_jpg/*")

  # `labels[i]` is the label for the image in `filenames[i].
  labels = tf.constant([int(re.search('/([0-2])_', filename).group(1)) for filename in filenames])
  print(labels)

  dataset = tf.data.Dataset.from_tensor_slices((filenames, labels))
  dataset = dataset.map(_parse_function)
  dataset = dataset.batch(batch_size)

  print(dataset.output_shapes)

  # Return the dataset.
  return dataset



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



