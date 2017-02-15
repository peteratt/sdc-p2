# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Evaluation for CIFAR-10.

Accuracy:
cifar10_train.py achieves 83.0% accuracy after 100K steps (256 epochs
of data) as judged by cifar10_eval.py.

Speed:
On a single Tesla K40, cifar10_train.py processes a single batch of 128 images
in 0.25-0.35 sec (i.e. 350 - 600 images /sec). The model reaches ~86%
accuracy after 100K steps in 8 hours of training time.

Usage:
Please see the tutorial and website for how to download the CIFAR-10
data set, compile the program and train the model.

http://tensorflow.org/tutorials/deep_cnn/
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import math

import numpy as np
import tensorflow as tf

import trafficsigns
import trafficsigns_input

NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = 10000

EVAL_DIR = 'trafficsigns_eval'
TEST_DIR = 'trafficsigns_test'

# Data sources
EVAL_DATA = 'traffic-signs-data/valid.p'
TEST_DATA = 'traffic-signs-data/test.p'

# Directory where to read training checkpoints
CHECKPOINT_DIR = 'trafficsigns_train'

MODE_EVAL = 0
MODE_TEST = 1

mode = MODE_EVAL


def eval_once(saver, summary_writer, top_k_op, summary_op):
    """Run Eval once.

    Args:
      saver: Saver.
      summary_writer: Summary writer.
      top_k_op: Top K op.
      summary_op: Summary op.
    """
    with tf.Session() as sess:
        ckpt = tf.train.get_checkpoint_state(CHECKPOINT_DIR)
        if ckpt and ckpt.model_checkpoint_path:
            # Restores from checkpoint
            saver.restore(sess, ckpt.model_checkpoint_path)
            # Assuming model_checkpoint_path looks something like:
            #   /my-favorite-path/cifar10_train/model.ckpt-0,
            # extract global_step from it.
            global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
        else:
            print('No checkpoint file found')
            return

        # Start the queue runners.
        coord = tf.train.Coordinator()
        threads = []

        try:
            for qr in tf.get_collection(tf.GraphKeys.QUEUE_RUNNERS):
                threads.extend(qr.create_threads(sess, coord=coord, daemon=True,
                                                 start=True))

            num_iter = int(math.ceil(NUM_EXAMPLES_PER_EPOCH_FOR_EVAL / trafficsigns.BATCH_SIZE))
            true_count = 0  # Counts the number of correct predictions.
            total_sample_count = num_iter * trafficsigns.BATCH_SIZE
            step = 0
            while step < num_iter and not coord.should_stop():
                predictions = sess.run([top_k_op])
                true_count += np.sum(predictions)
                step += 1

            # Compute precision @ 1.
            precision = true_count / total_sample_count
            print('%s: precision @ 1 = %.3f' % (datetime.now(), precision))

            summary = tf.Summary()
            summary.ParseFromString(sess.run(summary_op))
            summary.value.add(tag='Precision @ 1', simple_value=precision)
            summary_writer.add_summary(summary, global_step)
        except Exception as e:  # pylint: disable=broad-except
            coord.request_stop(e)

        coord.request_stop()
        coord.join(threads, stop_grace_period_secs=10)


def _generate_image_and_label_batch(image, label, min_queue_examples, batch_size):
    """Construct a queued batch of images and labels.

      Args:
        image: 3-D Tensor of [height, width, 3] of type.float32.
        label: 1-D Tensor of type.int32
        min_queue_examples: int32, minimum number of samples to retain
          in the queue that provides of batches of examples.
        batch_size: Number of images per batch.

      Returns:
        images: Images. 4D tensor of [batch_size, height, width, 3] size.
        labels: Labels. 1D tensor of [batch_size] size.
      """
    # Create a queue that shuffles the examples, and then
    # read 'batch_size' images + labels from the example queue.
    num_preprocess_threads = 16
    images, label_batch = tf.train.batch(
        [image, label],
        batch_size=batch_size,
        num_threads=num_preprocess_threads,
        capacity=min_queue_examples + 3 * batch_size)

    # Display the training images in the visualizer.
    tf.image_summary('images', images)
    return images, tf.reshape(label_batch, [batch_size])


def inputs(data_set):
    """Construct input for traffic signal evaluation using the pickled datasets

    Args:
        data_set: Pickled dataset for traffic signs

    Returns:
        images: Images. 4D tensor of [batch_size, IMAGE_SIZE, IMAGE_SIZE, 3] size.
        labels: Labels. 1D tensor of [batch_size] size.
    """
    image_feed, label_feed = data_set.next_example()

    # Apply distortions
    reshaped_image = tf.cast(image_feed, tf.float32)
    reshaped_label = tf.cast(label_feed, tf.int32)

    height = trafficsigns.IMAGE_SIZE
    width = trafficsigns.IMAGE_SIZE

    # Image processing for evaluation.
    # Crop the central [height, width] of the image.
    resized_image = tf.image.resize_image_with_crop_or_pad(reshaped_image,
                                                           width, height)

    # Subtract off the mean and divide by the variance of the pixels.
    float_image = tf.image.per_image_standardization(resized_image)

    # Set the shapes of tensors.
    float_image.set_shape([height, width, 3])
    reshaped_label.set_shape([1])

    # Ensure that the random shuffling has good mixing properties.
    min_fraction_of_examples_in_queue = 0.4
    min_queue_examples = int(data_set.num_examples *
                             min_fraction_of_examples_in_queue)
    print('Filling queue with %d traffic sign images before starting to evaluate. '
          'This will take a few minutes.' % min_queue_examples)

    # Generate a batch of images and labels by building up a queue of examples.
    return _generate_image_and_label_batch(float_image, reshaped_label,
                                           min_queue_examples, trafficsigns.BATCH_SIZE)


def evaluate():
    """Eval traffic signs for a number of steps."""
    with tf.Graph().as_default() as g:
        # Get evaluation images and labels for traffic signs

        if mode == MODE_EVAL:
            eval_data = trafficsigns_input.read_eval_data()
            images, labels = inputs(eval_data)
        elif mode == MODE_TEST:
            test_data = trafficsigns_input.read_test_data()
            images, labels = inputs(test_data)
        else:
            raise Exception

        # Build a Graph that computes the logits predictions from the
        # inference model.
        logits = trafficsigns.inference(images)

        # Calculate predictions.
        top_k_op = tf.nn.in_top_k(logits, labels, 1)

        # Restore the moving average version of the learned variables for eval.
        variable_averages = tf.train.ExponentialMovingAverage(
            trafficsigns.MOVING_AVERAGE_DECAY)
        variables_to_restore = variable_averages.variables_to_restore()
        saver = tf.train.Saver(variables_to_restore)

        # Build the summary operation based on the TF collection of Summaries.
        summary_op = tf.summary.merge_all()

        summary_writer = tf.summary.FileWriter(EVAL_DIR, g)

        while True:
            eval_once(saver, summary_writer, top_k_op, summary_op)



def main(argv=None):  # pylint: disable=unused-argument
    if tf.gfile.Exists(EVAL_DIR):
        tf.gfile.DeleteRecursively(EVAL_DIR)
    tf.gfile.MakeDirs(EVAL_DIR)
    evaluate()


if __name__ == '__main__':
    tf.app.run()
