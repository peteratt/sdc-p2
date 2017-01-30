"""A binary to train traffic signs using a single GPU.

Accuracy:
trafficsigns_train.py achieves ~86% accuracy after 100K steps (256 epochs of
data) as judged by trafficsigns_eval.py.

"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time

import pickle

import datetime
import tensorflow as tf
import numpy as np

import trafficsigns


# Process images of this size. Note that this differs from the original traffic signs
# image size of 32 x 32. If one alters this number, then the entire model
# architecture will change and any model would need to be retrained.
IMAGE_SIZE = 24

# Global constants describing the traffic signs data set.
NUM_CLASSES = 43
NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 50000
NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = 10000

# Directory for input data
INPUT_DATA_DIR = 'traffic-signs-data'

# Directory where to write event logs and checkpoint.
TRAINING_DIR = 'trafficsigns_train'

# Number of batches to run.
MAX_STEPS = 1000000

LOG_DEVICE_PLACEMENT = False

class Datasets:
    def __init__(self, train, validation, test):
        self.train = train
        self.validation = validation
        self.test = test


class Dataset:
    def __init__(self, X, y):
        self.X = X
        self.y = y
        self.num_examples = len(X)
        self.current_index = 0

    def next_example(self):
        res = self.X[self.current_index], np.array([self.y[self.current_index]])
        self.current_index = (self.current_index + 1) % self.num_examples

        return res


def _generate_image_and_label_batch(image, label, min_queue_examples, batch_size, shuffle):
    """Construct a queued batch of images and labels.

      Args:
        image: 3-D Tensor of [height, width, 3] of type.float32.
        label: 1-D Tensor of type.int32
        min_queue_examples: int32, minimum number of samples to retain
          in the queue that provides of batches of examples.
        batch_size: Number of images per batch.
        shuffle: boolean indicating whether to use a shuffling queue.

      Returns:
        images: Images. 4D tensor of [batch_size, height, width, 3] size.
        labels: Labels. 1D tensor of [batch_size] size.
      """
    # Create a queue that shuffles the examples, and then
    # read 'batch_size' images + labels from the example queue.
    num_preprocess_threads = 16
    if shuffle:
        images, label_batch = tf.train.shuffle_batch(
            [image, label],
            batch_size=batch_size,
            num_threads=num_preprocess_threads,
            capacity=min_queue_examples + 3 * batch_size,
            min_after_dequeue=min_queue_examples)
    else:
        images, label_batch = tf.train.batch(
            [image, label],
            batch_size=batch_size,
            num_threads=num_preprocess_threads,
            capacity=min_queue_examples + 3 * batch_size)

    # Display the training images in the visualizer.
    tf.image_summary('images', images)
    return images, tf.reshape(label_batch, [batch_size])


def read_data_sets(input_data_dir):
    training_file = input_data_dir + '/train.p'
    testing_file = input_data_dir + '/test.p'

    with open(training_file, mode='rb') as f:
        train = pickle.load(f)

    with open(testing_file, mode='rb') as f:
        test = pickle.load(f)

    return Datasets(Dataset(train['features'], train['labels']),
                    Dataset(train['features'], train['labels']),
                    Dataset(test['features'], test['labels']))


def distorted_inputs(data_set):
    """Construct distorted input for traffic signal training using the pickled datasets

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

    height = IMAGE_SIZE
    width = IMAGE_SIZE

    # Image processing for training the network. Note the many random
    # distortions applied to the image.

    # Randomly crop a [height, width] section of the image.
    distorted_image = tf.random_crop(reshaped_image, [height, width, 3])

    # Randomly flip the image horizontally.
    distorted_image = tf.image.random_flip_left_right(distorted_image)

    # Because these operations are not commutative, consider randomizing
    # the order their operation.
    distorted_image = tf.image.random_brightness(distorted_image,
                                                 max_delta=63)
    distorted_image = tf.image.random_contrast(distorted_image,
                                               lower=0.2, upper=1.8)

    # Subtract off the mean and divide by the variance of the pixels.
    float_image = tf.image.per_image_standardization(distorted_image)

    # Set the shapes of tensors.
    float_image.set_shape([height, width, 3])

    print(reshaped_label)
    reshaped_label.set_shape([1])

    # Ensure that the random shuffling has good mixing properties.
    min_fraction_of_examples_in_queue = 0.4
    min_queue_examples = int(data_set.num_examples *
                             min_fraction_of_examples_in_queue)
    print('Filling queue with %d traffic sign images before starting to train. '
          'This will take a few minutes.' % min_queue_examples)

    # Generate a batch of images and labels by building up a queue of examples.
    return _generate_image_and_label_batch(float_image, reshaped_label,
                                                               min_queue_examples, trafficsigns.BATCH_SIZE,
                                                               shuffle=True)


def run_training():
    """Train traffic signs for a number of steps."""
    # Get the sets of images and labels for training, validation, and
    # test on traffic signs.
    data_sets = read_data_sets(INPUT_DATA_DIR)

    # Tell TensorFlow that the model will be built into the default Graph.
    with tf.Graph().as_default():
        global_step = tf.contrib.framework.get_or_create_global_step()

        # Fill a feed with the distorted set of images and labels
        # for this particular training step.
        images, labels = distorted_inputs(data_sets.train)

        # Build a Graph that computes predictions from the inference model.
        logits = trafficsigns.inference(images)

        # Add to the Graph the Ops for loss calculation.
        loss = trafficsigns.loss(logits, labels)

        # Add to the Graph the Ops that calculate and apply gradients.
        train_op = trafficsigns.training(loss, global_step)

        class _LoggerHook(tf.train.SessionRunHook):
            """Logs loss and runtime."""

            def begin(self):
                self._step = -1

            def before_run(self, run_context):
                self._step += 1
                self._start_time = time.time()
                return tf.train.SessionRunArgs(loss)  # Asks for loss value.

            def after_run(self, run_context, run_values):
                duration = time.time() - self._start_time
                loss_value = run_values.results
                if self._step % 10 == 0:
                    num_examples_per_step = trafficsigns.BATCH_SIZE
                    examples_per_sec = num_examples_per_step / duration
                    sec_per_batch = float(duration)

                    format_str = ('%s: step %d, loss = %.2f (%.1f examples/sec; %.3f '
                                  'sec/batch)')
                    print(format_str % (datetime.datetime.now(), self._step, loss_value,
                                        examples_per_sec, sec_per_batch))

        with tf.train.MonitoredTrainingSession(
                checkpoint_dir=TRAINING_DIR,
                hooks=[tf.train.StopAtStepHook(last_step=MAX_STEPS),
                       tf.train.NanTensorHook(loss),
                       _LoggerHook()],
                config=tf.ConfigProto(
                    log_device_placement=LOG_DEVICE_PLACEMENT)) as mon_sess:
            while not mon_sess.should_stop():
                mon_sess.run(train_op)


def main(argv=None):
    if tf.gfile.Exists(TRAINING_DIR):
        tf.gfile.DeleteRecursively(TRAINING_DIR)
    tf.gfile.MakeDirs(TRAINING_DIR)
    run_training()

if __name__ == '__main__':
  tf.app.run()
