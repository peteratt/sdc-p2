import pickle
import numpy as np

TRAIN_FILE = '/train.p'
VALIDATE_FILE = '/valid.p'
TEST_FILE = '/test.p'

INPUT_DATA_DIR = 'traffic-signs-data'
EVAL_FILE_NAME = 'eval.p'


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


def read_train_data():
    training_file = INPUT_DATA_DIR + TRAIN_FILE

    with open(training_file, mode='rb') as f:
        train = pickle.load(f)

    return Dataset(train['features'], train['labels'])


def read_eval_data():
    eval_file = INPUT_DATA_DIR + VALIDATE_FILE

    with open(eval_file, mode='rb') as f:
        eval = pickle.load(f)

    return Dataset(eval['features'], eval['labels'])


def read_test_data():
    testing_file = INPUT_DATA_DIR + TEST_FILE

    with open(testing_file, mode='rb') as f:
        test = pickle.load(f)

    return Dataset(test['features'], test['labels'])