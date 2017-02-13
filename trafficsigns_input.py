import pickle
import numpy as np

INPUT_DATA_DIR = 'traffic-signs-data'


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


def read_data_sets():
    _prepare_eval_data()

    training_file = INPUT_DATA_DIR + '/train.p'
    eval_file = INPUT_DATA_DIR + '/eval.p'
    testing_file = INPUT_DATA_DIR + '/test.p'

    with open(training_file, mode='rb') as f:
        train = pickle.load(f)

    with open(eval_file, mode='rb') as f:
        evaluate = pickle.load(f)

    with open(testing_file, mode='rb') as f:
        test = pickle.load(f)

    return Datasets(Dataset(train['features'], train['labels']),
                    Dataset(evaluate['features'], evaluate['labels']),
                    Dataset(test['features'], test['labels']))


def read_train_data():
    _prepare_eval_data()
    training_file = INPUT_DATA_DIR + '/train.p'

    with open(training_file, mode='rb') as f:
        train = pickle.load(f)

    return Dataset(train['features'], train['labels'])


def read_eval_data():
    _prepare_eval_data()
    eval_file = INPUT_DATA_DIR + '/eval.p'

    with open(eval_file, mode='rb') as f:
        eval = pickle.load(f)

    return Dataset(eval['features'], eval['labels'])


def read_test_data():
    _prepare_eval_data()

    testing_file = INPUT_DATA_DIR + '/test.p'

    with open(testing_file, mode='rb') as f:
        test = pickle.load(f)

    return Dataset(test['features'], test['labels'])


def _prepare_eval_data():
    pass
