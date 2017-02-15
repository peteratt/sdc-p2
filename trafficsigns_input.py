import pickle
import numpy as np
import os.path

INPUT_DATA_DIR = 'traffic-signs-data'
EVAL_FILE_NAME = 'eval.p'


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


def _extract_validation_set(train_features, train_labels):
    total_images, rows, cols = train_features.shape

    new_X_train = np.copy(train_features)
    new_y_train = np.copy(train_labels)

    X_validate = np.empty((0, rows, cols), dtype=train_features.dtype)
    y_validate = np.array([], dtype=train_labels.dtype)

    n_img_per_class = _get_n_img_per_class(train_labels)
    start_index = 0

    for n_img in n_img_per_class:
        n_picks = int(n_img / 10)

        index_interval = list(range(start_index, start_index + n_img))
        index_list = np.random.choice(index_interval, n_picks, replace=False)
        index_list = np.sort(index_list)

        X_validate = np.append(X_validate, np.take(train_features, index_list, 0), 0)
        y_validate = np.append(y_validate, np.take(train_labels, index_list))

        new_X_train = np.delete(new_X_train, index_list, 0)
        new_y_train = np.delete(new_y_train, index_list)

        start_index = start_index + n_img

    return {
        'X_train': new_X_train,
        'y_train': new_y_train,
        'X_validate': X_validate,
        'y_validate': y_validate
    }


def _get_n_img_per_class(labels):
    n_img_per_class = []
    current_y = 0
    current_count = 0

    for y in labels:
        if y == current_y:
            current_count += 1
        else:
            current_y = y
            n_img_per_class.append(current_count)
            current_count = 1

    n_img_per_class.append(current_count)
    return n_img_per_class


def _prepare_eval_data():
    eval_file = INPUT_DATA_DIR + '/' + EVAL_FILE_NAME

    if os.path.isfile(eval_file):
        return

    training_file = INPUT_DATA_DIR + '/train.p'

    with open(training_file, mode='rb') as f:
        train = pickle.load(f)

    extracted_set = _extract_validation_set(train['features'], train['labels'])

    extracted_train = {
        'features': extracted_set['X_train'],
        'labels': extracted_set['y_train']
    }

    extracted_evaluate = {
        'features': extracted_set['X_validate'],
        'labels': extracted_set['y_validate']
    }

    with open(training_file, mode='wb') as f:
        pickle.dump(extracted_train, f, protocol=pickle.HIGHEST_PROTOCOL)

    with open(eval_file, mode='wb') as f:
        pickle.dump(extracted_evaluate, f, protocol=pickle.HIGHEST_PROTOCOL)

def main():
    _prepare_eval_data()

if __name__ == '__main__':
    main()