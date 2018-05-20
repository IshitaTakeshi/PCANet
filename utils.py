import numpy as np
import pickle

from chainer.datasets import get_mnist, get_cifar10
from chainer.cuda import get_device


def check_gpu_enabled():
    """Return true if GPU is available"""
    return get_device().id >= 0


def reshape_dataset(train, test):
    def channels_last(X):
        X = np.swapaxes(X, 1, 2)
        X = np.swapaxes(X, 2, 3)
        return X

    X_train, y_train = train._datasets[0], train._datasets[1]
    X_test, y_test = test._datasets[0], test._datasets[1]
    X_train, X_test = channels_last(X_train), channels_last(X_test)
    return ((X_train, y_train), (X_test, y_test))



def save_model(model, filename):
    with open(filename, "wb") as f:
        pickle.dump(model, f)


def pick(train_set, test_set, n_train, n_test):
    images_train, y_train = train_set
    images_test, y_test = test_set
    train_set = images_train[:n_train], y_train[:n_train]
    test_set = images_test[:n_test], y_test[:n_test]
    return train_set, test_set


def load_cifar():
    train, test = get_cifar10(ndim=3)
    return reshape_dataset(train, test)


def load_mnist():
    train, test = get_mnist(ndim=3)
    return reshape_dataset(train, test)


def concatenate_dicts(*dicts):
    """Concatenate multiple directories into one"""
    merged = []
    for d in dicts:
        merged += list(d.items())
    return dict(merged)
