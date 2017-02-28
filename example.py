from os.path import exists, join
import pickle
import gzip
from urllib.request import urlretrieve

import numpy as np
from mnist import MNIST
from sklearn.datasets import fetch_mldata
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.utils import shuffle
from pcanet import PCANet


def load_mnist():
    # mnist = fetch_mldata("MNIST original", data_home=".")
    # print(mnist["target"].shape)
    # print(mnist["data"].shape)
    # exit(0)
    # url = "http://deeplearning.net/data/mnist/mnist.pkl.gz"
    # mnist_compressed = "mnist.pkl.gz"

    # if not exists(mnist_compressed):
    #     print("Downloading MNIST")
    #     urlretrieve(url, mnist_compressed)

    # # Load the dataset
    # with gzip.open(mnist_compressed, "rb") as f:
    #     u = pickle._Unpickler(f)
    #     u.encoding = "latin1"
    #     data = u.load()
    # data = [(X.reshape(-1, 28, 28), y) for X, y in data]

    mnist = MNIST("mnist")
    X_train, y_train = mnist.load_training()
    X_test, y_test = mnist.load_testing()

    X_train, y_train = np.array(X_train), np.array(y_train)
    X_train = X_train.reshape(-1, 28, 28)
    X_test, y_test = np.array(X_test), np.array(y_test)
    X_test = X_test.reshape(-1, 28, 28)
    train_set = X_train, y_train
    test_set = X_test, y_test
    return train_set, test_set


def params_to_str(params):
    keys = sorted(params.keys())
    return "_".join([key + "_" + str(params[key]) for key in keys])


if __name__ == "__main__":
    train_set, test_set = load_mnist()

    images_train, y_train = train_set
    images_test, y_test = test_set

    n_train = len(y_train)
    n_test = len(y_test)

    images_train, y_train = images_train[:n_train], y_train[:n_train]
    images_test, y_test = images_test[:n_test], y_test[:n_test]

    params = {
        "image_shape": 28,
        "filter_shape_l1": 4, "step_shape_l1": 2, "n_l1_output": 3,
        "filter_shape_l2": 4, "step_shape_l2": 1, "n_l2_output": 3,
        "block_shape": 5
    }

    # from ensemble import Bagging
    # model = Bagging(
    #     n_estimators=400,
    #     sampling_ratio=0.3,
    #     n_jobs=4,
    #     image_shape=28,
    #     filter_shape_l1=2, step_shape_l1=1, n_l1_output=4,
    #     filter_shape_l2=2, step_shape_l2=1, n_l2_output=4,
    #     block_shape=2)

    model = PCANet(**params)
    model.validate_structure()
    model.fit(images_train)
    X_train = model.transform(images_train)
    X_test = model.transform(images_test)
    del(model)
    del(images_train)
    del(images_test)

    model = RandomForestClassifier(n_estimators=100, random_state=1234, n_jobs=-1)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print("Accuracy: " + str(accuracy))

    filename = (params_to_str(params) +
                "_n_train_" + str(n_train) +
                "_n_test_" + str(n_test) +
                "_accuracy_" + str(accuracy) + ".pkl")

    with open(join("pickles", filename), "wb") as f:
        pickle.dump(model, f)
