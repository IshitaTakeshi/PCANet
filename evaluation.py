from os.path import exists, join
import pickle
import json
import gzip
from urllib.request import urlretrieve
from argparse import ArgumentParser
import hashlib
import time
import timeit
from multiprocessing import cpu_count

import numpy as np
from mnist import MNIST
from sklearn.datasets import fetch_mldata
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.utils import shuffle

from pcanet import PCANet
from ensemble import Bagging


def load_mnist():
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


def run_classifier(X_train, X_test, y_train, y_test):
    model = RandomForestClassifier(n_estimators=100, random_state=1234, n_jobs=-1)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return y_test, y_pred


def run_pcanet_normal(estimator_params,
                      images_train, images_test, y_train, y_test):
    model = PCANet(**estimator_params)
    model.validate_structure()

    t1 = timeit.default_timer()
    model.fit(images_train)
    t2 = timeit.default_timer()
    training_time = t2 - t1

    X_train = model.transform(images_train)
    X_test = model.transform(images_test)

    y_test, y_pred = run_classifier(X_train, X_test, y_train, y_test)
    accuracy = accuracy_score(y_test, y_pred)

    return model, accuracy, training_time


# TODO Change n_estimators and sampling_ratio on evaluation
def run_pcanet_ensemble(ensemble_params, estimator_params,
                        images_train, images_test, y_train, y_test):
    model = Bagging(
        n_estimators=ensemble_params["n_estimators"],
        sampling_ratio=ensemble_params["sampling_ratio"],
        n_jobs=ensemble_params["n_jobs"],
        **estimator_params)

    t1 = timeit.default_timer()
    model.fit(images_train)
    t2 = timeit.default_timer()
    training_time = t2 - t1

    X_train = model.transform(images_train)
    X_test = model.transform(images_test)

    y_test, y_pred = run_classifier(X_train, X_test, y_train, y_test)
    accuracy = accuracy_score(y_test, y_pred)

    return model, accuracy, training_time


def parse_args():
    parser = ArgumentParser()

    parser.add_argument("--image-shape", dest="image_shape", type=int,
            required=True)
    parser.add_argument("--filter-shape-l1", dest="filter_shape_l1", type=int,
            required=True)
    parser.add_argument("--step-shape-l1", dest="step_shape_l1", type=int,
            required=True)
    parser.add_argument("--n-l1-output", dest="n_l1_output", type=int,
            required=True)
    parser.add_argument("--filter-shape-l2", dest="filter_shape_l2", type=int,
            required=True)
    parser.add_argument("--step-shape-l2", dest="step_shape_l2", type=int,
            required=True)
    parser.add_argument("--n-l2-output", dest="n_l2_output", type=int,
            required=True)
    parser.add_argument("--block-shape", dest="block_shape", type=int,
            required=True)
    parser.add_argument("--n-estimators", dest="n_estimators", type=int,
            required=True)
    parser.add_argument("--sampling-ratio", dest="sampling_ratio", type=float,
            required=True)
    parser.add_argument("--n-jobs", dest="n_jobs", type=int,
            required=True)
    return parser.parse_args()


def save_model(model, filename):
    with open(filename, "wb") as f:
        pickle.dump(model, f)


def model_filename():
    t = str(time.time()).encode("utf-8")
    return hashlib.sha256(t).hexdigest() + ".pkl"


def pick(train_set, test_set, n_train, n_test):
    images_train, y_train = train_set
    images_test, y_test = test_set
    train_set = images_train[:n_train], y_train[:n_train]
    test_set = images_test[:n_test], y_test[:n_test]
    return train_set, test_set


def run_evaluation(datasize, estimator_params, ensemble_params):
    # params = vars(parse_args())

    pickle_dir = "pickles"

    n_train, n_test = datasize["n_train"], datasize["n_test"]

    train_set, test_set = load_mnist()
    train_set, test_set = pick(train_set, test_set, n_train, n_test)
    (images_train, y_train), (images_test, y_test) = train_set, test_set

    params = dict(
        list(estimator_params.items()) +
        list(ensemble_params.items())
    )

    params["n_train"] = n_train
    params["n_test"] = n_test

    model, accuracy, training_time = run_pcanet_normal(
        estimator_params,
        images_train, images_test, y_train, y_test
    )

    filename = model_filename()
    save_model(model, join(pickle_dir, filename))
    params["normal-model"] = filename
    params["normal-accuracy"] = accuracy
    params["normal-training-time"] = training_time

    model, accuracy, training_time = run_pcanet_ensemble(
        ensemble_params, estimator_params,
        images_train, images_test, y_train, y_test
    )

    filename = model_filename()
    save_model(model, join(pickle_dir, filename))
    params["ensemble-model"] = filename
    params["ensemble-accuracy"] = accuracy
    params["ensemble-training-time"] = training_time

    with open("result.json", "a") as f:
        json.dump(params, f, indent=2, sort_keys=True)


if __name__ == "__main__":
    datasize = {
        "n_train": None,
        "n_test": None
    }
    estimator_params = {
        "image_shape": 28,
        "filter_shape_l1": 4, "step_shape_l1": 2, "n_l1_output": 3,
        "filter_shape_l2": 4, "step_shape_l2": 1, "n_l2_output": 3,
        "block_shape": 5
    }
    ensemble_params = {
        "n_estimators": 10,
        "sampling_ratio": 0.3,
        "n_jobs": cpu_count()
    }

    sampling_ratio_list = [
        0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09,
        0.1,
    ]

    for sampling_ratio in np.arange(0.01, 0.11, 0.01).astype(float):
        for n_estimators in np.arange(10, 210, 10).astype(int):
            # convert from numpy object
            ensemble_params["n_estimators"] = int(n_estimators)
            ensemble_params["sampling_ratio"] = float(sampling_ratio)
            run_evaluation(datasize, estimator_params, ensemble_params)
