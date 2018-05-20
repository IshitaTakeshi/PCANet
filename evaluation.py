import os
from os.path import exists, join
import json
import gzip
from argparse import ArgumentParser
import hashlib
import time
import timeit
from urllib.request import urlopen
import tarfile

import numpy as np

from sklearn.svm import LinearSVC, SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

from pcanet import PCANet
from ensemble import Bagging
import utils


pickle_dir = "pickles"


def params_to_str(params):
    keys = sorted(params.keys())
    return "_".join([key + "_" + str(params[key]) for key in keys])


def run_classifier(X_train, X_test, y_train, y_test):
    model = SVC(C=10)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return y_test, y_pred


def run_pcanet_normal(transformer_params,
                      images_train, images_test, y_train, y_test):
    model = PCANet(**transformer_params)
    model.validate_structure()

    t1 = timeit.default_timer()
    model.fit(images_train)
    t2 = timeit.default_timer()
    train_time = t2 - t1

    t1 = timeit.default_timer()
    X_train = model.transform(images_train)
    t2 = timeit.default_timer()
    transform_time = t2 - t1
    X_test = model.transform(images_test)

    y_test, y_pred = run_classifier(X_train, X_test, y_train, y_test)
    accuracy = accuracy_score(y_test, y_pred)

    return model, accuracy, train_time, transform_time


def run_pcanet_ensemble(ensemble_params, transformer_params,
                        images_train, images_test, y_train, y_test):
    model = Bagging(
        ensemble_params["n_estimators"],
        ensemble_params["sampling_ratio"],
        ensemble_params["n_jobs"],
        **transformer_params)

    t1 = timeit.default_timer()
    model.fit(images_train, y_train)
    t2 = timeit.default_timer()
    train_time = t2 - t1

    t1 = timeit.default_timer()
    y_pred = model.predict(images_test)
    t2 = timeit.default_timer()
    predict_time = t2 - t1

    accuracy = accuracy_score(y_test, y_pred)

    return model, accuracy, train_time, predict_time


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
    parser.add_argument("--filter-shape-pooling", dest="filter_shape_pooling", type=int,
            required=True)
    parser.add_argument("--step-shape-pooling", dest="step_shape_pooling", type=int,
            required=True)
    parser.add_argument("--n-estimators", dest="n_estimators", type=int,
            required=True)
    parser.add_argument("--sampling-ratio", dest="sampling_ratio", type=float,
            required=True)
    parser.add_argument("--n-jobs", dest="n_jobs", type=int,
            required=True)
    return parser.parse_args()


def model_filename():
    t = str(time.time()).encode("utf-8")
    return hashlib.sha256(t).hexdigest() + ".pkl"


def evaluate_ensemble(train_set, test_set,
                      ensemble_params, transformer_params):
    (images_train, y_train), (images_test, y_test) = train_set, test_set

    model, accuracy, train_time, predict_time = run_pcanet_ensemble(
        ensemble_params, transformer_params,
        images_train, images_test, y_train, y_test
    )

    filename = model_filename()
    utils.save_model(model, join(pickle_dir, filename))

    params = {}
    params["ensemble-model"] = filename
    params["ensemble-accuracy"] = accuracy
    params["ensemble-train-time"] = train_time
    params["ensemble-predict-time"] = predict_time
    return params


def evaluate_normal(train_set, test_set, transformer_params):
    (images_train, y_train), (images_test, y_test) = train_set, test_set

    model, accuracy, train_time, transform_time = run_pcanet_normal(
        transformer_params,
        images_train, images_test, y_train, y_test
    )

    filename = model_filename()
    utils.save_model(model, join(pickle_dir, filename))

    params = {}
    params["normal-model"] = filename
    params["normal-accuracy"] = accuracy
    params["normal-train-time"] = train_time
    params["normal-transform-time"] = transform_time
    return params


def export_json(result, filename):
    with open(filename, "a") as f:
        json.dump(result, f, sort_keys=True, indent=2)


def run(dataset, datasize, transformer_params, ensemble_params,
        model_type, filename="result.json"):
    train_set, test_set = dataset

    train_set, test_set = utils.pick(
        train_set, test_set,
        datasize["n_train"], datasize["n_test"]
    )

    # Set the actual data size
    datasize["n_train"], datasize["n_test"] = len(train_set[1]), len(test_set[1])

    if model_type == "normal":
        result = evaluate_normal(train_set, test_set, transformer_params)
    elif model_type == "ensemble":
        result = evaluate_ensemble(train_set, test_set,
                                   ensemble_params, transformer_params)
    else:
        raise ValueError("Invalid model type '{}'".format(model_type))

    params = utils.concatenate_dicts(
        datasize,
        transformer_params,
        ensemble_params,
        result
    )

    params["model-type"] = model_type

    export_json(params, filename)
    print(json.dumps(params, sort_keys=True))


def run_cifar(n_train=None, n_test=None, model_type="normal"):
    datasize = {"n_train": n_train, "n_test": n_test}
    transformer_params = {
        "image_shape": 32,
        "filter_shape_l1": 5, "step_shape_l1": 1, "n_l1_output": 16,
        "filter_shape_l2": 5, "step_shape_l2": 1, "n_l2_output": 8,
        "filter_shape_pooling": 8, "step_shape_pooling": 4
    }
    ensemble_params = {
        "n_estimators" : 10,
        "sampling_ratio" : 0.1,
        "n_jobs" : -1
    }
    dataset = utils.load_cifar()
    run(dataset, datasize, transformer_params, ensemble_params, model_type)


def run_mnist(n_train=None, n_test=None, model_type="normal"):
    datasize = {"n_train": n_train, "n_test": n_test}
    transformer_params = {
        "image_shape": 28,
        "filter_shape_l1": 5, "step_shape_l1": 1, "n_l1_output": 8,
        "filter_shape_l2": 5, "step_shape_l2": 1, "n_l2_output": 4,
        "filter_shape_pooling": 5, "step_shape_pooling": 5
    }
    ensemble_params = {
        "n_estimators" : 40,
        "sampling_ratio" : 0.03,
        "n_jobs" : -1
    }
    dataset = utils.load_mnist()
    run(dataset, datasize, transformer_params, ensemble_params, model_type)


if __name__ == "__main__":
    print("MNIST")
    run_mnist(n_train=100, n_test=100, model_type="ensemble")
    # print("CIFAR")
    # run_cifar(n_train=None, n_test=None, model_type="normal")
