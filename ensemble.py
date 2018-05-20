from multiprocessing import cpu_count
from itertools import repeat

from sklearn.svm import SVC
from multiprocessing import Pool
from numpy.random import randint
from pcanet import PCANet
import numpy as np


def most_frequent_label(v):
    values, counts = np.unique(v, return_counts=True)
    return values[np.argmax(counts)]


def predict(transformer, estimator, images):
    X = transformer.transform(images)
    y = estimator.predict(X)
    return y


def fit(transformer, estimator, images, y):
    transformer.fit(images)
    X = transformer.transform(images)
    estimator.fit(X, y)
    return transformer, estimator


def fit_random(transformer, estimator, images, y, sampling_ratio):
    n_images = images.shape[0]
    n_samples = int(n_images * sampling_ratio)
    indices = randint(0, n_images, n_samples)
    return fit(transformer, estimator, images[indices], y[indices])


class Bagging(object):
    def __init__(self, n_estimators, sampling_ratio, n_jobs=-1,
                 **transformer_params):
        """
        n_estimators: int
            The number of estimators
        sampling_ratio: int
            The number of samples to draw from X to train each base transformer.
        n_jobs: int
            The number of jobs to run in parallel.
            The number of cores is set if -1.
        transformer_params: dict
            Parameters for PCANet.__init__
        """

        self.transformers = \
            [PCANet(**transformer_params) for i in range(n_estimators)]
        # Validate only the first transformer
        # since all of transformers share the same hyperparameters
        self.transformers[0].validate_structure()
        self.estimators = [SVC(C=1e8) for i in range(n_estimators)]

        self.sampling_ratio = sampling_ratio

        self.n_jobs = n_jobs
        if n_jobs == -1:
            self.n_jobs = cpu_count()

    def fit(self, images, y):
        # run fit_random in parallel
        g = zip(
            self.transformers,
            self.estimators,
            repeat(images),
            repeat(y),
            repeat(self.sampling_ratio)
        )
        with Pool(processes=self.n_jobs) as pool:
            t = pool.starmap(fit_random, g)
        self.transformers, self.estimators = zip(*t)
        return self

    def predict(self, images):
        # run transform(transformer, images) in parallel
        g = zip(self.transformers, self.estimators, repeat(images))
        with Pool(processes=self.n_jobs) as pool:
            Y = pool.starmap(predict, g)
        Y = np.array(Y)  # Y is of shape (n_estimators, n_classes)
        y = [most_frequent_label(Y[:, i]) for i in range(Y.shape[1])]
        return np.array(y)
